"""
Data cleaning functionality for experimental data analysis.

This module provides the DataCleaner class for handling missing values,
duplicates, and outliers in pandas DataFrames while preserving original data integrity.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Literal
from ..core.exceptions import DataValidationError


class DataCleaner:
    """Handles missing values, duplicates, and outliers in DataFrames.

    All operations return new DataFrames and never modify the original data,
    ensuring data integrity preservation throughout the cleaning pipeline.

    Attributes:
        strict_mode: If True, raises exceptions on invalid inputs.
                    If False, returns None with warnings for some operations.
    """

    # Supported strategies for missing value handling
    MISSING_VALUE_STRATEGIES = ["drop", "fill_mean", "fill_median", "fill_mode", "fill_forward", "fill_backward", "fill_value"]

    # Supported methods for outlier detection
    OUTLIER_METHODS = ["iqr", "zscore", "modified_zscore"]

    def __init__(self, strict_mode: bool = True):
        """Initialize the data cleaner.

        Args:
            strict_mode: If True, raises exceptions on validation failures.
                        If False, returns None for some invalid operations.
        """
        self.strict_mode = strict_mode

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = "drop",
        columns: Optional[List[str]] = None,
        fill_value: Optional[Any] = None
    ) -> pd.DataFrame:
        """Handle missing values with specified strategy.

        Applies the specified strategy to handle missing values in the DataFrame.
        The original DataFrame is never modified.

        Args:
            df: Input DataFrame to clean
            strategy: Strategy for handling missing values. Options:
                - "drop": Remove rows with missing values
                - "fill_mean": Fill with column mean (numeric columns only)
                - "fill_median": Fill with column median (numeric columns only)
                - "fill_mode": Fill with column mode (most frequent value)
                - "fill_forward": Forward fill (propagate last valid value)
                - "fill_backward": Backward fill (use next valid value)
                - "fill_value": Fill with a specified value
            columns: List of columns to apply strategy to. If None, applies to all columns.
            fill_value: Value to use when strategy is "fill_value"

        Returns:
            New DataFrame with missing values handled

        Raises:
            DataValidationError: If input validation fails or invalid strategy specified
        """
        # Validate input type
        if not isinstance(df, pd.DataFrame):
            raise DataValidationError(
                f"Input must be a pandas DataFrame, got {type(df).__name__}",
                context={"operation": "handle_missing_values"}
            )

        # Validate strategy
        if strategy not in self.MISSING_VALUE_STRATEGIES:
            raise DataValidationError(
                f"Invalid strategy '{strategy}'. Must be one of: {self.MISSING_VALUE_STRATEGIES}",
                context={"operation": "handle_missing_values", "strategy": strategy}
            )

        # Validate fill_value is provided when needed
        if strategy == "fill_value" and fill_value is None:
            raise DataValidationError(
                "fill_value must be specified when using 'fill_value' strategy",
                context={"operation": "handle_missing_values", "strategy": strategy}
            )

        # Create a copy to preserve original
        result = df.copy()

        # Determine which columns to process
        if columns is not None:
            # Validate columns exist
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                raise DataValidationError(
                    f"Columns not found in DataFrame: {missing_cols}",
                    context={"operation": "handle_missing_values", "missing_columns": missing_cols}
                )
            target_columns = columns
        else:
            target_columns = list(df.columns)

        # Apply strategy
        if strategy == "drop":
            result = result.dropna(subset=target_columns)

        elif strategy == "fill_mean":
            for col in target_columns:
                if pd.api.types.is_numeric_dtype(result[col]):
                    result[col] = result[col].fillna(result[col].mean())

        elif strategy == "fill_median":
            for col in target_columns:
                if pd.api.types.is_numeric_dtype(result[col]):
                    result[col] = result[col].fillna(result[col].median())

        elif strategy == "fill_mode":
            for col in target_columns:
                mode_values = result[col].mode()
                if len(mode_values) > 0:
                    result[col] = result[col].fillna(mode_values.iloc[0])

        elif strategy == "fill_forward":
            result[target_columns] = result[target_columns].ffill()

        elif strategy == "fill_backward":
            result[target_columns] = result[target_columns].bfill()

        elif strategy == "fill_value":
            result[target_columns] = result[target_columns].fillna(fill_value)

        return result

    def remove_duplicates(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
        keep: Literal["first", "last", False] = "first"
    ) -> pd.DataFrame:
        """Remove duplicate records from DataFrame.

        Removes duplicate rows based on specified columns or all columns.
        The original DataFrame is never modified.

        Args:
            df: Input DataFrame to clean
            subset: List of column names to consider for identifying duplicates.
                   If None, all columns are used.
            keep: Which duplicate to keep:
                - "first": Keep first occurrence
                - "last": Keep last occurrence
                - False: Remove all duplicates

        Returns:
            New DataFrame with duplicates removed

        Raises:
            DataValidationError: If input validation fails
        """
        # Validate input type
        if not isinstance(df, pd.DataFrame):
            raise DataValidationError(
                f"Input must be a pandas DataFrame, got {type(df).__name__}",
                context={"operation": "remove_duplicates"}
            )

        # Validate subset columns exist
        if subset is not None:
            missing_cols = [col for col in subset if col not in df.columns]
            if missing_cols:
                raise DataValidationError(
                    f"Columns not found in DataFrame: {missing_cols}",
                    context={"operation": "remove_duplicates", "missing_columns": missing_cols}
                )

        # Validate keep parameter
        valid_keep_values = ["first", "last", False]
        if keep not in valid_keep_values:
            raise DataValidationError(
                f"Invalid keep value '{keep}'. Must be one of: {valid_keep_values}",
                context={"operation": "remove_duplicates", "keep": keep}
            )

        # Remove duplicates on a copy
        result = df.drop_duplicates(subset=subset, keep=keep)

        return result

    def detect_outliers(
        self,
        df: pd.DataFrame,
        method: str = "iqr",
        columns: Optional[List[str]] = None,
        threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """Detect outliers using specified method.

        Identifies outliers in numeric columns using statistical methods.
        Returns a boolean DataFrame where True indicates an outlier.

        Args:
            df: Input DataFrame to analyze
            method: Method for outlier detection:
                - "iqr": Interquartile Range method (values outside 1.5*IQR)
                - "zscore": Z-score method (|z| > threshold, default 3)
                - "modified_zscore": Modified Z-score using MAD (|z| > threshold, default 3.5)
            columns: List of columns to check for outliers. If None, checks all numeric columns.
            threshold: Threshold for zscore/modified_zscore methods.
                      Default is 3 for zscore, 3.5 for modified_zscore.

        Returns:
            Boolean DataFrame with same shape as input (for selected columns),
            where True indicates an outlier

        Raises:
            DataValidationError: If input validation fails or invalid method specified
        """
        # Validate input type
        if not isinstance(df, pd.DataFrame):
            raise DataValidationError(
                f"Input must be a pandas DataFrame, got {type(df).__name__}",
                context={"operation": "detect_outliers"}
            )

        # Validate method
        if method not in self.OUTLIER_METHODS:
            raise DataValidationError(
                f"Invalid method '{method}'. Must be one of: {self.OUTLIER_METHODS}",
                context={"operation": "detect_outliers", "method": method}
            )

        # Determine columns to check
        if columns is not None:
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                raise DataValidationError(
                    f"Columns not found in DataFrame: {missing_cols}",
                    context={"operation": "detect_outliers", "missing_columns": missing_cols}
                )
            target_columns = columns
        else:
            # Use only numeric columns
            target_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        # Filter to only numeric columns from target
        numeric_columns = [col for col in target_columns if pd.api.types.is_numeric_dtype(df[col])]

        if not numeric_columns:
            # Return empty boolean DataFrame if no numeric columns
            return pd.DataFrame(index=df.index)

        # Initialize result DataFrame
        outliers = pd.DataFrame(False, index=df.index, columns=numeric_columns)

        if method == "iqr":
            for col in numeric_columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers[col] = (df[col] < lower_bound) | (df[col] > upper_bound)

        elif method == "zscore":
            z_threshold = threshold if threshold is not None else 3.0
            for col in numeric_columns:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    z_scores = np.abs((df[col] - mean) / std)
                    outliers[col] = z_scores > z_threshold

        elif method == "modified_zscore":
            # Modified Z-score using Median Absolute Deviation (MAD)
            z_threshold = threshold if threshold is not None else 3.5
            for col in numeric_columns:
                median = df[col].median()
                mad = np.median(np.abs(df[col] - median))
                if mad > 0:
                    # 0.6745 is the constant for normal distribution
                    modified_z = 0.6745 * (df[col] - median) / mad
                    outliers[col] = np.abs(modified_z) > z_threshold

        return outliers

    def remove_outliers(
        self,
        df: pd.DataFrame,
        method: str = "iqr",
        columns: Optional[List[str]] = None,
        threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """Remove rows containing outliers.

        Identifies and removes rows that contain outlier values in the specified columns.
        The original DataFrame is never modified.

        Args:
            df: Input DataFrame to clean
            method: Method for outlier detection (see detect_outliers for options)
            columns: List of columns to check for outliers
            threshold: Threshold for detection methods

        Returns:
            New DataFrame with outlier rows removed

        Raises:
            DataValidationError: If input validation fails
        """
        outlier_mask = self.detect_outliers(df, method=method, columns=columns, threshold=threshold)

        if outlier_mask.empty:
            return df.copy()

        # Remove rows where any column has an outlier
        rows_with_outliers = outlier_mask.any(axis=1)
        result = df[~rows_with_outliers].copy()

        return result

    def get_cleaning_summary(
        self,
        original_df: pd.DataFrame,
        cleaned_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate a summary of cleaning operations.

        Compares original and cleaned DataFrames to provide a summary of changes.

        Args:
            original_df: Original DataFrame before cleaning
            cleaned_df: Cleaned DataFrame after operations

        Returns:
            Dictionary with cleaning summary including rows removed, etc.
        """
        summary = {
            "original_rows": len(original_df),
            "cleaned_rows": len(cleaned_df),
            "rows_removed": len(original_df) - len(cleaned_df),
            "removal_percentage": round((1 - len(cleaned_df) / len(original_df)) * 100, 2) if len(original_df) > 0 else 0,
            "original_missing_values": original_df.isnull().sum().sum(),
            "cleaned_missing_values": cleaned_df.isnull().sum().sum(),
            "original_columns": list(original_df.columns),
            "cleaned_columns": list(cleaned_df.columns),
        }

        return summary
