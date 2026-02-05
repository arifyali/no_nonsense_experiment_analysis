"""
Data preprocessing functionality for experimental data analysis.

This module provides the Preprocessor class for normalizing, encoding,
and transforming pandas DataFrames while preserving original data integrity.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Callable, Literal
from ..core.exceptions import DataValidationError


class Preprocessor:
    """Transforms data for experimental analysis.

    All operations return new DataFrames and never modify the original data,
    ensuring data integrity preservation throughout the preprocessing pipeline.

    Attributes:
        strict_mode: If True, raises exceptions on invalid inputs.
    """

    # Supported normalization methods
    NORMALIZATION_METHODS = ["minmax", "zscore", "robust", "maxabs"]

    # Supported encoding methods
    ENCODING_METHODS = ["onehot", "label", "ordinal", "binary"]

    def __init__(self, strict_mode: bool = True):
        """Initialize the preprocessor.

        Args:
            strict_mode: If True, raises exceptions on validation failures.
        """
        self.strict_mode = strict_mode
        # Store fitted parameters for inverse transforms
        self._normalization_params: Dict[str, Dict[str, Any]] = {}
        self._encoding_params: Dict[str, Dict[str, Any]] = {}

    def normalize_columns(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = "minmax",
        **kwargs
    ) -> pd.DataFrame:
        """Normalize specified columns using the given method.

        Applies normalization to transform column values to a standard scale.
        The original DataFrame is never modified.

        Args:
            df: Input DataFrame to transform
            columns: List of column names to normalize
            method: Normalization method to use:
                - "minmax": Scale to [0, 1] range (default)
                - "zscore": Standardize to mean=0, std=1
                - "robust": Scale using median and IQR (robust to outliers)
                - "maxabs": Scale by maximum absolute value to [-1, 1]
            **kwargs: Additional arguments for specific methods:
                - feature_range: Tuple (min, max) for minmax scaling (default: (0, 1))

        Returns:
            New DataFrame with normalized columns

        Raises:
            DataValidationError: If input validation fails or invalid method specified
        """
        # Validate input type
        if not isinstance(df, pd.DataFrame):
            raise DataValidationError(
                f"Input must be a pandas DataFrame, got {type(df).__name__}",
                context={"operation": "normalize_columns"}
            )

        # Validate method
        if method not in self.NORMALIZATION_METHODS:
            raise DataValidationError(
                f"Invalid normalization method '{method}'. Must be one of: {self.NORMALIZATION_METHODS}",
                context={"operation": "normalize_columns", "method": method}
            )

        # Validate columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise DataValidationError(
                f"Columns not found in DataFrame: {missing_cols}",
                context={"operation": "normalize_columns", "missing_columns": missing_cols}
            )

        # Validate columns are numeric
        non_numeric = [col for col in columns if not pd.api.types.is_numeric_dtype(df[col])]
        if non_numeric:
            raise DataValidationError(
                f"Cannot normalize non-numeric columns: {non_numeric}",
                context={"operation": "normalize_columns", "non_numeric_columns": non_numeric}
            )

        # Create a copy to preserve original
        result = df.copy()

        for col in columns:
            col_data = result[col].values.astype(float)

            if method == "minmax":
                feature_range = kwargs.get("feature_range", (0, 1))
                col_min = np.nanmin(col_data)
                col_max = np.nanmax(col_data)

                if col_max - col_min == 0:
                    # Constant column - set to middle of range
                    normalized = np.full_like(col_data, (feature_range[0] + feature_range[1]) / 2)
                else:
                    normalized = (col_data - col_min) / (col_max - col_min)
                    normalized = normalized * (feature_range[1] - feature_range[0]) + feature_range[0]

                self._normalization_params[col] = {
                    "method": method,
                    "min": col_min,
                    "max": col_max,
                    "feature_range": feature_range
                }

            elif method == "zscore":
                col_mean = np.nanmean(col_data)
                col_std = np.nanstd(col_data)

                if col_std == 0:
                    normalized = np.zeros_like(col_data)
                else:
                    normalized = (col_data - col_mean) / col_std

                self._normalization_params[col] = {
                    "method": method,
                    "mean": col_mean,
                    "std": col_std
                }

            elif method == "robust":
                col_median = np.nanmedian(col_data)
                q1 = np.nanpercentile(col_data, 25)
                q3 = np.nanpercentile(col_data, 75)
                iqr = q3 - q1

                if iqr == 0:
                    normalized = np.zeros_like(col_data)
                else:
                    normalized = (col_data - col_median) / iqr

                self._normalization_params[col] = {
                    "method": method,
                    "median": col_median,
                    "iqr": iqr
                }

            elif method == "maxabs":
                col_maxabs = np.nanmax(np.abs(col_data))

                if col_maxabs == 0:
                    normalized = col_data
                else:
                    normalized = col_data / col_maxabs

                self._normalization_params[col] = {
                    "method": method,
                    "maxabs": col_maxabs
                }

            result[col] = normalized

        return result

    def encode_categorical(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = "onehot",
        **kwargs
    ) -> pd.DataFrame:
        """Encode categorical variables using the specified method.

        Transforms categorical columns into numeric representations.
        The original DataFrame is never modified.

        Args:
            df: Input DataFrame to transform
            columns: List of categorical column names to encode
            method: Encoding method to use:
                - "onehot": One-hot encoding (creates binary columns for each category)
                - "label": Label encoding (assigns integer to each category)
                - "ordinal": Ordinal encoding (requires ordering in kwargs)
                - "binary": Binary encoding (uses binary representation)
            **kwargs: Additional arguments for specific methods:
                - drop_first: bool - Drop first category in onehot (default: False)
                - ordering: Dict[str, List] - Category ordering for ordinal encoding

        Returns:
            New DataFrame with encoded columns

        Raises:
            DataValidationError: If input validation fails or invalid method specified
        """
        # Validate input type
        if not isinstance(df, pd.DataFrame):
            raise DataValidationError(
                f"Input must be a pandas DataFrame, got {type(df).__name__}",
                context={"operation": "encode_categorical"}
            )

        # Validate method
        if method not in self.ENCODING_METHODS:
            raise DataValidationError(
                f"Invalid encoding method '{method}'. Must be one of: {self.ENCODING_METHODS}",
                context={"operation": "encode_categorical", "method": method}
            )

        # Validate columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise DataValidationError(
                f"Columns not found in DataFrame: {missing_cols}",
                context={"operation": "encode_categorical", "missing_columns": missing_cols}
            )

        # Create a copy to preserve original
        result = df.copy()

        if method == "onehot":
            drop_first = kwargs.get("drop_first", False)
            prefix = kwargs.get("prefix", None)

            for col in columns:
                # Get dummies for this column
                dummies = pd.get_dummies(
                    result[col],
                    prefix=prefix if prefix else col,
                    drop_first=drop_first,
                    dtype=int
                )

                # Store encoding parameters
                self._encoding_params[col] = {
                    "method": method,
                    "categories": result[col].unique().tolist(),
                    "columns": dummies.columns.tolist()
                }

                # Drop original column and add dummies
                result = result.drop(columns=[col])
                result = pd.concat([result, dummies], axis=1)

        elif method == "label":
            for col in columns:
                categories = result[col].unique()
                category_map = {cat: i for i, cat in enumerate(sorted(categories, key=str))}

                self._encoding_params[col] = {
                    "method": method,
                    "category_map": category_map
                }

                result[col] = result[col].map(category_map)

        elif method == "ordinal":
            ordering = kwargs.get("ordering", {})

            for col in columns:
                if col in ordering:
                    category_order = ordering[col]
                    category_map = {cat: i for i, cat in enumerate(category_order)}
                else:
                    # Use natural ordering if not specified
                    categories = sorted(result[col].unique(), key=str)
                    category_map = {cat: i for i, cat in enumerate(categories)}

                self._encoding_params[col] = {
                    "method": method,
                    "category_map": category_map
                }

                # Handle unknown categories
                result[col] = result[col].map(lambda x: category_map.get(x, -1))

        elif method == "binary":
            for col in columns:
                categories = sorted(result[col].unique(), key=str)
                n_categories = len(categories)
                n_bits = max(1, int(np.ceil(np.log2(n_categories + 1))))

                category_map = {cat: i for i, cat in enumerate(categories)}

                self._encoding_params[col] = {
                    "method": method,
                    "category_map": category_map,
                    "n_bits": n_bits
                }

                # Convert to numpy array for bitwise operations (pandas 2.x compatibility)
                int_encoded = result[col].map(category_map).values.astype(np.int64)

                # Create binary columns using numpy bitwise operations
                for bit in range(n_bits):
                    result[f"{col}_bit{bit}"] = ((int_encoded >> bit) & 1).astype(int)

                # Remove original column
                result = result.drop(columns=[col])

        return result

    def create_features(
        self,
        df: pd.DataFrame,
        feature_specs: Dict[str, Any]
    ) -> pd.DataFrame:
        """Create new features based on specifications.

        Creates derived features from existing columns using various operations.
        The original DataFrame is never modified.

        Args:
            df: Input DataFrame to transform
            feature_specs: Dictionary specifying features to create. Supported formats:
                - {"new_col": {"operation": "sum", "columns": ["a", "b"]}}
                - {"new_col": {"operation": "product", "columns": ["a", "b"]}}
                - {"new_col": {"operation": "ratio", "numerator": "a", "denominator": "b"}}
                - {"new_col": {"operation": "difference", "columns": ["a", "b"]}}
                - {"new_col": {"operation": "log", "column": "a"}}
                - {"new_col": {"operation": "sqrt", "column": "a"}}
                - {"new_col": {"operation": "power", "column": "a", "exponent": 2}}
                - {"new_col": {"operation": "bin", "column": "a", "bins": 5}}
                - {"new_col": {"operation": "interaction", "columns": ["a", "b"]}}
                - {"new_col": {"operation": "custom", "func": callable, "columns": ["a", "b"]}}

        Returns:
            New DataFrame with additional feature columns

        Raises:
            DataValidationError: If input validation fails or invalid specs provided
        """
        # Validate input type
        if not isinstance(df, pd.DataFrame):
            raise DataValidationError(
                f"Input must be a pandas DataFrame, got {type(df).__name__}",
                context={"operation": "create_features"}
            )

        if not isinstance(feature_specs, dict):
            raise DataValidationError(
                f"feature_specs must be a dictionary, got {type(feature_specs).__name__}",
                context={"operation": "create_features"}
            )

        # Create a copy to preserve original
        result = df.copy()

        for new_col, spec in feature_specs.items():
            if not isinstance(spec, dict):
                raise DataValidationError(
                    f"Feature specification for '{new_col}' must be a dictionary",
                    context={"operation": "create_features", "feature": new_col}
                )

            if "operation" not in spec:
                raise DataValidationError(
                    f"Feature specification for '{new_col}' must include 'operation'",
                    context={"operation": "create_features", "feature": new_col}
                )

            operation = spec["operation"]

            try:
                if operation == "sum":
                    cols = spec.get("columns", [])
                    self._validate_columns_exist(df, cols, operation, new_col)
                    result[new_col] = df[cols].sum(axis=1)

                elif operation == "product":
                    cols = spec.get("columns", [])
                    self._validate_columns_exist(df, cols, operation, new_col)
                    result[new_col] = df[cols].prod(axis=1)

                elif operation == "ratio":
                    num_col = spec.get("numerator")
                    den_col = spec.get("denominator")
                    self._validate_columns_exist(df, [num_col, den_col], operation, new_col)
                    # Handle division by zero
                    result[new_col] = df[num_col] / df[den_col].replace(0, np.nan)

                elif operation == "difference":
                    cols = spec.get("columns", [])
                    self._validate_columns_exist(df, cols, operation, new_col)
                    if len(cols) >= 2:
                        result[new_col] = df[cols[0]] - df[cols[1]]
                    else:
                        raise DataValidationError(
                            f"Difference operation requires at least 2 columns",
                            context={"operation": "create_features", "feature": new_col}
                        )

                elif operation == "log":
                    col = spec.get("column")
                    self._validate_columns_exist(df, [col], operation, new_col)
                    # Use log1p to handle zeros, or log with small offset
                    base = spec.get("base", "natural")
                    if base == "natural":
                        result[new_col] = np.log(df[col].clip(lower=1e-10))
                    elif base == "10":
                        result[new_col] = np.log10(df[col].clip(lower=1e-10))
                    elif base == "2":
                        result[new_col] = np.log2(df[col].clip(lower=1e-10))

                elif operation == "sqrt":
                    col = spec.get("column")
                    self._validate_columns_exist(df, [col], operation, new_col)
                    result[new_col] = np.sqrt(df[col].clip(lower=0))

                elif operation == "power":
                    col = spec.get("column")
                    exponent = spec.get("exponent", 2)
                    self._validate_columns_exist(df, [col], operation, new_col)
                    result[new_col] = np.power(df[col], exponent)

                elif operation == "bin":
                    col = spec.get("column")
                    bins = spec.get("bins", 5)
                    labels = spec.get("labels", None)
                    self._validate_columns_exist(df, [col], operation, new_col)
                    result[new_col] = pd.cut(df[col], bins=bins, labels=labels)

                elif operation == "interaction":
                    cols = spec.get("columns", [])
                    self._validate_columns_exist(df, cols, operation, new_col)
                    # Multiply all columns together
                    result[new_col] = df[cols].prod(axis=1)

                elif operation == "custom":
                    func = spec.get("func")
                    cols = spec.get("columns", [])
                    if not callable(func):
                        raise DataValidationError(
                            f"Custom operation requires a callable 'func'",
                            context={"operation": "create_features", "feature": new_col}
                        )
                    self._validate_columns_exist(df, cols, operation, new_col)
                    result[new_col] = df[cols].apply(func, axis=1)

                else:
                    raise DataValidationError(
                        f"Unknown operation '{operation}' for feature '{new_col}'",
                        context={"operation": "create_features", "feature": new_col}
                    )

            except DataValidationError:
                raise
            except Exception as e:
                raise DataValidationError(
                    f"Error creating feature '{new_col}': {str(e)}",
                    context={"operation": "create_features", "feature": new_col}
                )

        return result

    def _validate_columns_exist(
        self,
        df: pd.DataFrame,
        columns: List[str],
        operation: str,
        feature: str
    ) -> None:
        """Validate that columns exist in DataFrame.

        Args:
            df: DataFrame to check
            columns: List of column names
            operation: Name of operation being performed
            feature: Name of feature being created

        Raises:
            DataValidationError: If columns are missing
        """
        if not columns:
            raise DataValidationError(
                f"No columns specified for {operation} operation",
                context={"operation": "create_features", "feature": feature}
            )

        missing = [col for col in columns if col and col not in df.columns]
        if missing:
            raise DataValidationError(
                f"Columns not found for feature '{feature}': {missing}",
                context={"operation": "create_features", "feature": feature, "missing": missing}
            )

    def get_normalization_params(self) -> Dict[str, Dict[str, Any]]:
        """Get stored normalization parameters.

        Returns:
            Dictionary mapping column names to their normalization parameters
        """
        return self._normalization_params.copy()

    def get_encoding_params(self) -> Dict[str, Dict[str, Any]]:
        """Get stored encoding parameters.

        Returns:
            Dictionary mapping column names to their encoding parameters
        """
        return self._encoding_params.copy()

    def inverse_normalize(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Inverse normalize columns using stored parameters.

        Args:
            df: DataFrame with normalized columns
            columns: Columns to inverse normalize. If None, uses all stored params.

        Returns:
            DataFrame with original scale restored

        Raises:
            DataValidationError: If no normalization params stored for column
        """
        if not isinstance(df, pd.DataFrame):
            raise DataValidationError(
                f"Input must be a pandas DataFrame, got {type(df).__name__}",
                context={"operation": "inverse_normalize"}
            )

        result = df.copy()
        target_cols = columns if columns else list(self._normalization_params.keys())

        for col in target_cols:
            if col not in self._normalization_params:
                raise DataValidationError(
                    f"No normalization parameters stored for column '{col}'",
                    context={"operation": "inverse_normalize", "column": col}
                )

            if col not in df.columns:
                continue

            params = self._normalization_params[col]
            method = params["method"]
            col_data = result[col].values.astype(float)

            if method == "minmax":
                feature_range = params["feature_range"]
                col_min, col_max = params["min"], params["max"]
                # Reverse the minmax transformation
                normalized = (col_data - feature_range[0]) / (feature_range[1] - feature_range[0])
                original = normalized * (col_max - col_min) + col_min

            elif method == "zscore":
                col_mean, col_std = params["mean"], params["std"]
                original = col_data * col_std + col_mean

            elif method == "robust":
                col_median, iqr = params["median"], params["iqr"]
                original = col_data * iqr + col_median

            elif method == "maxabs":
                maxabs = params["maxabs"]
                original = col_data * maxabs

            result[col] = original

        return result
