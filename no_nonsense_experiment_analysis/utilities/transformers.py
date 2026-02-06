"""
Data transformation utility functions for experimental analysis.

This module provides data manipulation utilities for pivoting, aggregating,
and summarizing experimental data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Callable, Literal


class DataTransformers:
    """Data manipulation utilities for experimental analysis.

    This class provides static methods for common data transformations
    used in experimental data analysis.
    """

    @staticmethod
    def pivot_experimental_data(
        df: pd.DataFrame,
        index: Union[str, List[str]],
        columns: str,
        values: str,
        aggfunc: Union[str, Callable] = "mean",
        fill_value: Optional[Any] = None
    ) -> pd.DataFrame:
        """Pivot experimental data for analysis.

        Args:
            df: Input DataFrame
            index: Column(s) to use as row index
            columns: Column to use for new column headers
            values: Column to aggregate
            aggfunc: Aggregation function ('mean', 'sum', 'count', etc.)
            fill_value: Value to fill missing cells

        Returns:
            Pivoted DataFrame

        Raises:
            ValueError: If required columns are not in DataFrame
        """
        # Validate columns exist
        required_cols = [columns, values]
        if isinstance(index, list):
            required_cols.extend(index)
        else:
            required_cols.append(index)

        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")

        return pd.pivot_table(
            df,
            index=index,
            columns=columns,
            values=values,
            aggfunc=aggfunc,
            fill_value=fill_value
        )

    @staticmethod
    def aggregate_by_groups(
        df: pd.DataFrame,
        group_cols: Union[str, List[str]],
        value_cols: Optional[Union[str, List[str]]] = None,
        agg_funcs: Union[str, List[str], Dict[str, Union[str, List[str]]]] = "mean"
    ) -> pd.DataFrame:
        """Aggregate data by specified grouping columns.

        Args:
            df: Input DataFrame
            group_cols: Column(s) to group by
            value_cols: Column(s) to aggregate. If None, aggregates all numeric columns.
            agg_funcs: Aggregation function(s) to apply:
                - String: Apply same function to all columns ('mean', 'sum', etc.)
                - List: Apply multiple functions to all columns
                - Dict: Map column names to function(s)

        Returns:
            Aggregated DataFrame

        Raises:
            ValueError: If required columns are not in DataFrame
        """
        # Convert to list if needed
        if isinstance(group_cols, str):
            group_cols = [group_cols]

        missing = [col for col in group_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Grouping columns not found: {missing}")

        grouped = df.groupby(group_cols)

        if value_cols is None:
            # Use all numeric columns except group columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            value_cols = [col for col in numeric_cols if col not in group_cols]
        elif isinstance(value_cols, str):
            value_cols = [value_cols]

        missing_vals = [col for col in value_cols if col not in df.columns]
        if missing_vals:
            raise ValueError(f"Value columns not found: {missing_vals}")

        if isinstance(agg_funcs, dict):
            result = grouped[value_cols].agg(agg_funcs)
        else:
            result = grouped[value_cols].agg(agg_funcs)

        return result.reset_index()

    @staticmethod
    def calculate_summary_statistics(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        include_percentiles: bool = True,
        include_missing: bool = True
    ) -> Dict[str, Any]:
        """Calculate comprehensive summary statistics.

        Args:
            df: Input DataFrame
            columns: Columns to summarize (defaults to all numeric)
            include_percentiles: Whether to include percentile values
            include_missing: Whether to include missing value counts

        Returns:
            Dictionary with summary statistics for each column
        """
        if columns:
            target_df = df[columns].select_dtypes(include=[np.number])
        else:
            target_df = df.select_dtypes(include=[np.number])

        summary = {}

        for col in target_df.columns:
            series = target_df[col]
            col_stats = {
                'count': int(series.count()),
                'mean': float(series.mean()),
                'std': float(series.std()),
                'min': float(series.min()),
                'max': float(series.max()),
                'median': float(series.median()),
                'skewness': float(series.skew()),
                'kurtosis': float(series.kurtosis()),
            }

            if include_percentiles:
                col_stats['percentile_25'] = float(series.quantile(0.25))
                col_stats['percentile_75'] = float(series.quantile(0.75))
                col_stats['iqr'] = col_stats['percentile_75'] - col_stats['percentile_25']

            if include_missing:
                col_stats['missing_count'] = int(series.isnull().sum())
                col_stats['missing_percent'] = float(series.isnull().mean() * 100)

            summary[col] = col_stats

        # Add overall summary
        summary['_overall'] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'numeric_columns': len(target_df.columns),
            'memory_usage_bytes': int(df.memory_usage(deep=True).sum())
        }

        return summary

    @staticmethod
    def melt_wide_to_long(
        df: pd.DataFrame,
        id_vars: Union[str, List[str]],
        value_vars: Optional[List[str]] = None,
        var_name: str = "variable",
        value_name: str = "value"
    ) -> pd.DataFrame:
        """Convert wide-format data to long-format.

        Args:
            df: Input DataFrame in wide format
            id_vars: Column(s) to use as identifier variables
            value_vars: Columns to unpivot. If None, uses all non-id columns.
            var_name: Name for the new variable column
            value_name: Name for the new value column

        Returns:
            DataFrame in long format
        """
        return pd.melt(
            df,
            id_vars=id_vars,
            value_vars=value_vars,
            var_name=var_name,
            value_name=value_name
        )

    @staticmethod
    def calculate_group_differences(
        df: pd.DataFrame,
        group_col: str,
        value_col: str,
        baseline_group: Optional[str] = None
    ) -> pd.DataFrame:
        """Calculate differences between groups relative to a baseline.

        Args:
            df: Input DataFrame
            group_col: Column containing group labels
            value_col: Column containing values to compare
            baseline_group: Group to use as baseline (defaults to first group)

        Returns:
            DataFrame with group statistics and differences from baseline
        """
        if group_col not in df.columns:
            raise ValueError(f"Group column '{group_col}' not found")
        if value_col not in df.columns:
            raise ValueError(f"Value column '{value_col}' not found")

        groups = df.groupby(group_col)[value_col]

        stats = pd.DataFrame({
            'count': groups.count(),
            'mean': groups.mean(),
            'std': groups.std(),
            'median': groups.median(),
            'min': groups.min(),
            'max': groups.max()
        })

        # Determine baseline
        if baseline_group is None:
            baseline_group = stats.index[0]
        elif baseline_group not in stats.index:
            raise ValueError(f"Baseline group '{baseline_group}' not found")

        baseline_mean = stats.loc[baseline_group, 'mean']
        baseline_std = stats.loc[baseline_group, 'std']

        # Calculate differences
        stats['diff_from_baseline'] = stats['mean'] - baseline_mean
        stats['pct_diff_from_baseline'] = (
            (stats['mean'] - baseline_mean) / baseline_mean * 100
            if baseline_mean != 0 else np.nan
        )

        # Calculate standardized difference (effect size)
        if baseline_std > 0:
            stats['standardized_diff'] = stats['diff_from_baseline'] / baseline_std
        else:
            stats['standardized_diff'] = np.nan

        stats['baseline'] = stats.index == baseline_group

        return stats.reset_index()

    @staticmethod
    def bin_continuous_variable(
        df: pd.DataFrame,
        column: str,
        bins: Union[int, List[float]] = 5,
        labels: Optional[List[str]] = None,
        strategy: Literal["uniform", "quantile", "custom"] = "uniform"
    ) -> pd.DataFrame:
        """Bin a continuous variable into categories.

        Args:
            df: Input DataFrame
            column: Column to bin
            bins: Number of bins or list of bin edges
            labels: Labels for the bins
            strategy: Binning strategy:
                - "uniform": Equal-width bins
                - "quantile": Equal-frequency bins
                - "custom": Use provided bin edges

        Returns:
            DataFrame with new binned column
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")

        result = df.copy()
        new_col = f"{column}_binned"

        if strategy == "quantile":
            result[new_col] = pd.qcut(
                df[column],
                q=bins if isinstance(bins, int) else len(bins) - 1,
                labels=labels,
                duplicates='drop'
            )
        elif strategy == "uniform" or strategy == "custom":
            result[new_col] = pd.cut(
                df[column],
                bins=bins,
                labels=labels,
                include_lowest=True
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return result

    @staticmethod
    def calculate_rolling_statistics(
        df: pd.DataFrame,
        column: str,
        window: int,
        statistics: List[str] = ["mean", "std"],
        min_periods: Optional[int] = None
    ) -> pd.DataFrame:
        """Calculate rolling window statistics.

        Args:
            df: Input DataFrame
            column: Column to calculate statistics for
            window: Rolling window size
            statistics: List of statistics to calculate
            min_periods: Minimum observations required

        Returns:
            DataFrame with rolling statistics columns
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")

        result = df.copy()
        rolling = df[column].rolling(window=window, min_periods=min_periods)

        stat_funcs = {
            'mean': rolling.mean,
            'std': rolling.std,
            'min': rolling.min,
            'max': rolling.max,
            'sum': rolling.sum,
            'median': rolling.median,
            'var': rolling.var,
            'count': rolling.count,
        }

        for stat in statistics:
            if stat in stat_funcs:
                result[f"{column}_rolling_{stat}_{window}"] = stat_funcs[stat]()
            else:
                raise ValueError(f"Unknown statistic: {stat}")

        return result

    @staticmethod
    def normalize_by_group(
        df: pd.DataFrame,
        value_col: str,
        group_col: str,
        method: Literal["zscore", "minmax", "percent_of_max"] = "zscore"
    ) -> pd.DataFrame:
        """Normalize values within each group.

        Args:
            df: Input DataFrame
            value_col: Column to normalize
            group_col: Column defining groups
            method: Normalization method:
                - "zscore": (x - mean) / std
                - "minmax": (x - min) / (max - min)
                - "percent_of_max": x / max * 100

        Returns:
            DataFrame with new normalized column
        """
        if value_col not in df.columns:
            raise ValueError(f"Value column '{value_col}' not found")
        if group_col not in df.columns:
            raise ValueError(f"Group column '{group_col}' not found")

        result = df.copy()
        new_col = f"{value_col}_normalized"

        def normalize_group(group):
            if method == "zscore":
                mean = group[value_col].mean()
                std = group[value_col].std()
                if std > 0:
                    return (group[value_col] - mean) / std
                return pd.Series(0, index=group.index)
            elif method == "minmax":
                min_val = group[value_col].min()
                max_val = group[value_col].max()
                if max_val > min_val:
                    return (group[value_col] - min_val) / (max_val - min_val)
                return pd.Series(0.5, index=group.index)
            elif method == "percent_of_max":
                max_val = group[value_col].max()
                if max_val > 0:
                    return group[value_col] / max_val * 100
                return pd.Series(0, index=group.index)
            else:
                raise ValueError(f"Unknown method: {method}")

        result[new_col] = df.groupby(group_col, group_keys=False).apply(normalize_group)

        return result

    @staticmethod
    def create_interaction_terms(
        df: pd.DataFrame,
        columns: List[str],
        include_squares: bool = False
    ) -> pd.DataFrame:
        """Create interaction terms between columns.

        Args:
            df: Input DataFrame
            columns: Columns to create interactions for
            include_squares: Whether to include squared terms

        Returns:
            DataFrame with new interaction columns
        """
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"Columns not found: {missing}")

        result = df.copy()

        # Create pairwise interactions
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                interaction_name = f"{col1}_x_{col2}"
                result[interaction_name] = df[col1] * df[col2]

        # Create squared terms if requested
        if include_squares:
            for col in columns:
                result[f"{col}_squared"] = df[col] ** 2

        return result

    @staticmethod
    def lag_column(
        df: pd.DataFrame,
        column: str,
        lags: Union[int, List[int]] = 1,
        group_col: Optional[str] = None
    ) -> pd.DataFrame:
        """Create lagged versions of a column.

        Args:
            df: Input DataFrame
            column: Column to lag
            lags: Number of lags or list of specific lag values
            group_col: Optional column to group by before lagging

        Returns:
            DataFrame with new lagged columns
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")

        result = df.copy()
        lags_list = [lags] if isinstance(lags, int) else lags

        for lag in lags_list:
            lag_name = f"{column}_lag{lag}"
            if group_col:
                result[lag_name] = df.groupby(group_col)[column].shift(lag)
            else:
                result[lag_name] = df[column].shift(lag)

        return result

    @staticmethod
    def detect_and_encode_datetime_features(
        df: pd.DataFrame,
        datetime_col: str,
        features: List[str] = ["year", "month", "day", "dayofweek", "hour"]
    ) -> pd.DataFrame:
        """Extract datetime features from a datetime column.

        Args:
            df: Input DataFrame
            datetime_col: Column containing datetime values
            features: List of features to extract:
                - year, month, day, hour, minute, second
                - dayofweek (0=Monday), dayofyear, weekofyear
                - quarter, is_weekend, is_month_start, is_month_end

        Returns:
            DataFrame with new datetime feature columns
        """
        if datetime_col not in df.columns:
            raise ValueError(f"Column '{datetime_col}' not found")

        result = df.copy()
        dt = pd.to_datetime(df[datetime_col])

        feature_funcs = {
            'year': lambda x: x.dt.year,
            'month': lambda x: x.dt.month,
            'day': lambda x: x.dt.day,
            'hour': lambda x: x.dt.hour,
            'minute': lambda x: x.dt.minute,
            'second': lambda x: x.dt.second,
            'dayofweek': lambda x: x.dt.dayofweek,
            'dayofyear': lambda x: x.dt.dayofyear,
            'weekofyear': lambda x: x.dt.isocalendar().week,
            'quarter': lambda x: x.dt.quarter,
            'is_weekend': lambda x: x.dt.dayofweek >= 5,
            'is_month_start': lambda x: x.dt.is_month_start,
            'is_month_end': lambda x: x.dt.is_month_end,
        }

        for feature in features:
            if feature in feature_funcs:
                result[f"{datetime_col}_{feature}"] = feature_funcs[feature](dt)
            else:
                raise ValueError(f"Unknown datetime feature: {feature}")

        return result
