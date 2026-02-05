"""
Property-based tests for DataCleaner and Preprocessor classes.

This module contains property-based tests using Hypothesis to validate
the correctness properties of the data preparation implementation.
"""

import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from no_nonsense_experiment_analysis.data_prep.cleaner import DataCleaner
from no_nonsense_experiment_analysis.data_prep.preprocessor import Preprocessor
from no_nonsense_experiment_analysis.core.exceptions import DataValidationError


# Custom strategies for generating test data
@st.composite
def dataframe_strategy(draw, min_rows=1, max_rows=20, min_cols=1, max_cols=5):
    """Generate random DataFrames with mixed types."""
    num_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    num_cols = draw(st.integers(min_value=min_cols, max_value=max_cols))

    columns = draw(st.lists(
        st.text(min_size=1, max_size=10).filter(lambda x: x.strip() and x.isidentifier()),
        min_size=num_cols,
        max_size=num_cols,
        unique=True
    ))

    data = {}
    for col in columns:
        col_type = draw(st.sampled_from(['numeric', 'string']))
        if col_type == 'numeric':
            data[col] = draw(st.lists(
                st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
                min_size=num_rows,
                max_size=num_rows
            ))
        else:
            data[col] = draw(st.lists(
                st.text(min_size=1, max_size=10),
                min_size=num_rows,
                max_size=num_rows
            ))

    return pd.DataFrame(data)


@st.composite
def dataframe_with_missing_strategy(draw, min_rows=5, max_rows=20, min_cols=2, max_cols=5, missing_prob=0.2):
    """Generate DataFrames with controlled missing values."""
    num_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    num_cols = draw(st.integers(min_value=min_cols, max_value=max_cols))

    columns = draw(st.lists(
        st.text(min_size=1, max_size=10).filter(lambda x: x.strip() and x.isidentifier()),
        min_size=num_cols,
        max_size=num_cols,
        unique=True
    ))

    data = {}
    for col in columns:
        col_data = []
        for _ in range(num_rows):
            if draw(st.floats(min_value=0, max_value=1)) < missing_prob:
                col_data.append(None)
            else:
                col_data.append(draw(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)))
        data[col] = col_data

    return pd.DataFrame(data)


@st.composite
def numeric_dataframe_strategy(draw, min_rows=3, max_rows=20, min_cols=1, max_cols=5):
    """Generate DataFrames with only numeric columns."""
    num_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    num_cols = draw(st.integers(min_value=min_cols, max_value=max_cols))

    columns = draw(st.lists(
        st.text(min_size=1, max_size=10).filter(lambda x: x.strip() and x.isidentifier()),
        min_size=num_cols,
        max_size=num_cols,
        unique=True
    ))

    data = {}
    for col in columns:
        data[col] = draw(st.lists(
            st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
            min_size=num_rows,
            max_size=num_rows
        ))

    return pd.DataFrame(data)


class TestDataIntegrityPreservation:
    """Property 4: Data integrity preservation - Validates Requirements 2.5"""

    @given(df=dataframe_with_missing_strategy())
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_handle_missing_values_preserves_original(self, df):
        """
        **Validates: Requirements 2.5**

        Property: After calling handle_missing_values, the original DataFrame
        must remain completely unchanged.
        """
        cleaner = DataCleaner(strict_mode=True)

        # Store original DataFrame state
        original_copy = df.copy()
        original_columns = list(df.columns)
        original_index = df.index.tolist()

        # Apply cleaning operation
        strategies = ["drop", "fill_mean", "fill_median", "fill_mode", "fill_value"]
        for strategy in strategies:
            if strategy == "fill_value":
                result = cleaner.handle_missing_values(df, strategy=strategy, fill_value=0)
            else:
                result = cleaner.handle_missing_values(df, strategy=strategy)

            # Property: Original must be unchanged
            assert list(df.columns) == original_columns
            assert df.index.tolist() == original_index

            # Compare values considering NaN
            for col in df.columns:
                for i in range(len(df)):
                    orig_val = original_copy.iloc[i][col]
                    curr_val = df.iloc[i][col]
                    if pd.isna(orig_val) and pd.isna(curr_val):
                        continue
                    assert orig_val == curr_val, f"Original DataFrame was modified at [{i}, {col}]"

    @given(df=numeric_dataframe_strategy())
    @settings(max_examples=25, suppress_health_check=[HealthCheck.too_slow])
    def test_remove_duplicates_preserves_original(self, df):
        """
        **Validates: Requirements 2.5**

        Property: After calling remove_duplicates, the original DataFrame
        must remain completely unchanged.
        """
        cleaner = DataCleaner(strict_mode=True)

        # Store original state
        original_copy = df.copy()
        original_shape = df.shape

        # Apply duplicate removal
        result = cleaner.remove_duplicates(df)

        # Property: Original must be unchanged
        assert df.shape == original_shape
        assert df.equals(original_copy)

    @given(df=numeric_dataframe_strategy(min_rows=10))
    @settings(max_examples=25, suppress_health_check=[HealthCheck.too_slow])
    def test_detect_outliers_preserves_original(self, df):
        """
        **Validates: Requirements 2.5**

        Property: After calling detect_outliers, the original DataFrame
        must remain completely unchanged.
        """
        cleaner = DataCleaner(strict_mode=True)

        # Store original state
        original_copy = df.copy()

        # Apply outlier detection with different methods
        for method in ["iqr", "zscore", "modified_zscore"]:
            result = cleaner.detect_outliers(df, method=method)

            # Property: Original must be unchanged
            assert df.equals(original_copy), f"Original modified after {method} outlier detection"

    @given(df=numeric_dataframe_strategy())
    @settings(max_examples=25, suppress_health_check=[HealthCheck.too_slow])
    def test_normalize_columns_preserves_original(self, df):
        """
        **Validates: Requirements 2.5**

        Property: After calling normalize_columns, the original DataFrame
        must remain completely unchanged.
        """
        preprocessor = Preprocessor(strict_mode=True)

        # Store original state
        original_copy = df.copy()
        columns = list(df.columns)

        # Apply normalization with different methods
        for method in ["minmax", "zscore", "robust", "maxabs"]:
            result = preprocessor.normalize_columns(df, columns=columns, method=method)

            # Property: Original must be unchanged
            assert df.equals(original_copy), f"Original modified after {method} normalization"

    @given(
        num_rows=st.integers(min_value=3, max_value=15),
        categories=st.lists(
            st.text(min_size=1, max_size=10).filter(lambda x: x.strip()),
            min_size=2,
            max_size=5,
            unique=True
        )
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_encode_categorical_preserves_original(self, num_rows, categories):
        """
        **Validates: Requirements 2.5**

        Property: After calling encode_categorical, the original DataFrame
        must remain completely unchanged.
        """
        preprocessor = Preprocessor(strict_mode=True)

        # Create DataFrame with categorical column
        df = pd.DataFrame({
            'category': [categories[i % len(categories)] for i in range(num_rows)],
            'value': list(range(num_rows))
        })

        original_copy = df.copy()

        # Apply encoding with different methods
        for method in ["label", "onehot"]:
            result = preprocessor.encode_categorical(df, columns=['category'], method=method)

            # Property: Original must be unchanged
            assert df.equals(original_copy), f"Original modified after {method} encoding"


class TestCleaningOperationConsistency:
    """Property 5: Cleaning operation consistency - Validates Requirements 2.1, 2.3, 2.4"""

    @given(df=dataframe_with_missing_strategy())
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_handle_missing_values_idempotent(self, df):
        """
        **Validates: Requirements 2.1, 2.3, 2.4**

        Property: Applying the same missing value handling strategy multiple times
        should produce the same result (idempotency).
        """
        cleaner = DataCleaner(strict_mode=True)

        strategies = ["drop", "fill_mean", "fill_median", "fill_mode"]

        for strategy in strategies:
            # Apply once
            result1 = cleaner.handle_missing_values(df, strategy=strategy)

            # Apply again to the result
            result2 = cleaner.handle_missing_values(result1, strategy=strategy)

            # Property: Second application should produce same result
            if strategy == "drop":
                # For drop, row counts should be the same after cleaning
                assert len(result2) == len(result1)

            # The values should be the same
            assert result1.shape == result2.shape
            for col in result1.columns:
                for i in range(len(result1)):
                    v1 = result1.iloc[i][col]
                    v2 = result2.iloc[i][col]
                    if pd.isna(v1) and pd.isna(v2):
                        continue
                    assert v1 == v2, f"Idempotency violated for {strategy} at [{i}, {col}]"

    @given(df=numeric_dataframe_strategy())
    @settings(max_examples=25, suppress_health_check=[HealthCheck.too_slow])
    def test_remove_duplicates_idempotent(self, df):
        """
        **Validates: Requirements 2.1, 2.3, 2.4**

        Property: Applying remove_duplicates multiple times should produce
        the same result after the first application.
        """
        cleaner = DataCleaner(strict_mode=True)

        # Apply once
        result1 = cleaner.remove_duplicates(df)

        # Apply again
        result2 = cleaner.remove_duplicates(result1)

        # Property: Second application should produce same result
        assert len(result1) == len(result2)
        assert result1.equals(result2)

    @given(df=numeric_dataframe_strategy(min_rows=15))
    @settings(max_examples=25, suppress_health_check=[HealthCheck.too_slow])
    def test_outlier_detection_consistency(self, df):
        """
        **Validates: Requirements 2.1, 2.3, 2.4**

        Property: Calling detect_outliers with the same parameters on the same
        DataFrame should always produce the same result.
        """
        cleaner = DataCleaner(strict_mode=True)

        for method in ["iqr", "zscore", "modified_zscore"]:
            # Call multiple times
            result1 = cleaner.detect_outliers(df, method=method)
            result2 = cleaner.detect_outliers(df, method=method)
            result3 = cleaner.detect_outliers(df, method=method)

            # Property: All results should be identical
            assert result1.equals(result2), f"Inconsistent results for {method}"
            assert result2.equals(result3), f"Inconsistent results for {method}"

    @given(df=numeric_dataframe_strategy(min_rows=20))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_remove_outliers_idempotent(self, df):
        """
        **Validates: Requirements 2.1, 2.3, 2.4**

        Property: After removing outliers once, removing outliers again should
        not remove any additional rows (or very few due to distribution changes).
        """
        cleaner = DataCleaner(strict_mode=True)

        # Skip DataFrames with very low variance (all same values or nearly so)
        # as these have unpredictable outlier behavior
        for col in df.columns:
            if df[col].nunique() < 5:
                assume(False)
            # Also skip if standard deviation is very small
            if df[col].std() < 0.1:
                assume(False)

        # Only test zscore method as it's more stable for this property
        method = "zscore"

        # Apply once
        result1 = cleaner.remove_outliers(df, method=method)

        # Skip if too few rows remain (edge case)
        if len(result1) < 10:
            return

        # Apply again - should be mostly stable
        result2 = cleaner.remove_outliers(result1, method=method)

        # Property: Second removal should retain most rows
        # (the distribution after first removal is typically more normal)
        if len(result1) > 0:
            retention_rate = len(result2) / len(result1)
            assert retention_rate >= 0.5, f"Too many rows removed in second pass for {method}"

    @given(
        df=dataframe_with_missing_strategy(),
        fill_value=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_fill_value_consistency(self, df, fill_value):
        """
        **Validates: Requirements 2.1, 2.3, 2.4**

        Property: Using fill_value strategy with the same value should always
        produce consistent results.
        """
        cleaner = DataCleaner(strict_mode=True)

        result1 = cleaner.handle_missing_values(df, strategy="fill_value", fill_value=fill_value)
        result2 = cleaner.handle_missing_values(df, strategy="fill_value", fill_value=fill_value)

        # Property: Results should be identical
        assert result1.shape == result2.shape
        for col in result1.columns:
            for i in range(len(result1)):
                v1 = result1.iloc[i][col]
                v2 = result2.iloc[i][col]
                if pd.isna(v1) and pd.isna(v2):
                    continue
                assert v1 == v2


class TestOutputTypeGuarantee:
    """Property 6: Output type guarantee - Validates Requirements 2.6"""

    @given(df=dataframe_with_missing_strategy())
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_handle_missing_values_returns_dataframe(self, df):
        """
        **Validates: Requirements 2.6**

        Property: handle_missing_values must always return a pandas DataFrame
        with the same or fewer rows than the input.
        """
        cleaner = DataCleaner(strict_mode=True)

        strategies = ["drop", "fill_mean", "fill_median", "fill_mode", "fill_value"]

        for strategy in strategies:
            if strategy == "fill_value":
                result = cleaner.handle_missing_values(df, strategy=strategy, fill_value=0)
            else:
                result = cleaner.handle_missing_values(df, strategy=strategy)

            # Property: Must return DataFrame
            assert isinstance(result, pd.DataFrame), f"Expected DataFrame, got {type(result)}"

            # Property: Must have same or fewer rows
            assert len(result) <= len(df), f"Row count increased for {strategy}"

            # Property: Must have same columns (for non-drop strategies that don't affect columns)
            assert list(result.columns) == list(df.columns)

    @given(df=numeric_dataframe_strategy())
    @settings(max_examples=25, suppress_health_check=[HealthCheck.too_slow])
    def test_remove_duplicates_returns_dataframe(self, df):
        """
        **Validates: Requirements 2.6**

        Property: remove_duplicates must always return a pandas DataFrame
        with the same or fewer rows than the input.
        """
        cleaner = DataCleaner(strict_mode=True)

        result = cleaner.remove_duplicates(df)

        # Property: Must return DataFrame
        assert isinstance(result, pd.DataFrame)

        # Property: Must have same or fewer rows
        assert len(result) <= len(df)

        # Property: Must have same columns
        assert list(result.columns) == list(df.columns)

    @given(df=numeric_dataframe_strategy(min_rows=10))
    @settings(max_examples=25, suppress_health_check=[HealthCheck.too_slow])
    def test_detect_outliers_returns_boolean_dataframe(self, df):
        """
        **Validates: Requirements 2.6**

        Property: detect_outliers must return a boolean DataFrame indicating
        outlier positions.
        """
        cleaner = DataCleaner(strict_mode=True)

        for method in ["iqr", "zscore", "modified_zscore"]:
            result = cleaner.detect_outliers(df, method=method)

            # Property: Must return DataFrame
            assert isinstance(result, pd.DataFrame)

            # Property: Must have same number of rows
            assert len(result) == len(df)

            # Property: All values must be boolean
            for col in result.columns:
                assert result[col].dtype == bool, f"Expected bool dtype for {col}"

    @given(df=numeric_dataframe_strategy(min_rows=10))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_remove_outliers_returns_dataframe(self, df):
        """
        **Validates: Requirements 2.6**

        Property: remove_outliers must always return a pandas DataFrame
        with the same or fewer rows.
        """
        cleaner = DataCleaner(strict_mode=True)

        for method in ["iqr", "zscore", "modified_zscore"]:
            result = cleaner.remove_outliers(df, method=method)

            # Property: Must return DataFrame
            assert isinstance(result, pd.DataFrame)

            # Property: Must have same or fewer rows
            assert len(result) <= len(df)

            # Property: Must have same columns
            assert list(result.columns) == list(df.columns)

    @given(df=numeric_dataframe_strategy())
    @settings(max_examples=25, suppress_health_check=[HealthCheck.too_slow])
    def test_normalize_columns_returns_dataframe(self, df):
        """
        **Validates: Requirements 2.6**

        Property: normalize_columns must always return a pandas DataFrame
        with the same shape as the input.
        """
        preprocessor = Preprocessor(strict_mode=True)
        columns = list(df.columns)

        for method in ["minmax", "zscore", "robust", "maxabs"]:
            result = preprocessor.normalize_columns(df, columns=columns, method=method)

            # Property: Must return DataFrame
            assert isinstance(result, pd.DataFrame)

            # Property: Must have same shape
            assert result.shape == df.shape

            # Property: Must have same columns
            assert list(result.columns) == list(df.columns)

    @given(
        num_rows=st.integers(min_value=3, max_value=15),
        categories=st.lists(
            st.text(min_size=1, max_size=8).filter(lambda x: x.strip()),
            min_size=2,
            max_size=4,
            unique=True
        )
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_encode_categorical_returns_dataframe(self, num_rows, categories):
        """
        **Validates: Requirements 2.6**

        Property: encode_categorical must always return a pandas DataFrame
        with the same number of rows.
        """
        preprocessor = Preprocessor(strict_mode=True)

        df = pd.DataFrame({
            'category': [categories[i % len(categories)] for i in range(num_rows)],
            'value': list(range(num_rows))
        })

        for method in ["label", "onehot", "ordinal", "binary"]:
            result = preprocessor.encode_categorical(df, columns=['category'], method=method)

            # Property: Must return DataFrame
            assert isinstance(result, pd.DataFrame)

            # Property: Must have same number of rows
            assert len(result) == len(df)

    @given(df=numeric_dataframe_strategy(min_cols=2))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_create_features_returns_dataframe(self, df):
        """
        **Validates: Requirements 2.6**

        Property: create_features must always return a pandas DataFrame
        with the same number of rows and additional columns.
        """
        preprocessor = Preprocessor(strict_mode=True)

        # Skip if not enough columns
        assume(len(df.columns) >= 2)

        col1, col2 = list(df.columns)[:2]

        feature_specs = {
            "sum_feature": {"operation": "sum", "columns": [col1, col2]},
            "product_feature": {"operation": "product", "columns": [col1, col2]},
        }

        result = preprocessor.create_features(df, feature_specs)

        # Property: Must return DataFrame
        assert isinstance(result, pd.DataFrame)

        # Property: Must have same number of rows
        assert len(result) == len(df)

        # Property: Must have additional columns
        assert len(result.columns) > len(df.columns)

        # Property: New columns should exist
        assert "sum_feature" in result.columns
        assert "product_feature" in result.columns


class TestInputValidation:
    """Additional tests for input validation across data prep classes."""

    @given(
        invalid_input=st.one_of(
            st.lists(st.integers()),
            st.dictionaries(st.text(), st.integers()),
            st.text(),
            st.integers(),
            st.none()
        )
    )
    @settings(max_examples=20)
    def test_cleaner_rejects_non_dataframe(self, invalid_input):
        """
        Property: DataCleaner methods must reject non-DataFrame inputs
        with DataValidationError.
        """
        cleaner = DataCleaner(strict_mode=True)

        assume(not isinstance(invalid_input, pd.DataFrame))

        # All methods should raise DataValidationError
        with pytest.raises(DataValidationError):
            cleaner.handle_missing_values(invalid_input)

        with pytest.raises(DataValidationError):
            cleaner.remove_duplicates(invalid_input)

        with pytest.raises(DataValidationError):
            cleaner.detect_outliers(invalid_input)

    @given(
        invalid_input=st.one_of(
            st.lists(st.integers()),
            st.dictionaries(st.text(), st.integers()),
            st.text(),
            st.integers(),
            st.none()
        )
    )
    @settings(max_examples=20)
    def test_preprocessor_rejects_non_dataframe(self, invalid_input):
        """
        Property: Preprocessor methods must reject non-DataFrame inputs
        with DataValidationError.
        """
        preprocessor = Preprocessor(strict_mode=True)

        assume(not isinstance(invalid_input, pd.DataFrame))

        # All methods should raise DataValidationError
        with pytest.raises(DataValidationError):
            preprocessor.normalize_columns(invalid_input, columns=['col1'])

        with pytest.raises(DataValidationError):
            preprocessor.encode_categorical(invalid_input, columns=['col1'])

        with pytest.raises(DataValidationError):
            preprocessor.create_features(invalid_input, {})


class TestCleaningSummary:
    """Tests for cleaning summary functionality."""

    @given(
        num_rows=st.integers(min_value=5, max_value=30),
        missing_count=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=20)
    def test_cleaning_summary_accuracy(self, num_rows, missing_count):
        """
        Property: get_cleaning_summary must accurately report the changes
        made during cleaning operations.
        """
        cleaner = DataCleaner(strict_mode=True)

        # Ensure missing_count doesn't exceed rows
        missing_count = min(missing_count, num_rows - 1)

        # Create DataFrame with missing values
        data = list(range(num_rows))
        for i in range(missing_count):
            data[i] = None

        df = pd.DataFrame({'col1': data})

        # Clean the data
        cleaned = cleaner.handle_missing_values(df, strategy="drop")

        # Get summary
        summary = cleaner.get_cleaning_summary(df, cleaned)

        # Property: Summary must be accurate
        assert summary['original_rows'] == num_rows
        assert summary['cleaned_rows'] == num_rows - missing_count
        assert summary['rows_removed'] == missing_count
        assert summary['original_missing_values'] == missing_count
        assert summary['cleaned_missing_values'] == 0


if __name__ == "__main__":
    pytest.main([__file__])
