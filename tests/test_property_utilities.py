"""
Property-based tests for utilities module.

This module contains property-based tests using Hypothesis to validate
the correctness properties of the statistical functions, visualization tools,
and data transformers.
"""

import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, assume, settings, HealthCheck
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from no_nonsense_experiment_analysis.utilities import (
    StatisticalFunctions,
    VisualizationTools,
    DataTransformers
)


# Custom strategies for generating test data
@st.composite
def numeric_array_strategy(draw, min_size=5, max_size=50, allow_nan=False):
    """Generate a numeric array."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    values = [
        draw(st.floats(min_value=-100, max_value=100, allow_nan=allow_nan, allow_infinity=False))
        for _ in range(size)
    ]
    return np.array(values)


@st.composite
def two_group_strategy(draw, min_size=5, max_size=30):
    """Generate two numeric arrays for comparison."""
    size1 = draw(st.integers(min_value=min_size, max_value=max_size))
    size2 = draw(st.integers(min_value=min_size, max_value=max_size))

    group1 = [
        draw(st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False))
        for _ in range(size1)
    ]
    group2 = [
        draw(st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False))
        for _ in range(size2)
    ]

    return np.array(group1), np.array(group2)


@st.composite
def p_values_strategy(draw, min_size=2, max_size=20):
    """Generate valid p-values array."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    p_values = [
        draw(st.floats(min_value=0.001, max_value=0.999, allow_nan=False))
        for _ in range(size)
    ]
    return np.array(p_values)


@st.composite
def dataframe_strategy(draw, min_rows=10, max_rows=50, n_numeric_cols=3):
    """Generate a DataFrame with numeric columns."""
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))

    data = {}
    for i in range(n_numeric_cols):
        col_name = f"col_{i}"
        data[col_name] = [
            draw(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False))
            for _ in range(n_rows)
        ]

    # Add a group column
    groups = ['A', 'B', 'C']
    data['group'] = [groups[j % len(groups)] for j in range(n_rows)]

    return pd.DataFrame(data)


class TestMathematicalOperationCorrectness:
    """Property 10: Mathematical operation correctness - Validates Requirements 4.5"""

    @given(groups=two_group_strategy())
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_effect_size_symmetry(self, groups):
        """
        **Validates: Requirements 4.5**

        Property: Effect size between groups should be symmetric (opposite sign
        when groups are swapped).
        """
        group1, group2 = groups

        # Ensure groups have variance
        assume(np.std(group1) > 0.01)
        assume(np.std(group2) > 0.01)

        d1 = StatisticalFunctions.calculate_effect_size(group1, group2)
        d2 = StatisticalFunctions.calculate_effect_size(group2, group1)

        # Property: d(A, B) = -d(B, A)
        assert abs(d1 + d2) < 0.0001, "Effect size should be symmetric"

    @given(groups=two_group_strategy())
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_hedges_g_smaller_than_cohens_d(self, groups):
        """
        **Validates: Requirements 4.5**

        Property: Hedges' g should have smaller absolute value than Cohen's d
        due to bias correction.
        """
        group1, group2 = groups

        assume(np.std(group1) > 0.01)
        assume(np.std(group2) > 0.01)

        cohens_d = StatisticalFunctions.calculate_effect_size(group1, group2, "cohens_d")
        hedges_g = StatisticalFunctions.calculate_effect_size(group1, group2, "hedges_g")

        # Property: |g| <= |d| (Hedges' correction reduces absolute value)
        assert abs(hedges_g) <= abs(cohens_d) + 0.0001

    @given(data=numeric_array_strategy(min_size=10))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_bootstrap_ci_contains_point_estimate(self, data):
        """
        **Validates: Requirements 4.5**

        Property: Bootstrap confidence interval for mean should contain the
        sample mean (with high probability for 99% CI).
        """
        assume(np.std(data) > 0.01)

        mean = np.mean(data)
        lower, upper = StatisticalFunctions.bootstrap_confidence_interval(
            data, np.mean, confidence_level=0.99, n_bootstrap=500, random_state=42
        )

        # Property: CI should contain the point estimate
        assert lower <= mean <= upper, "99% CI should contain sample mean"

    @given(data=numeric_array_strategy(min_size=10))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_bootstrap_ci_width_increases_with_confidence(self, data):
        """
        **Validates: Requirements 4.5**

        Property: Higher confidence level should produce wider confidence intervals.
        """
        assume(np.std(data) > 0.01)

        # Calculate CIs at different levels
        lower_90, upper_90 = StatisticalFunctions.bootstrap_confidence_interval(
            data, np.mean, confidence_level=0.90, n_bootstrap=200, random_state=42
        )
        lower_99, upper_99 = StatisticalFunctions.bootstrap_confidence_interval(
            data, np.mean, confidence_level=0.99, n_bootstrap=200, random_state=42
        )

        width_90 = upper_90 - lower_90
        width_99 = upper_99 - lower_99

        # Property: 99% CI should be wider than 90% CI
        assert width_99 >= width_90 - 0.01, "Higher confidence should give wider CI"

    @given(p_values=p_values_strategy())
    @settings(max_examples=30)
    def test_bonferroni_correction_increases_p_values(self, p_values):
        """
        **Validates: Requirements 4.5**

        Property: Bonferroni correction should increase all p-values
        (more conservative).
        """
        corrected, _ = StatisticalFunctions.multiple_comparison_correction(
            p_values, method="bonferroni"
        )

        # Property: Corrected p-values >= original p-values
        assert np.all(corrected >= p_values - 0.0001)

        # Property: Corrected p-values <= 1
        assert np.all(corrected <= 1.0)

    @given(p_values=p_values_strategy())
    @settings(max_examples=30)
    def test_benjamini_hochberg_less_conservative_than_bonferroni(self, p_values):
        """
        **Validates: Requirements 4.5**

        Property: Benjamini-Hochberg should be less conservative than Bonferroni
        (i.e., corrected p-values should be smaller or equal).
        """
        bonf_corrected, _ = StatisticalFunctions.multiple_comparison_correction(
            p_values, method="bonferroni"
        )
        bh_corrected, _ = StatisticalFunctions.multiple_comparison_correction(
            p_values, method="benjamini_hochberg"
        )

        # Property: BH p-values <= Bonferroni p-values
        assert np.all(bh_corrected <= bonf_corrected + 0.0001)

    @given(
        effect_size=st.floats(min_value=0.2, max_value=1.5, allow_nan=False),
        sample_size=st.integers(min_value=10, max_value=200)
    )
    @settings(max_examples=30)
    def test_power_increases_with_sample_size(self, effect_size, sample_size):
        """
        **Validates: Requirements 4.5**

        Property: Statistical power should increase with sample size.
        """
        power_n = StatisticalFunctions.calculate_power(effect_size, sample_size)
        power_2n = StatisticalFunctions.calculate_power(effect_size, sample_size * 2)

        # Property: Power with 2n should be >= power with n
        assert power_2n >= power_n - 0.0001

        # Property: Power should be between 0 and 1
        assert 0 <= power_n <= 1
        assert 0 <= power_2n <= 1

    @given(
        effect_size=st.floats(min_value=0.3, max_value=1.0, allow_nan=False),
        sample_size=st.integers(min_value=20, max_value=100)
    )
    @settings(max_examples=20)
    def test_power_increases_with_effect_size(self, effect_size, sample_size):
        """
        **Validates: Requirements 4.5**

        Property: Statistical power should increase with effect size.
        """
        power_d = StatisticalFunctions.calculate_power(effect_size, sample_size)
        power_2d = StatisticalFunctions.calculate_power(effect_size * 2, sample_size)

        # Property: Power with 2*d should be >= power with d
        assert power_2d >= power_d - 0.0001


class TestInterfaceConsistency:
    """Property 11: Interface consistency - Validates Requirements 4.6"""

    def test_effect_size_types_return_float(self):
        """
        **Validates: Requirements 4.6**

        Property: All effect size calculation methods should return float.
        """
        group1 = np.random.normal(10, 2, 30)
        group2 = np.random.normal(12, 2, 30)

        for effect_type in StatisticalFunctions.EFFECT_SIZE_TYPES:
            result = StatisticalFunctions.calculate_effect_size(
                group1, group2, effect_type
            )
            assert isinstance(result, float), f"{effect_type} should return float"

    def test_correction_methods_return_arrays(self):
        """
        **Validates: Requirements 4.6**

        Property: All correction methods should return arrays of same length.
        """
        p_values = [0.01, 0.05, 0.10, 0.20]

        for method in StatisticalFunctions.CORRECTION_METHODS:
            corrected, reject = StatisticalFunctions.multiple_comparison_correction(
                p_values, method=method
            )

            assert len(corrected) == len(p_values)
            assert len(reject) == len(p_values)
            assert isinstance(corrected, np.ndarray)
            assert isinstance(reject, np.ndarray)

    def test_bootstrap_returns_tuple(self):
        """
        **Validates: Requirements 4.6**

        Property: Bootstrap methods should return tuples of correct length.
        """
        data = np.random.normal(10, 2, 30)

        result = StatisticalFunctions.bootstrap_confidence_interval(data)
        assert isinstance(result, tuple)
        assert len(result) == 2

        group1 = np.random.normal(10, 2, 30)
        group2 = np.random.normal(12, 2, 30)

        result2 = StatisticalFunctions.bootstrap_two_sample_difference(group1, group2)
        assert isinstance(result2, tuple)
        assert len(result2) == 3

    def test_visualization_returns_figure(self):
        """
        **Validates: Requirements 4.6**

        Property: All visualization methods should return matplotlib Figure.
        """
        data = np.random.normal(10, 2, 50)
        groups = {'A': np.random.normal(10, 2, 30), 'B': np.random.normal(12, 2, 30)}

        fig1 = VisualizationTools.plot_distribution(data)
        assert isinstance(fig1, plt.Figure)
        plt.close(fig1)

        fig2 = VisualizationTools.plot_comparison(groups)
        assert isinstance(fig2, plt.Figure)
        plt.close(fig2)

    def test_transformers_return_dataframe(self):
        """
        **Validates: Requirements 4.6**

        Property: Transformer methods that modify data should return DataFrame.
        """
        df = pd.DataFrame({
            'group': ['A', 'B', 'A', 'B'] * 10,
            'value': np.random.normal(10, 2, 40),
            'time': pd.date_range('2024-01-01', periods=40)
        })

        result = DataTransformers.aggregate_by_groups(df, 'group', 'value')
        assert isinstance(result, pd.DataFrame)

        result2 = DataTransformers.bin_continuous_variable(df, 'value', bins=5)
        assert isinstance(result2, pd.DataFrame)

        result3 = DataTransformers.lag_column(df, 'value', lags=1)
        assert isinstance(result3, pd.DataFrame)

    def test_summary_statistics_returns_dict(self):
        """
        **Validates: Requirements 4.6**

        Property: calculate_summary_statistics should return dictionary with
        expected structure.
        """
        df = pd.DataFrame({
            'a': np.random.normal(10, 2, 50),
            'b': np.random.normal(20, 3, 50)
        })

        summary = DataTransformers.calculate_summary_statistics(df)

        assert isinstance(summary, dict)
        assert 'a' in summary
        assert 'b' in summary
        assert '_overall' in summary

        # Check required keys in column summary
        for col in ['a', 'b']:
            assert 'mean' in summary[col]
            assert 'std' in summary[col]
            assert 'count' in summary[col]

    @given(df=dataframe_strategy())
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_aggregation_preserves_groups(self, df):
        """
        **Validates: Requirements 4.6**

        Property: Aggregation should produce one row per group.
        """
        result = DataTransformers.aggregate_by_groups(df, 'group')

        n_groups = df['group'].nunique()
        assert len(result) == n_groups

    @given(df=dataframe_strategy())
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_melt_increases_rows(self, df):
        """
        **Validates: Requirements 4.6**

        Property: Melting (wide to long) should multiply rows by number of
        value columns.
        """
        numeric_cols = [c for c in df.columns if c != 'group']
        result = DataTransformers.melt_wide_to_long(
            df, id_vars='group', value_vars=numeric_cols
        )

        expected_rows = len(df) * len(numeric_cols)
        assert len(result) == expected_rows


class TestDataTransformerCorrectness:
    """Additional tests for data transformer correctness."""

    def test_group_differences_baseline_has_zero_diff(self):
        """
        Property: Baseline group should have zero difference from itself.
        """
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'C', 'C'],
            'value': [10, 12, 20, 22, 15, 17]
        })

        result = DataTransformers.calculate_group_differences(
            df, 'group', 'value', baseline_group='A'
        )

        baseline_row = result[result['group'] == 'A']
        assert baseline_row['diff_from_baseline'].iloc[0] == 0

    def test_rolling_statistics_window_size(self):
        """
        Property: Rolling statistics should have correct number of NaN values.
        """
        df = pd.DataFrame({'value': range(10)})

        result = DataTransformers.calculate_rolling_statistics(
            df, 'value', window=3, statistics=['mean']
        )

        # First (window-1) values should be NaN
        rolling_col = 'value_rolling_mean_3'
        assert result[rolling_col].iloc[:2].isna().all()
        assert not result[rolling_col].iloc[2:].isna().any()

    def test_interaction_terms_count(self):
        """
        Property: Number of interaction terms should be n*(n-1)/2.
        """
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })

        result = DataTransformers.create_interaction_terms(df, ['a', 'b', 'c'])

        # 3 columns -> 3 interactions: a_x_b, a_x_c, b_x_c
        expected_new_cols = 3
        assert len(result.columns) == len(df.columns) + expected_new_cols

        # With squares: 3 interactions + 3 squares = 6 new columns
        result_with_sq = DataTransformers.create_interaction_terms(
            df, ['a', 'b', 'c'], include_squares=True
        )
        assert len(result_with_sq.columns) == len(df.columns) + 6


class TestErrorHandling:
    """Tests for proper error handling in utilities."""

    def test_effect_size_requires_min_samples(self):
        """Effect size should require minimum samples in each group."""
        with pytest.raises(ValueError):
            StatisticalFunctions.calculate_effect_size([1], [2, 3, 4])

    def test_invalid_effect_type_raises_error(self):
        """Invalid effect type should raise ValueError."""
        with pytest.raises(ValueError):
            StatisticalFunctions.calculate_effect_size(
                [1, 2, 3], [4, 5, 6], effect_type="invalid"
            )

    def test_invalid_correction_method_raises_error(self):
        """Invalid correction method should raise ValueError."""
        with pytest.raises(ValueError):
            StatisticalFunctions.multiple_comparison_correction(
                [0.01, 0.05], method="invalid"
            )

    def test_empty_p_values_raises_error(self):
        """Empty p-values should raise ValueError."""
        with pytest.raises(ValueError):
            StatisticalFunctions.multiple_comparison_correction([])

    def test_missing_column_raises_error(self):
        """Missing column should raise ValueError in transformers."""
        df = pd.DataFrame({'a': [1, 2, 3]})

        with pytest.raises(ValueError):
            DataTransformers.aggregate_by_groups(df, 'missing_col')

        with pytest.raises(ValueError):
            DataTransformers.bin_continuous_variable(df, 'missing')


if __name__ == "__main__":
    pytest.main([__file__])
