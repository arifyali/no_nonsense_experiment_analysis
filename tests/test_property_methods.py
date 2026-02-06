"""
Property-based tests for experimental methods framework.

This module contains property-based tests using Hypothesis to validate
the correctness properties of the experimental methods implementation.
"""

import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from no_nonsense_experiment_analysis.methods import (
    ExperimentalMethod,
    MethodChain,
    MethodResult,
    ABTest,
    OneWayANOVA,
    ChiSquareTest,
    LinearRegressionAnalysis,
    default_registry
)
from no_nonsense_experiment_analysis.core.exceptions import MethodExecutionError


# Custom strategies for generating test data
@st.composite
def ab_test_data_strategy(draw, min_rows=20, max_rows=100):
    """Generate valid A/B test data."""
    n = draw(st.integers(min_value=min_rows, max_value=max_rows))

    # Generate group assignments (roughly balanced)
    groups = ['A', 'B']
    group_col = [groups[i % 2] for i in range(n)]

    # Generate metric values with some difference between groups
    base_metric = draw(st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False))
    effect_size = draw(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False))

    metric_col = []
    for g in group_col:
        noise = draw(st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False))
        if g == 'A':
            metric_col.append(base_metric + noise)
        else:
            metric_col.append(base_metric + effect_size + noise)

    return pd.DataFrame({
        'group': group_col,
        'metric': metric_col
    })


@st.composite
def anova_data_strategy(draw, min_rows=30, max_rows=100, n_groups=3):
    """Generate valid ANOVA data."""
    n = draw(st.integers(min_value=min_rows, max_value=max_rows))

    # Generate group assignments
    groups = [f'Group_{i}' for i in range(n_groups)]
    group_col = [groups[i % n_groups] for i in range(n)]

    # Generate metric values
    base_metric = draw(st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False))
    metric_col = [
        base_metric + draw(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False))
        for _ in range(n)
    ]

    return pd.DataFrame({
        'group': group_col,
        'metric': metric_col
    })


@st.composite
def regression_data_strategy(draw, min_rows=30, max_rows=100):
    """Generate valid regression data."""
    n = draw(st.integers(min_value=min_rows, max_value=max_rows))

    x1 = [draw(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)) for _ in range(n)]
    x2 = [draw(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)) for _ in range(n)]
    y = [x1[i] * 2 + x2[i] * 0.5 + draw(st.floats(min_value=-1, max_value=1, allow_nan=False, allow_infinity=False))
         for i in range(n)]

    return pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'y': y
    })


class TestMethodResultStandardization:
    """Property 7: Method result standardization - Validates Requirements 3.3, 3.6"""

    @given(df=ab_test_data_strategy())
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_ab_test_result_structure(self, df):
        """
        **Validates: Requirements 3.3, 3.6**

        Property: ABTest.execute() must always return a MethodResult object
        containing all required fields.
        """
        ab_test = ABTest(alpha=0.05)

        result = ab_test.execute(df, group_col='group', metric_col='metric')

        # Property: Must return MethodResult
        assert isinstance(result, MethodResult)

        # Property: Must have method_name
        assert result.method_name == "A/B Test"

        # Property: Must have parameters dict
        assert isinstance(result.parameters, dict)
        assert 'group_col' in result.parameters
        assert 'metric_col' in result.parameters

        # Property: Must have statistics dict with required keys
        assert isinstance(result.statistics, dict)
        assert 'mean_group_a' in result.statistics or 'mean_group_b' in result.statistics
        assert 't_statistic' in result.statistics

        # Property: Must have p_values dict
        assert isinstance(result.p_values, dict)
        assert len(result.p_values) > 0

        # Property: Must have effect_sizes dict
        assert isinstance(result.effect_sizes, dict)
        assert 'cohens_d' in result.effect_sizes

        # Property: Must have metadata dict
        assert isinstance(result.metadata, dict)
        assert 'significant' in result.metadata
        assert isinstance(result.metadata['significant'], bool)

    @given(df=anova_data_strategy())
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_anova_result_structure(self, df):
        """
        **Validates: Requirements 3.3, 3.6**

        Property: OneWayANOVA.execute() must always return a MethodResult object
        containing all required fields.
        """
        anova = OneWayANOVA(alpha=0.05, post_hoc=False)

        result = anova.execute(df, group_col='group', metric_col='metric')

        # Property: Must return MethodResult
        assert isinstance(result, MethodResult)

        # Property: Must have method_name
        assert result.method_name == "One-Way ANOVA"

        # Property: Must have required statistics
        assert 'f_statistic' in result.statistics
        assert 'degrees_of_freedom_between' in result.statistics
        assert 'degrees_of_freedom_within' in result.statistics

        # Property: Must have p_values
        assert 'anova_f_test' in result.p_values

        # Property: Must have effect_sizes
        assert 'eta_squared' in result.effect_sizes

        # Property: Metadata must have significant flag as bool
        assert isinstance(result.metadata['significant'], bool)

    @given(df=regression_data_strategy())
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_regression_result_structure(self, df):
        """
        **Validates: Requirements 3.3, 3.6**

        Property: LinearRegressionAnalysis.execute() must return a MethodResult
        with standardized structure.
        """
        regression = LinearRegressionAnalysis()

        result = regression.execute(df, target_col='y', feature_cols=['x1', 'x2'])

        # Property: Must return MethodResult
        assert isinstance(result, MethodResult)

        # Property: Must have method_name
        assert result.method_name == "Linear Regression"

        # Property: Must have R-squared
        assert 'r_squared' in result.statistics
        assert 0 <= result.statistics['r_squared'] <= 1

        # Property: Must have coefficients
        assert 'coef_intercept' in result.statistics

    def test_all_methods_return_methodresult(self):
        """
        **Validates: Requirements 3.3, 3.6**

        Property: All registered methods must return MethodResult objects
        with consistent structure.
        """
        # Create test data that works for multiple methods
        np.random.seed(42)
        df_ab = pd.DataFrame({
            'group': ['A', 'B'] * 25,
            'metric': np.random.normal(10, 2, 50)
        })

        df_anova = pd.DataFrame({
            'group': ['A', 'B', 'C'] * 20,
            'outcome': np.random.normal(50, 10, 60)
        })

        df_chi = pd.DataFrame({
            'treatment': ['Control', 'Treatment'] * 50,
            'outcome': np.random.choice([0, 1], 100)
        })

        df_reg = pd.DataFrame({
            'x1': np.random.normal(0, 1, 50),
            'x2': np.random.normal(0, 1, 50),
            'y': np.random.normal(0, 1, 50),
            'y_binary': np.random.choice([0, 1], 50)
        })

        test_cases = [
            (ABTest(), df_ab, {'group_col': 'group', 'metric_col': 'metric'}),
            (OneWayANOVA(post_hoc=False), df_anova, {'group_col': 'group', 'metric_col': 'outcome'}),
            (ChiSquareTest(test_type='independence'), df_chi, {'group_col': 'treatment', 'outcome_col': 'outcome'}),
            (LinearRegressionAnalysis(), df_reg, {'target_col': 'y', 'feature_cols': ['x1', 'x2']}),
        ]

        for method, data, kwargs in test_cases:
            result = method.execute(data, **kwargs)

            # Property: Must be MethodResult
            assert isinstance(result, MethodResult), f"{method.__class__.__name__} did not return MethodResult"

            # Property: Must have method_name (string)
            assert isinstance(result.method_name, str)
            assert len(result.method_name) > 0

            # Property: Must have all required dict attributes
            assert isinstance(result.parameters, dict)
            assert isinstance(result.statistics, dict)
            assert isinstance(result.p_values, dict)
            assert isinstance(result.effect_sizes, dict)
            assert isinstance(result.metadata, dict)


class TestMethodChainingCompatibility:
    """Property 8: Method chaining compatibility - Validates Requirements 3.5"""

    def test_method_chain_basic_execution(self):
        """
        **Validates: Requirements 3.5**

        Property: MethodChain should successfully execute multiple methods
        in sequence on the same data.
        """
        np.random.seed(42)
        df = pd.DataFrame({
            'group': ['A', 'B'] * 30,
            'metric': np.random.normal(10, 2, 60)
        })

        # Create a chain with multiple methods
        chain = MethodChain()
        chain.add(ABTest(alpha=0.05), group_col='group', metric_col='metric')
        chain.add(ABTest(alpha=0.01), group_col='group', metric_col='metric')

        # Execute chain
        results = chain.execute(df)

        # Property: Should return list of results
        assert isinstance(results, list)
        assert len(results) == 2

        # Property: Each result should be MethodResult
        for result in results:
            assert isinstance(result, MethodResult)

    def test_method_chain_combined_results(self):
        """
        **Validates: Requirements 3.5**

        Property: get_combined_results() should aggregate results from all methods.
        """
        np.random.seed(42)
        df = pd.DataFrame({
            'group': ['A', 'B'] * 30,
            'metric': np.random.normal(10, 2, 60)
        })

        chain = MethodChain()
        chain.add(ABTest(alpha=0.05), group_col='group', metric_col='metric')
        chain.execute(df)

        combined = chain.get_combined_results()

        # Property: Should have all expected keys
        assert 'methods_executed' in combined
        assert 'all_statistics' in combined
        assert 'all_p_values' in combined
        assert 'all_effect_sizes' in combined

        # Property: methods_executed should list method names
        assert len(combined['methods_executed']) > 0

    def test_method_chain_validation(self):
        """
        **Validates: Requirements 3.5**

        Property: MethodChain.validate_chain() should detect empty chains.
        """
        chain = MethodChain()

        is_valid, errors = chain.validate_chain()

        # Property: Empty chain should be invalid
        assert not is_valid
        assert len(errors) > 0
        assert "empty" in errors[0].lower()

    def test_method_chain_fluent_interface(self):
        """
        **Validates: Requirements 3.5**

        Property: MethodChain.add() should return self for fluent interface.
        """
        chain = MethodChain()

        result = chain.add(ABTest(), group_col='group', metric_col='metric')

        # Property: add() should return the chain for fluent chaining
        assert result is chain

    @given(
        alpha1=st.floats(min_value=0.001, max_value=0.1, allow_nan=False),
        alpha2=st.floats(min_value=0.001, max_value=0.1, allow_nan=False)
    )
    @settings(max_examples=10)
    def test_chain_with_different_parameters(self, alpha1, alpha2):
        """
        **Validates: Requirements 3.5**

        Property: Methods with different parameters should work in a chain.
        """
        np.random.seed(42)
        df = pd.DataFrame({
            'group': ['A', 'B'] * 30,
            'metric': np.random.normal(10, 2, 60)
        })

        chain = MethodChain()
        chain.add(ABTest(alpha=alpha1), group_col='group', metric_col='metric')
        chain.add(ABTest(alpha=alpha2), group_col='group', metric_col='metric')

        results = chain.execute(df)

        # Property: Both should execute successfully
        assert len(results) == 2

        # Property: Parameters should be preserved
        assert results[0].parameters['alpha'] == alpha1
        assert results[1].parameters['alpha'] == alpha2


class TestParameterValidationConsistency:
    """Property 9: Parameter validation consistency - Validates Requirements 3.4, 7.3"""

    @given(
        alpha=st.floats(min_value=0.001, max_value=0.5, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=20)
    def test_valid_alpha_parameter(self, alpha):
        """
        **Validates: Requirements 3.4, 7.3**

        Property: Methods should accept valid alpha values without error.
        """
        ab_test = ABTest(alpha=alpha)

        # Property: Should create instance without error
        assert ab_test.alpha == alpha

        # Property: Parameters should include alpha
        params = ab_test.get_parameters()
        assert 'alpha' in params

    def test_invalid_inputs_raise_methodexecutionerror(self):
        """
        **Validates: Requirements 3.4, 7.3**

        Property: validate_and_execute should raise MethodExecutionError
        for invalid inputs with clear error messages.
        """
        ab_test = ABTest()

        # Test with non-DataFrame input
        with pytest.raises(MethodExecutionError) as exc_info:
            ab_test.validate_and_execute([1, 2, 3], group_col='group', metric_col='metric')

        assert "DataFrame" in str(exc_info.value)

        # Test with missing required parameters
        df = pd.DataFrame({'group': ['A', 'B'], 'metric': [1, 2]})

        with pytest.raises(MethodExecutionError) as exc_info:
            ab_test.validate_and_execute(df)  # Missing group_col and metric_col

        # Should mention missing parameters
        assert "Missing" in str(exc_info.value) or "required" in str(exc_info.value.message).lower()

    @given(
        col_name=st.text(min_size=1, max_size=20).filter(lambda x: x.strip() and x.isidentifier())
    )
    @settings(max_examples=15)
    def test_missing_column_validation(self, col_name):
        """
        **Validates: Requirements 3.4, 7.3**

        Property: Methods should return False for validate_inputs when
        required columns are missing.
        """
        ab_test = ABTest()

        # Create DataFrame without the specified column
        df = pd.DataFrame({
            'other_col': [1, 2, 3, 4],
            'another_col': [5, 6, 7, 8]
        })

        # Assume col_name is not in our test DataFrame
        assume(col_name not in df.columns)

        # Property: Should return False for missing column
        is_valid = ab_test.validate_inputs(df, group_col=col_name, metric_col='other_col')
        assert not is_valid

    def test_validation_error_messages_are_descriptive(self):
        """
        **Validates: Requirements 3.4, 7.3**

        Property: Validation error messages should be descriptive and helpful.
        """
        ab_test = ABTest()

        # Empty DataFrame
        empty_df = pd.DataFrame()
        errors = ab_test.get_validation_errors(empty_df, group_col='group', metric_col='metric')

        # Property: Should report that DataFrame is empty
        assert len(errors) > 0
        assert any("empty" in e.lower() for e in errors)

    def test_all_methods_have_get_parameters(self):
        """
        **Validates: Requirements 3.4, 7.3**

        Property: All methods must implement get_parameters() returning
        a non-empty dictionary.
        """
        methods = [
            ABTest(),
            OneWayANOVA(),
            ChiSquareTest(test_type='independence'),
            LinearRegressionAnalysis(),
        ]

        for method in methods:
            params = method.get_parameters()

            # Property: Must return dict
            assert isinstance(params, dict), f"{method.__class__.__name__}.get_parameters() should return dict"

            # Property: Must have at least one parameter
            assert len(params) > 0, f"{method.__class__.__name__} should have parameters"

            # Property: All values should be strings (descriptions)
            for key, value in params.items():
                assert isinstance(key, str)
                assert isinstance(value, str)

    def test_method_info_completeness(self):
        """
        **Validates: Requirements 3.4, 7.3**

        Property: get_method_info() should return complete method metadata.
        """
        ab_test = ABTest()

        info = ab_test.get_method_info()

        # Property: Should have all required fields
        assert 'name' in info
        assert 'description' in info
        assert 'parameters' in info
        assert 'class' in info

        # Property: Class name should match
        assert info['class'] == 'ABTest'


class TestMethodValidation:
    """Additional validation tests for method inputs."""

    def test_ab_test_requires_exactly_two_groups(self):
        """
        Property: ABTest should only validate data with exactly 2 groups.
        """
        ab_test = ABTest()

        # One group - invalid
        df_one = pd.DataFrame({'group': ['A'] * 10, 'metric': range(10)})
        assert not ab_test.validate_inputs(df_one, group_col='group', metric_col='metric')

        # Two groups - valid
        df_two = pd.DataFrame({'group': ['A', 'B'] * 5, 'metric': range(10)})
        assert ab_test.validate_inputs(df_two, group_col='group', metric_col='metric')

        # Three groups - invalid
        df_three = pd.DataFrame({'group': ['A', 'B', 'C'] * 4, 'metric': range(12)})
        assert not ab_test.validate_inputs(df_three, group_col='group', metric_col='metric')

    def test_anova_requires_at_least_two_groups(self):
        """
        Property: ANOVA should require at least 2 groups.
        """
        anova = OneWayANOVA()

        # One group - invalid
        df_one = pd.DataFrame({'group': ['A'] * 10, 'metric': range(10)})
        assert not anova.validate_inputs(df_one, group_col='group', metric_col='metric')

        # Two groups - valid
        df_two = pd.DataFrame({'group': ['A', 'B'] * 5, 'metric': range(10)})
        assert anova.validate_inputs(df_two, group_col='group', metric_col='metric')

        # Three groups - valid
        df_three = pd.DataFrame({'group': ['A', 'B', 'C'] * 4, 'metric': range(12)})
        assert anova.validate_inputs(df_three, group_col='group', metric_col='metric')

    def test_numeric_metric_required(self):
        """
        Property: Methods requiring numeric metrics should reject non-numeric columns.
        """
        ab_test = ABTest()

        # String metric - invalid
        df_string = pd.DataFrame({
            'group': ['A', 'B'] * 5,
            'metric': ['low', 'high'] * 5
        })
        assert not ab_test.validate_inputs(df_string, group_col='group', metric_col='metric')

        # Numeric metric - valid
        df_numeric = pd.DataFrame({
            'group': ['A', 'B'] * 5,
            'metric': range(10)
        })
        assert ab_test.validate_inputs(df_numeric, group_col='group', metric_col='metric')


if __name__ == "__main__":
    pytest.main([__file__])
