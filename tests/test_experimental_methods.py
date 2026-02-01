"""
Tests for experimental methods implementations.

This module tests the basic functionality of all experimental methods
to ensure they work correctly with sample data.
"""

import pytest
import pandas as pd
import numpy as np
from no_nonsense_experiment_analysis.methods import (
    ABTest, OneWayANOVA, ChiSquareTest, 
    LinearRegressionAnalysis, LogisticRegressionAnalysis,
    default_registry
)


@pytest.fixture
def sample_ab_data():
    """Create sample A/B test data."""
    np.random.seed(42)
    n = 100
    data = pd.DataFrame({
        'group': np.random.choice(['A', 'B'], n),
        'metric': np.random.normal(10, 2, n)
    })
    # Make group B slightly different
    data.loc[data['group'] == 'B', 'metric'] += 1
    return data


@pytest.fixture
def sample_anova_data():
    """Create sample ANOVA data."""
    np.random.seed(42)
    n = 150
    data = pd.DataFrame({
        'group': np.random.choice(['A', 'B', 'C'], n),
        'outcome': np.random.normal(50, 10, n)
    })
    return data


@pytest.fixture
def sample_chi_data():
    """Create sample chi-square data."""
    np.random.seed(42)
    n = 200
    data = pd.DataFrame({
        'treatment': np.random.choice(['Control', 'Treatment'], n),
        'success': np.random.choice([0, 1], n, p=[0.6, 0.4])
    })
    return data


@pytest.fixture
def sample_regression_data():
    """Create sample regression data."""
    np.random.seed(42)
    n = 100
    data = pd.DataFrame({
        'x1': np.random.normal(0, 1, n),
        'x2': np.random.normal(0, 1, n),
        'y_continuous': np.random.normal(0, 1, n),
        'y_binary': np.random.choice([0, 1], n)
    })
    return data


class TestABTest:
    """Test A/B testing functionality."""
    
    def test_ab_test_validation(self, sample_ab_data):
        """Test input validation for A/B test."""
        ab_test = ABTest()
        
        # Valid inputs
        assert ab_test.validate_inputs(sample_ab_data, group_col='group', metric_col='metric')
        
        # Invalid inputs
        assert not ab_test.validate_inputs(sample_ab_data, group_col='nonexistent', metric_col='metric')
        assert not ab_test.validate_inputs(sample_ab_data, group_col='group', metric_col='nonexistent')
    
    def test_ab_test_execution(self, sample_ab_data):
        """Test A/B test execution."""
        ab_test = ABTest(alpha=0.05)
        result = ab_test.execute(sample_ab_data, group_col='group', metric_col='metric')
        
        assert result.method_name == "A/B Test"
        assert 'mean_group_a' in result.statistics
        assert 'mean_group_b' in result.statistics
        assert 'two_sample_ttest' in result.p_values
        assert 'cohens_d' in result.effect_sizes
        assert isinstance(result.metadata['significant'], bool)
    
    def test_ab_test_parameters(self):
        """Test A/B test parameter retrieval."""
        ab_test = ABTest()
        params = ab_test.get_parameters()
        
        assert 'group_col' in params
        assert 'metric_col' in params
        assert 'alpha' in params


class TestOneWayANOVA:
    """Test ANOVA functionality."""
    
    def test_anova_validation(self, sample_anova_data):
        """Test input validation for ANOVA."""
        anova = OneWayANOVA()
        
        # Valid inputs
        assert anova.validate_inputs(sample_anova_data, group_col='group', metric_col='outcome')
        
        # Invalid inputs
        assert not anova.validate_inputs(sample_anova_data, group_col='nonexistent', metric_col='outcome')
    
    def test_anova_execution(self, sample_anova_data):
        """Test ANOVA execution."""
        anova = OneWayANOVA(alpha=0.05, post_hoc=False)
        result = anova.execute(sample_anova_data, group_col='group', metric_col='outcome')
        
        assert result.method_name == "One-Way ANOVA"
        assert 'f_statistic' in result.statistics
        assert 'anova_f_test' in result.p_values
        assert 'eta_squared' in result.effect_sizes
        assert isinstance(result.metadata['significant'], bool)


class TestChiSquareTest:
    """Test Chi-square functionality."""
    
    def test_chi_square_validation(self, sample_chi_data):
        """Test input validation for chi-square test."""
        chi_test = ChiSquareTest(test_type='independence')
        
        # Valid inputs
        assert chi_test.validate_inputs(sample_chi_data, group_col='treatment', outcome_col='success')
        
        # Invalid inputs
        assert not chi_test.validate_inputs(sample_chi_data, group_col='nonexistent', outcome_col='success')
    
    def test_chi_square_execution(self, sample_chi_data):
        """Test chi-square test execution."""
        chi_test = ChiSquareTest(test_type='independence')
        result = chi_test.execute(sample_chi_data, group_col='treatment', outcome_col='success')
        
        assert result.method_name == "Chi-Square Test of Independence"
        assert 'chi2_statistic' in result.statistics
        assert 'chi2_test' in result.p_values
        assert 'cramers_v' in result.effect_sizes
        assert isinstance(result.metadata['significant'], bool)


class TestLinearRegression:
    """Test linear regression functionality."""
    
    def test_linear_regression_validation(self, sample_regression_data):
        """Test input validation for linear regression."""
        regression = LinearRegressionAnalysis()
        
        # Valid inputs
        assert regression.validate_inputs(sample_regression_data, 
                                        target_col='y_continuous', 
                                        feature_cols=['x1', 'x2'])
        
        # Invalid inputs
        assert not regression.validate_inputs(sample_regression_data, 
                                            target_col='nonexistent', 
                                            feature_cols=['x1', 'x2'])
    
    def test_linear_regression_execution(self, sample_regression_data):
        """Test linear regression execution."""
        regression = LinearRegressionAnalysis()
        result = regression.execute(sample_regression_data, 
                                  target_col='y_continuous', 
                                  feature_cols=['x1', 'x2'])
        
        assert result.method_name == "Linear Regression"
        assert 'r_squared' in result.statistics
        assert 'f_statistic' in result.statistics
        assert 'f_test' in result.p_values
        assert 'coef_intercept' in result.statistics
        assert 'coef_x1' in result.statistics


class TestLogisticRegression:
    """Test logistic regression functionality."""
    
    def test_logistic_regression_validation(self, sample_regression_data):
        """Test input validation for logistic regression."""
        logistic = LogisticRegressionAnalysis()
        
        # Valid inputs
        assert logistic.validate_inputs(sample_regression_data, 
                                      target_col='y_binary', 
                                      feature_cols=['x1', 'x2'])
        
        # Invalid inputs
        assert not logistic.validate_inputs(sample_regression_data, 
                                          target_col='y_continuous',  # Not binary
                                          feature_cols=['x1', 'x2'])
    
    def test_logistic_regression_execution(self, sample_regression_data):
        """Test logistic regression execution."""
        logistic = LogisticRegressionAnalysis()
        result = logistic.execute(sample_regression_data, 
                                target_col='y_binary', 
                                feature_cols=['x1', 'x2'])
        
        assert result.method_name == "Logistic Regression"
        assert 'accuracy' in result.statistics
        assert 'mcfadden_r2' in result.statistics
        assert 'coef_intercept' in result.statistics
        assert 'odds_ratio_x1' in result.statistics


class TestMethodRegistry:
    """Test method registry functionality."""
    
    def test_registry_list_methods(self):
        """Test listing available methods."""
        methods = default_registry.list_available_methods()
        
        expected_methods = [
            'ab_test', 'one_way_anova', 'chi_square_independence',
            'chi_square_goodness_of_fit', 'linear_regression', 'logistic_regression'
        ]
        
        for method in expected_methods:
            assert method in methods
    
    def test_registry_get_method(self):
        """Test getting method instances."""
        ab_test = default_registry.get_method('ab_test')
        assert isinstance(ab_test, ABTest)
        
        anova = default_registry.get_method('one_way_anova', alpha=0.01)
        assert isinstance(anova, OneWayANOVA)
        assert anova.alpha == 0.01
    
    def test_registry_method_info(self):
        """Test getting method information."""
        info = default_registry.get_method_info('ab_test')
        
        assert info['name'] == 'ab_test'
        assert 'parameters' in info
        assert 'group_col' in info['parameters']
        assert 'metric_col' in info['parameters']
    
    def test_registry_invalid_method(self):
        """Test error handling for invalid method names."""
        with pytest.raises(KeyError):
            default_registry.get_method('nonexistent_method')


if __name__ == "__main__":
    pytest.main([__file__])