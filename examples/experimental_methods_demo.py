"""
Demonstration of experimental methods in no_nonsense_experiment_analysis.

This script shows how to use the various experimental design methods
implemented in the package.
"""

import pandas as pd
import numpy as np
from no_nonsense_experiment_analysis.methods import (
    ABTest, OneWayANOVA, ChiSquareTest, 
    LinearRegressionAnalysis, LogisticRegressionAnalysis,
    default_registry
)

# Set random seed for reproducibility
np.random.seed(42)

def create_sample_data():
    """Create sample experimental data for demonstration."""
    n = 300
    
    # A/B test data
    ab_data = pd.DataFrame({
        'group': np.random.choice(['A', 'B'], n),
        'conversion_rate': np.random.normal(0.15, 0.05, n),
        'revenue': np.random.normal(100, 25, n)
    })
    # Make group B slightly better
    ab_data.loc[ab_data['group'] == 'B', 'conversion_rate'] += 0.03
    ab_data.loc[ab_data['group'] == 'B', 'revenue'] += 15
    
    # ANOVA data (3 groups)
    anova_data = pd.DataFrame({
        'treatment': np.random.choice(['Control', 'Treatment1', 'Treatment2'], n),
        'outcome': np.random.normal(50, 10, n)
    })
    # Add treatment effects
    anova_data.loc[anova_data['treatment'] == 'Treatment1', 'outcome'] += 5
    anova_data.loc[anova_data['treatment'] == 'Treatment2', 'outcome'] += 10
    
    # Chi-square data
    chi_data = pd.DataFrame({
        'group': np.random.choice(['Control', 'Treatment'], n),
        'success': np.random.choice([0, 1], n, p=[0.7, 0.3])
    })
    # Make treatment more successful
    treatment_mask = chi_data['group'] == 'Treatment'
    chi_data.loc[treatment_mask, 'success'] = np.random.choice([0, 1], treatment_mask.sum(), p=[0.5, 0.5])
    
    # Regression data
    regression_data = pd.DataFrame({
        'x1': np.random.normal(0, 1, n),
        'x2': np.random.normal(0, 1, n),
        'x3': np.random.normal(0, 1, n)
    })
    # Create dependent variable with some relationship
    regression_data['y_continuous'] = (2 * regression_data['x1'] + 
                                     1.5 * regression_data['x2'] + 
                                     np.random.normal(0, 1, n))
    
    # Binary outcome for logistic regression
    prob = 1 / (1 + np.exp(-(regression_data['x1'] + 0.5 * regression_data['x2'])))
    regression_data['y_binary'] = np.random.binomial(1, prob, n)
    
    return ab_data, anova_data, chi_data, regression_data

def demonstrate_ab_test(data):
    """Demonstrate A/B testing."""
    print("=== A/B Test Demonstration ===")
    
    ab_test = ABTest(alpha=0.05)
    
    # Test conversion rate
    result = ab_test.execute(data, group_col='group', metric_col='conversion_rate')
    
    print(f"Method: {result.method_name}")
    print(f"Groups compared: {result.metadata['groups']}")
    print(f"Mean Group A: {result.statistics['mean_group_a']:.4f}")
    print(f"Mean Group B: {result.statistics['mean_group_b']:.4f}")
    print(f"P-value: {result.p_values['two_sample_ttest']:.4f}")
    print(f"Effect size (Cohen's d): {result.effect_sizes['cohens_d']:.4f}")
    print(f"Significant: {result.metadata['significant']}")
    print(f"Interpretation: {result.metadata['interpretation']}")
    print()

def demonstrate_anova(data):
    """Demonstrate ANOVA."""
    print("=== ANOVA Demonstration ===")
    
    anova = OneWayANOVA(alpha=0.05, post_hoc=True)
    
    result = anova.execute(data, group_col='treatment', metric_col='outcome')
    
    print(f"Method: {result.method_name}")
    print(f"F-statistic: {result.statistics['f_statistic']:.4f}")
    print(f"P-value: {result.p_values['anova_f_test']:.4f}")
    print(f"Effect size (η²): {result.effect_sizes['eta_squared']:.4f}")
    print(f"Significant: {result.metadata['significant']}")
    print(f"Interpretation: {result.metadata['interpretation']}")
    
    if 'post_hoc_results' in result.metadata:
        print("Post-hoc comparisons:")
        for comparison, stats in result.metadata['post_hoc_results'].items():
            print(f"  {comparison}: p={stats['p_value_bonferroni']:.4f}, significant={stats['significant']}")
    print()

def demonstrate_chi_square(data):
    """Demonstrate Chi-square test."""
    print("=== Chi-Square Test Demonstration ===")
    
    chi_test = ChiSquareTest(alpha=0.05, test_type='independence')
    
    result = chi_test.execute(data, group_col='group', outcome_col='success')
    
    print(f"Method: {result.method_name}")
    print(f"Chi-square statistic: {result.statistics['chi2_statistic']:.4f}")
    print(f"P-value: {result.p_values['chi2_test']:.4f}")
    print(f"Effect size (Cramér's V): {result.effect_sizes['cramers_v']:.4f}")
    print(f"Significant: {result.metadata['significant']}")
    print(f"Interpretation: {result.metadata['interpretation']}")
    print(f"Assumptions met: {result.metadata['assumptions_met']}")
    print()

def demonstrate_linear_regression(data):
    """Demonstrate linear regression."""
    print("=== Linear Regression Demonstration ===")
    
    regression = LinearRegressionAnalysis(alpha=0.05)
    
    result = regression.execute(data, 
                              target_col='y_continuous', 
                              feature_cols=['x1', 'x2', 'x3'])
    
    print(f"Method: {result.method_name}")
    print(f"R-squared: {result.statistics['r_squared']:.4f}")
    print(f"Adjusted R-squared: {result.statistics['adjusted_r_squared']:.4f}")
    print(f"F-statistic: {result.statistics['f_statistic']:.4f}")
    print(f"F-test p-value: {result.p_values['f_test']:.4f}")
    print(f"RMSE: {result.statistics['rmse']:.4f}")
    
    print("Coefficients:")
    for coef in ['intercept', 'x1', 'x2', 'x3']:
        coef_val = result.statistics.get(f'coef_{coef}', 'N/A')
        p_val = result.p_values.get(f'coef_{coef}', 'N/A')
        if coef_val != 'N/A' and p_val != 'N/A':
            print(f"  {coef}: {coef_val:.4f} (p={p_val:.4f})")
    
    print(f"Interpretation: {result.metadata['interpretation']}")
    print()

def demonstrate_logistic_regression(data):
    """Demonstrate logistic regression."""
    print("=== Logistic Regression Demonstration ===")
    
    logistic = LogisticRegressionAnalysis(alpha=0.05)
    
    result = logistic.execute(data, 
                            target_col='y_binary', 
                            feature_cols=['x1', 'x2', 'x3'])
    
    print(f"Method: {result.method_name}")
    print(f"Accuracy: {result.statistics['accuracy']:.4f}")
    print(f"McFadden's R²: {result.statistics['mcfadden_r2']:.4f}")
    
    print("Coefficients and Odds Ratios:")
    for coef in ['intercept', 'x1', 'x2', 'x3']:
        coef_val = result.statistics.get(f'coef_{coef}', 'N/A')
        or_val = result.statistics.get(f'odds_ratio_{coef}', 'N/A')
        if coef_val != 'N/A' and or_val != 'N/A':
            print(f"  {coef}: coef={coef_val:.4f}, OR={or_val:.4f}")
    
    print(f"Interpretation: {result.metadata['interpretation']}")
    print()

def demonstrate_registry():
    """Demonstrate the method registry."""
    print("=== Method Registry Demonstration ===")
    
    print("Available methods:")
    for method_name in default_registry.list_available_methods():
        print(f"  - {method_name}")
    
    print("\nMethod information:")
    info = default_registry.get_method_info('ab_test')
    print(f"Method: {info['name']}")
    print(f"Class: {info['class']}")
    print("Parameters:")
    for param, desc in info['parameters'].items():
        print(f"  - {param}: {desc}")
    print()

def main():
    """Run all demonstrations."""
    print("No-Nonsense Experiment Analysis - Methods Demonstration")
    print("=" * 60)
    print()
    
    # Create sample data
    ab_data, anova_data, chi_data, regression_data = create_sample_data()
    
    # Demonstrate each method
    demonstrate_ab_test(ab_data)
    demonstrate_anova(anova_data)
    demonstrate_chi_square(chi_data)
    demonstrate_linear_regression(regression_data)
    demonstrate_logistic_regression(regression_data)
    demonstrate_registry()
    
    print("Demonstration complete!")

if __name__ == "__main__":
    main()