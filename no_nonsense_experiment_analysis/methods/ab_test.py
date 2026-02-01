"""
A/B Testing method implementation.

This module provides A/B testing functionality for comparing two groups
in experimental designs.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Optional
from .base import ExperimentalMethod
from ..core.models import MethodResult


class ABTest(ExperimentalMethod):
    """A/B testing method for comparing two experimental groups."""
    
    def __init__(self, alpha: float = 0.05, alternative: str = 'two-sided'):
        """Initialize A/B test parameters.
        
        Args:
            alpha: Significance level for statistical tests
            alternative: Type of alternative hypothesis ('two-sided', 'less', 'greater')
        """
        self.alpha = alpha
        self.alternative = alternative
    
    def validate_inputs(self, data: pd.DataFrame, **kwargs) -> bool:
        """Validate inputs for A/B testing.
        
        Args:
            data: DataFrame containing experimental data
            **kwargs: Additional parameters including:
                - group_col: Column name for group assignment
                - metric_col: Column name for the metric to analyze
        
        Returns:
            True if inputs are valid, False otherwise
        """
        group_col = kwargs.get('group_col')
        metric_col = kwargs.get('metric_col')
        
        if group_col is None or metric_col is None:
            return False
        
        if group_col not in data.columns or metric_col not in data.columns:
            return False
        
        # Check if we have exactly 2 groups
        unique_groups = data[group_col].nunique()
        if unique_groups != 2:
            return False
        
        # Check if metric column is numeric
        if not pd.api.types.is_numeric_dtype(data[metric_col]):
            return False
        
        return True
    
    def execute(self, data: pd.DataFrame, **kwargs) -> MethodResult:
        """Execute A/B test analysis.
        
        Args:
            data: DataFrame containing experimental data
            **kwargs: Parameters including group_col and metric_col
        
        Returns:
            MethodResult containing test statistics and results
        """
        group_col = kwargs['group_col']
        metric_col = kwargs['metric_col']
        
        # Get the two groups
        groups = data[group_col].unique()
        group_a = data[data[group_col] == groups[0]][metric_col].dropna()
        group_b = data[data[group_col] == groups[1]][metric_col].dropna()
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(group_a, group_b, alternative=self.alternative)
        
        # Calculate descriptive statistics
        mean_a = group_a.mean()
        mean_b = group_b.mean()
        std_a = group_a.std()
        std_b = group_b.std()
        n_a = len(group_a)
        n_b = len(group_b)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
        cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0
        
        # Calculate confidence interval for difference in means
        se_diff = pooled_std * np.sqrt(1/n_a + 1/n_b)
        df = n_a + n_b - 2
        t_critical = stats.t.ppf(1 - self.alpha/2, df)
        diff = mean_a - mean_b
        ci_lower = diff - t_critical * se_diff
        ci_upper = diff + t_critical * se_diff
        
        return MethodResult(
            method_name="A/B Test",
            parameters={
                'group_col': group_col,
                'metric_col': metric_col,
                'alpha': self.alpha,
                'alternative': self.alternative
            },
            statistics={
                'mean_group_a': mean_a,
                'mean_group_b': mean_b,
                'std_group_a': std_a,
                'std_group_b': std_b,
                'n_group_a': n_a,
                'n_group_b': n_b,
                't_statistic': t_stat,
                'difference_in_means': diff
            },
            p_values={
                'two_sample_ttest': p_value
            },
            confidence_intervals={
                'difference_in_means': (ci_lower, ci_upper)
            },
            effect_sizes={
                'cohens_d': cohens_d
            },
            metadata={
                'groups': list(groups),
                'significant': p_value < self.alpha,
                'interpretation': self._interpret_results(p_value, cohens_d)
            }
        )
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get method parameters and descriptions."""
        return {
            'group_col': 'Column name containing group assignments (A/B)',
            'metric_col': 'Column name containing the metric to analyze',
            'alpha': f'Significance level (default: {self.alpha})',
            'alternative': f'Alternative hypothesis type (default: {self.alternative})'
        }
    
    def _interpret_results(self, p_value: float, cohens_d: float) -> str:
        """Interpret the results of the A/B test."""
        significance = "significant" if p_value < self.alpha else "not significant"
        
        # Interpret effect size
        if abs(cohens_d) < 0.2:
            effect_size = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_size = "small"
        elif abs(cohens_d) < 0.8:
            effect_size = "medium"
        else:
            effect_size = "large"
        
        return f"The difference between groups is {significance} (p={p_value:.4f}) with a {effect_size} effect size (Cohen's d={cohens_d:.3f})"