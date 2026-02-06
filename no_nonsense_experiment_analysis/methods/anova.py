"""
ANOVA (Analysis of Variance) method implementation.

This module provides ANOVA functionality for comparing multiple groups
in experimental designs.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Tuple
from .base import ExperimentalMethod
from ..core.models import MethodResult


class OneWayANOVA(ExperimentalMethod):
    """One-way ANOVA for comparing multiple experimental groups."""

    # Class-level attributes
    method_name = "One-Way ANOVA"
    method_description = "Compare multiple groups using analysis of variance"
    required_params = ['group_col', 'metric_col']

    def __init__(self, alpha: float = 0.05, post_hoc: bool = True):
        """Initialize ANOVA parameters.
        
        Args:
            alpha: Significance level for statistical tests
            post_hoc: Whether to perform post-hoc pairwise comparisons
        """
        self.alpha = alpha
        self.post_hoc = post_hoc
    
    def validate_inputs(self, data: pd.DataFrame, **kwargs) -> bool:
        """Validate inputs for ANOVA.
        
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
        
        # Check if we have at least 2 groups
        unique_groups = data[group_col].nunique()
        if unique_groups < 2:
            return False
        
        # Check if metric column is numeric
        if not pd.api.types.is_numeric_dtype(data[metric_col]):
            return False
        
        # Check if each group has at least 2 observations
        group_sizes = data.groupby(group_col)[metric_col].count()
        if (group_sizes < 2).any():
            return False
        
        return True
    
    def execute(self, data: pd.DataFrame, **kwargs) -> MethodResult:
        """Execute one-way ANOVA analysis.
        
        Args:
            data: DataFrame containing experimental data
            **kwargs: Parameters including group_col and metric_col
        
        Returns:
            MethodResult containing ANOVA statistics and results
        """
        group_col = kwargs['group_col']
        metric_col = kwargs['metric_col']
        
        # Prepare data for ANOVA
        groups = []
        group_names = []
        for name, group in data.groupby(group_col):
            group_data = group[metric_col].dropna()
            if len(group_data) > 0:
                groups.append(group_data)
                group_names.append(name)
        
        # Perform one-way ANOVA
        f_stat, p_value = stats.f_oneway(*groups)
        
        # Calculate descriptive statistics
        group_stats = {}
        total_n = 0
        for i, (name, group_data) in enumerate(zip(group_names, groups)):
            group_stats[f'mean_{name}'] = group_data.mean()
            group_stats[f'std_{name}'] = group_data.std()
            group_stats[f'n_{name}'] = len(group_data)
            total_n += len(group_data)
        
        # Calculate effect size (eta-squared)
        # First, we need to calculate sum of squares
        all_data = pd.concat(groups)
        grand_mean = all_data.mean()
        
        # Between-group sum of squares
        ss_between = sum(len(group) * (group.mean() - grand_mean)**2 for group in groups)
        
        # Total sum of squares
        ss_total = sum((all_data - grand_mean)**2)
        
        # Effect size (eta-squared)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        result = MethodResult(
            method_name="One-Way ANOVA",
            parameters={
                'group_col': group_col,
                'metric_col': metric_col,
                'alpha': self.alpha,
                'post_hoc': self.post_hoc
            },
            statistics={
                'f_statistic': f_stat,
                'degrees_of_freedom_between': len(groups) - 1,
                'degrees_of_freedom_within': total_n - len(groups),
                'grand_mean': grand_mean,
                **group_stats
            },
            p_values={
                'anova_f_test': p_value
            },
            confidence_intervals={},
            effect_sizes={
                'eta_squared': eta_squared
            },
            metadata={
                'groups': group_names,
                'significant': bool(p_value < self.alpha),
                'interpretation': self._interpret_results(p_value, eta_squared, len(groups))
            }
        )
        
        # Perform post-hoc tests if requested and ANOVA is significant
        if self.post_hoc and p_value < self.alpha:
            pairwise_results = self._perform_post_hoc(groups, group_names)
            result.metadata['post_hoc_results'] = pairwise_results
        
        return result
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get method parameters and descriptions."""
        return {
            'group_col': 'Column name containing group assignments',
            'metric_col': 'Column name containing the metric to analyze',
            'alpha': f'Significance level (default: {self.alpha})',
            'post_hoc': f'Perform post-hoc pairwise comparisons (default: {self.post_hoc})'
        }
    
    def _perform_post_hoc(self, groups: List[np.ndarray], group_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Perform Tukey's HSD post-hoc test."""
        pairwise_results = {}
        
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                # Perform pairwise t-test with Bonferroni correction
                t_stat, p_val = stats.ttest_ind(groups[i], groups[j])
                
                # Apply Bonferroni correction
                n_comparisons = len(groups) * (len(groups) - 1) / 2
                p_val_corrected = min(p_val * n_comparisons, 1.0)
                
                pair_name = f"{group_names[i]}_vs_{group_names[j]}"
                pairwise_results[pair_name] = {
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'p_value_bonferroni': p_val_corrected,
                    'significant': bool(p_val_corrected < self.alpha)
                }
        
        return pairwise_results
    
    def _interpret_results(self, p_value: float, eta_squared: float, n_groups: int) -> str:
        """Interpret the results of the ANOVA."""
        significance = "significant" if p_value < self.alpha else "not significant"
        
        # Interpret effect size (eta-squared)
        if eta_squared < 0.01:
            effect_size = "negligible"
        elif eta_squared < 0.06:
            effect_size = "small"
        elif eta_squared < 0.14:
            effect_size = "medium"
        else:
            effect_size = "large"
        
        return f"The difference between {n_groups} groups is {significance} (p={p_value:.4f}) with a {effect_size} effect size (η²={eta_squared:.3f})"