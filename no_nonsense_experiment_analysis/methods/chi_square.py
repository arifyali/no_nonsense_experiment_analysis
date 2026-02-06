"""
Chi-square test method implementation.

This module provides chi-square testing functionality for analyzing
categorical data in experimental designs.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, Tuple
from .base import ExperimentalMethod
from ..core.models import MethodResult


class ChiSquareTest(ExperimentalMethod):
    """Chi-square test for independence and goodness of fit."""

    # Class-level attributes
    method_name = "Chi-Square Test"
    method_description = "Analyze categorical data using chi-square test"
    # Note: required_params depends on test_type, handled in __init__
    required_params = ['group_col', 'outcome_col']  # Default for independence test

    def __init__(self, alpha: float = 0.05, test_type: str = 'independence'):
        """Initialize chi-square test parameters.
        
        Args:
            alpha: Significance level for statistical tests
            test_type: Type of chi-square test ('independence' or 'goodness_of_fit')
        """
        self.alpha = alpha
        self.test_type = test_type
    
    def validate_inputs(self, data: pd.DataFrame, **kwargs) -> bool:
        """Validate inputs for chi-square test.
        
        Args:
            data: DataFrame containing experimental data
            **kwargs: Additional parameters including:
                - For independence test: group_col, outcome_col
                - For goodness of fit: observed_col, expected (optional)
        
        Returns:
            True if inputs are valid, False otherwise
        """
        if self.test_type == 'independence':
            group_col = kwargs.get('group_col')
            outcome_col = kwargs.get('outcome_col')
            
            if group_col is None or outcome_col is None:
                return False
            
            if group_col not in data.columns or outcome_col not in data.columns:
                return False
            
            # Check if both columns are categorical or can be treated as such
            return True
            
        elif self.test_type == 'goodness_of_fit':
            observed_col = kwargs.get('observed_col')
            
            if observed_col is None:
                return False
            
            if observed_col not in data.columns:
                return False
            
            return True
        
        return False
    
    def execute(self, data: pd.DataFrame, **kwargs) -> MethodResult:
        """Execute chi-square test analysis.
        
        Args:
            data: DataFrame containing experimental data
            **kwargs: Parameters specific to test type
        
        Returns:
            MethodResult containing chi-square statistics and results
        """
        if self.test_type == 'independence':
            return self._test_independence(data, **kwargs)
        elif self.test_type == 'goodness_of_fit':
            return self._test_goodness_of_fit(data, **kwargs)
        else:
            raise ValueError(f"Unknown test type: {self.test_type}")
    
    def _test_independence(self, data: pd.DataFrame, **kwargs) -> MethodResult:
        """Perform chi-square test of independence."""
        group_col = kwargs['group_col']
        outcome_col = kwargs['outcome_col']
        
        # Create contingency table
        contingency_table = pd.crosstab(data[group_col], data[outcome_col])
        
        # Perform chi-square test
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        # Calculate effect size (Cramér's V)
        n = contingency_table.sum().sum()
        cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
        
        # Calculate residuals
        residuals = (contingency_table - expected) / np.sqrt(expected)
        
        return MethodResult(
            method_name="Chi-Square Test of Independence",
            parameters={
                'group_col': group_col,
                'outcome_col': outcome_col,
                'alpha': self.alpha
            },
            statistics={
                'chi2_statistic': chi2_stat,
                'degrees_of_freedom': dof,
                'sample_size': n,
                'contingency_table': contingency_table.to_dict(),
                'expected_frequencies': expected.tolist(),
                'standardized_residuals': residuals.to_dict()
            },
            p_values={
                'chi2_test': p_value
            },
            confidence_intervals={},
            effect_sizes={
                'cramers_v': cramers_v
            },
            metadata={
                'significant': bool(p_value < self.alpha),
                'interpretation': self._interpret_independence_results(p_value, cramers_v, contingency_table.shape),
                'assumptions_met': bool(self._check_assumptions(expected))
            }
        )
    
    def _test_goodness_of_fit(self, data: pd.DataFrame, **kwargs) -> MethodResult:
        """Perform chi-square goodness of fit test."""
        observed_col = kwargs['observed_col']
        expected = kwargs.get('expected', None)
        
        # Get observed frequencies
        observed_counts = data[observed_col].value_counts().sort_index()
        
        # Set expected frequencies
        if expected is None:
            # Assume uniform distribution
            expected_freq = [len(data) / len(observed_counts)] * len(observed_counts)
        else:
            expected_freq = expected
        
        # Perform chi-square test
        chi2_stat, p_value = stats.chisquare(observed_counts, expected_freq)
        dof = len(observed_counts) - 1
        
        # Calculate effect size
        n = sum(observed_counts)
        cramers_v = np.sqrt(chi2_stat / n)
        
        return MethodResult(
            method_name="Chi-Square Goodness of Fit Test",
            parameters={
                'observed_col': observed_col,
                'alpha': self.alpha,
                'expected_provided': expected is not None
            },
            statistics={
                'chi2_statistic': chi2_stat,
                'degrees_of_freedom': dof,
                'sample_size': n,
                'observed_frequencies': observed_counts.to_dict(),
                'expected_frequencies': dict(zip(observed_counts.index, expected_freq))
            },
            p_values={
                'chi2_test': p_value
            },
            confidence_intervals={},
            effect_sizes={
                'cramers_v': cramers_v
            },
            metadata={
                'significant': bool(p_value < self.alpha),
                'interpretation': self._interpret_goodness_of_fit_results(p_value, cramers_v),
                'assumptions_met': bool(all(f >= 5 for f in expected_freq))
            }
        )
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get method parameters and descriptions."""
        if self.test_type == 'independence':
            return {
                'group_col': 'Column name containing group assignments',
                'outcome_col': 'Column name containing categorical outcomes',
                'alpha': f'Significance level (default: {self.alpha})'
            }
        else:
            return {
                'observed_col': 'Column name containing observed categorical data',
                'expected': 'List of expected frequencies (optional, defaults to uniform)',
                'alpha': f'Significance level (default: {self.alpha})'
            }
    
    def _check_assumptions(self, expected: np.ndarray) -> bool:
        """Check if chi-square test assumptions are met."""
        # All expected frequencies should be >= 5
        return np.all(expected >= 5)
    
    def _interpret_independence_results(self, p_value: float, cramers_v: float, table_shape: Tuple[int, int]) -> str:
        """Interpret the results of the independence test."""
        significance = "significant" if p_value < self.alpha else "not significant"
        
        # Interpret effect size (Cramér's V)
        if cramers_v < 0.1:
            effect_size = "negligible"
        elif cramers_v < 0.3:
            effect_size = "small"
        elif cramers_v < 0.5:
            effect_size = "medium"
        else:
            effect_size = "large"
        
        return f"The association between variables is {significance} (p={p_value:.4f}) with a {effect_size} effect size (Cramér's V={cramers_v:.3f})"
    
    def _interpret_goodness_of_fit_results(self, p_value: float, cramers_v: float) -> str:
        """Interpret the results of the goodness of fit test."""
        significance = "significant" if p_value < self.alpha else "not significant"
        
        if p_value < self.alpha:
            interpretation = "The observed distribution significantly differs from the expected distribution"
        else:
            interpretation = "The observed distribution does not significantly differ from the expected distribution"
        
        return f"{interpretation} (p={p_value:.4f}, Cramér's V={cramers_v:.3f})"