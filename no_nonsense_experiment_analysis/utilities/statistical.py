"""
Statistical utility functions.

This module will be implemented in task 6.1.
"""

import pandas as pd
from typing import Callable, List, Tuple


class StatisticalFunctions:
    """Common statistical operations for experimental analysis."""
    
    @staticmethod
    def calculate_effect_size(group1: pd.Series, group2: pd.Series) -> float:
        """Calculate effect size between two groups."""
        # TODO: Implement in task 6.1
        pass
    
    @staticmethod
    def bootstrap_confidence_interval(data: pd.Series, statistic: Callable) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for a statistic."""
        # TODO: Implement in task 6.1
        pass
    
    @staticmethod
    def multiple_comparison_correction(p_values: List[float], method: str = "bonferroni") -> List[float]:
        """Apply multiple comparison correction to p-values."""
        # TODO: Implement in task 6.1
        pass