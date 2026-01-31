"""
Base classes for experimental methods.

This module will be implemented in task 5.1.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any
from ..core.models import MethodResult


class ExperimentalMethod(ABC):
    """Abstract base class for all experimental methods."""
    
    @abstractmethod
    def validate_inputs(self, data: pd.DataFrame, **kwargs) -> bool:
        """Validate that inputs meet method requirements."""
        pass
    
    @abstractmethod
    def execute(self, data: pd.DataFrame, **kwargs) -> MethodResult:
        """Execute the experimental method."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get method parameters and their descriptions."""
        pass