"""
no_nonsense_experiment_analysis

A Python companion library for experimental data analysis, designed to work 
alongside the no_nonsense_experimental_design repository.

This package provides data scientists and analysts with a streamlined toolkit 
for experimental data analysis, following a clear workflow from data preparation 
through analysis to reporting.
"""

__version__ = "0.1.0"
__author__ = "Arif Ali"

# Import main modules for easy access
from . import data_prep
from . import methods
from . import utilities

# Import core classes for direct access
from .core.models import ValidationResult, MethodResult, AnalysisReport, WorkflowState
from .core.exceptions import (
    AnalysisError,
    DataValidationError, 
    MethodExecutionError,
    WorkflowError
)
from .core.workflow import WorkflowManager

__all__ = [
    "data_prep",
    "methods", 
    "utilities",
    "ValidationResult",
    "MethodResult",
    "AnalysisReport",
    "WorkflowState",
    "AnalysisError",
    "DataValidationError",
    "MethodExecutionError", 
    "WorkflowError",
    "WorkflowManager"
]