"""
no_nonsense_experiment_analysis

A Python companion library for experimental data analysis, designed to work 
alongside The No-Nonsense Guide to Experimental Design.

This package provides data scientists and analysts with a streamlined toolkit 
for experimental data analysis, implementing the practical methodologies outlined 
in the companion experimental design guide. The package follows a clear workflow 
from data preparation through analysis to reporting.

For detailed method documentation, see docs/METHODS.md
For theoretical background, see: 
https://github.com/mustafaysir/no_nonsense_experimental_design
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