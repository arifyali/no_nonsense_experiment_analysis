"""
Custom exception hierarchy for the experiment analysis package.

This module defines the exception classes used throughout the package
to provide clear error handling and user guidance.
"""


class AnalysisError(Exception):
    """Base exception for all analysis package errors.
    
    This is the root exception class that all other package-specific
    exceptions inherit from. It provides a common interface for
    error handling across the package.
    """
    
    def __init__(self, message: str, context: dict = None):
        """Initialize the analysis error.
        
        Args:
            message: Human-readable error message
            context: Optional dictionary with additional error context
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}
    
    def __str__(self) -> str:
        """Return a formatted error message with context."""
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


class DataValidationError(AnalysisError):
    """Raised when data validation fails.
    
    This exception is raised when input data does not meet the requirements
    for analysis, such as missing required columns, invalid data types,
    or structural issues.
    """
    pass


class MethodExecutionError(AnalysisError):
    """Raised when method execution fails.
    
    This exception is raised when an experimental method cannot be executed
    due to invalid parameters, incompatible data, or computational errors.
    """
    pass


class WorkflowError(AnalysisError):
    """Raised when workflow execution fails.
    
    This exception is raised when the analysis workflow encounters an error
    that prevents it from continuing, such as invalid state transitions
    or missing prerequisites.
    """
    pass