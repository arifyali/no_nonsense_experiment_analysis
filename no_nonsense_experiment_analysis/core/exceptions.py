"""
Custom exception hierarchy for the experiment analysis package.

This module defines the exception classes used throughout the package
to provide clear error handling and user guidance.
"""

from typing import Dict, List, Optional


class AnalysisError(Exception):
    """Base exception for all analysis package errors.

    This is the root exception class that all other package-specific
    exceptions inherit from. It provides a common interface for
    error handling across the package.
    """

    # Common guidance suggestions mapped to error keywords
    GUIDANCE_MAP: Dict[str, str] = {
        'dataframe': 'Ensure input is a pandas DataFrame using pd.DataFrame()',
        'column': 'Check that the specified column exists using df.columns',
        'missing': 'Consider using clean_missing parameter in prep() step',
        'type': 'Verify data types match expected types using df.dtypes',
        'empty': 'Ensure DataFrame has rows and columns before analysis',
    }

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

    def get_suggestions(self) -> List[str]:
        """Get suggested actions to resolve this error.

        Returns:
            List of suggestion strings based on error content
        """
        suggestions = []
        message_lower = self.message.lower()

        for keyword, suggestion in self.GUIDANCE_MAP.items():
            if keyword in message_lower:
                suggestions.append(suggestion)

        return suggestions if suggestions else ["Check the error message and context for details"]

    def __repr__(self) -> str:
        """Return detailed representation with suggestions."""
        base = str(self)
        suggestions = self.get_suggestions()
        if suggestions:
            return f"{base}\nSuggestions:\n" + "\n".join(f"  - {s}" for s in suggestions)
        return base


class DataValidationError(AnalysisError):
    """Raised when data validation fails.

    This exception is raised when input data does not meet the requirements
    for analysis, such as missing required columns, invalid data types,
    or structural issues.
    """

    GUIDANCE_MAP: Dict[str, str] = {
        **AnalysisError.GUIDANCE_MAP,
        'required': 'Add the missing required columns to your DataFrame',
        'numeric': 'Convert columns to numeric using pd.to_numeric()',
        'categorical': 'Convert to categorical using df[col].astype("category")',
        'duplicate': 'Remove duplicates using remove_duplicates=True in prep()',
    }


class MethodExecutionError(AnalysisError):
    """Raised when method execution fails.

    This exception is raised when an experimental method cannot be executed
    due to invalid parameters, incompatible data, or computational errors.
    """

    GUIDANCE_MAP: Dict[str, str] = {
        **AnalysisError.GUIDANCE_MAP,
        'parameter': 'Check method documentation for required parameters',
        'group': 'Ensure group column exists and has at least 2 unique values',
        'metric': 'Ensure metric column contains numeric values',
        'sample': 'Ensure sufficient sample size for the analysis method',
        'variance': 'Check that data has variance (not all identical values)',
    }


class WorkflowError(AnalysisError):
    """Raised when workflow execution fails.

    This exception is raised when the analysis workflow encounters an error
    that prevents it from continuing, such as invalid state transitions
    or missing prerequisites.
    """

    GUIDANCE_MAP: Dict[str, str] = {
        **AnalysisError.GUIDANCE_MAP,
        'method': 'Use default_registry.list_available_methods() to see available methods',
        'prep': 'Ensure prep() step completed successfully before analyze()',
        'validation': 'Check data quality issues reported in validation warnings',
        'workflow': 'Review completed_steps in workflow state for progress',
    }