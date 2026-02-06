"""
Base classes for experimental methods.

This module provides the abstract base class for all experimental methods,
defining the interface and common functionality for method execution,
validation, and chaining.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from ..core.models import MethodResult
from ..core.exceptions import MethodExecutionError


class ExperimentalMethod(ABC):
    """Abstract base class for all experimental methods.

    This class defines the interface that all experimental methods must implement.
    It provides common functionality for input validation, parameter handling,
    and method chaining support.

    Attributes:
        method_name: Human-readable name of the method
        method_description: Brief description of what the method does
        required_params: List of required parameter names
        optional_params: Dictionary of optional parameters with defaults
    """

    # Class-level attributes to be overridden by subclasses
    method_name: str = "Experimental Method"
    method_description: str = "Base experimental method"
    required_params: List[str] = []
    optional_params: Dict[str, Any] = {}

    @abstractmethod
    def validate_inputs(self, data: pd.DataFrame, **kwargs) -> bool:
        """Validate that inputs meet method requirements.

        Args:
            data: Input DataFrame to validate
            **kwargs: Method-specific parameters

        Returns:
            True if inputs are valid, False otherwise
        """
        pass

    @abstractmethod
    def execute(self, data: pd.DataFrame, **kwargs) -> MethodResult:
        """Execute the experimental method.

        Args:
            data: Input DataFrame for analysis
            **kwargs: Method-specific parameters

        Returns:
            MethodResult containing analysis results

        Raises:
            MethodExecutionError: If execution fails
        """
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get method parameters and their descriptions.

        Returns:
            Dictionary mapping parameter names to descriptions
        """
        pass

    def validate_and_execute(self, data: pd.DataFrame, **kwargs) -> MethodResult:
        """Validate inputs and execute the method.

        This is a convenience method that combines validation and execution
        with proper error handling.

        Args:
            data: Input DataFrame for analysis
            **kwargs: Method-specific parameters

        Returns:
            MethodResult containing analysis results

        Raises:
            MethodExecutionError: If validation fails or execution errors occur
        """
        # Validate DataFrame type
        if not isinstance(data, pd.DataFrame):
            raise MethodExecutionError(
                f"Input must be a pandas DataFrame, got {type(data).__name__}",
                context={"method": self.method_name}
            )

        # Check required parameters
        missing_params = self._check_required_params(**kwargs)
        if missing_params:
            raise MethodExecutionError(
                f"Missing required parameters: {missing_params}",
                context={"method": self.method_name, "missing": missing_params}
            )

        # Validate inputs using method-specific validation
        if not self.validate_inputs(data, **kwargs):
            validation_errors = self.get_validation_errors(data, **kwargs)
            raise MethodExecutionError(
                f"Input validation failed: {validation_errors}",
                context={"method": self.method_name, "errors": validation_errors}
            )

        # Execute the method
        try:
            return self.execute(data, **kwargs)
        except MethodExecutionError:
            raise
        except Exception as e:
            raise MethodExecutionError(
                f"Method execution failed: {str(e)}",
                context={"method": self.method_name, "error_type": type(e).__name__}
            )

    def _check_required_params(self, **kwargs) -> List[str]:
        """Check if all required parameters are provided.

        Args:
            **kwargs: Provided parameters

        Returns:
            List of missing required parameter names
        """
        return [p for p in self.required_params if p not in kwargs or kwargs[p] is None]

    def get_validation_errors(self, data: pd.DataFrame, **kwargs) -> List[str]:
        """Get detailed validation error messages.

        Override this method in subclasses to provide specific validation
        error messages.

        Args:
            data: Input DataFrame to validate
            **kwargs: Method-specific parameters

        Returns:
            List of validation error messages
        """
        errors = []

        # Check for empty DataFrame
        if data.empty:
            errors.append("DataFrame is empty")

        # Check required parameters
        missing_params = self._check_required_params(**kwargs)
        if missing_params:
            errors.append(f"Missing required parameters: {missing_params}")

        return errors

    def get_method_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the method.

        Returns:
            Dictionary containing method metadata, parameters, and descriptions
        """
        return {
            'name': self.method_name,
            'description': self.method_description,
            'required_params': self.required_params,
            'optional_params': self.optional_params,
            'parameters': self.get_parameters(),
            'class': self.__class__.__name__
        }

    def is_compatible_with(self, other: 'ExperimentalMethod') -> bool:
        """Check if this method is compatible for chaining with another method.

        Override this method in subclasses to define specific compatibility rules.

        Args:
            other: Another experimental method to check compatibility with

        Returns:
            True if methods can be chained, False otherwise
        """
        # Default implementation: all methods are compatible
        return True

    def get_output_columns(self) -> List[str]:
        """Get the columns that this method adds or requires in output.

        Override this method in subclasses to specify output column requirements.

        Returns:
            List of column names that this method produces or modifies
        """
        return []


class MethodChain:
    """Chains multiple experimental methods together.

    This class allows executing multiple methods in sequence, with
    results aggregated into a combined report.

    Attributes:
        methods: List of (method, kwargs) tuples to execute
        results: List of MethodResult objects from executed methods
    """

    def __init__(self):
        """Initialize an empty method chain."""
        self.methods: List[Tuple[ExperimentalMethod, Dict[str, Any]]] = []
        self.results: List[MethodResult] = []

    def add(self, method: ExperimentalMethod, **kwargs) -> 'MethodChain':
        """Add a method to the chain.

        Args:
            method: Experimental method to add
            **kwargs: Parameters for the method

        Returns:
            Self for method chaining
        """
        self.methods.append((method, kwargs))
        return self

    def validate_chain(self) -> Tuple[bool, List[str]]:
        """Validate that all methods in the chain are compatible.

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        if len(self.methods) == 0:
            errors.append("Method chain is empty")
            return False, errors

        # Check pairwise compatibility
        for i in range(len(self.methods) - 1):
            method1, _ = self.methods[i]
            method2, _ = self.methods[i + 1]

            if not method1.is_compatible_with(method2):
                errors.append(
                    f"Method '{method1.method_name}' is not compatible with "
                    f"'{method2.method_name}'"
                )

        return len(errors) == 0, errors

    def execute(self, data: pd.DataFrame) -> List[MethodResult]:
        """Execute all methods in the chain.

        Args:
            data: Input DataFrame for analysis

        Returns:
            List of MethodResult objects from all methods

        Raises:
            MethodExecutionError: If chain validation fails or any method fails
        """
        # Validate the chain first
        is_valid, errors = self.validate_chain()
        if not is_valid:
            raise MethodExecutionError(
                f"Method chain validation failed: {errors}",
                context={"errors": errors}
            )

        self.results = []

        for method, kwargs in self.methods:
            result = method.validate_and_execute(data, **kwargs)
            self.results.append(result)

        return self.results

    def get_combined_results(self) -> Dict[str, Any]:
        """Get combined results from all executed methods.

        Returns:
            Dictionary containing aggregated results from all methods
        """
        if not self.results:
            return {}

        combined = {
            'methods_executed': [r.method_name for r in self.results],
            'all_statistics': {},
            'all_p_values': {},
            'all_effect_sizes': {},
            'summary': []
        }

        for result in self.results:
            # Prefix statistics with method name
            prefix = result.method_name.replace(" ", "_").lower()

            for key, value in result.statistics.items():
                combined['all_statistics'][f"{prefix}_{key}"] = value

            for key, value in result.p_values.items():
                combined['all_p_values'][f"{prefix}_{key}"] = value

            for key, value in result.effect_sizes.items():
                combined['all_effect_sizes'][f"{prefix}_{key}"] = value

            # Add interpretation to summary
            if 'interpretation' in result.metadata:
                combined['summary'].append({
                    'method': result.method_name,
                    'interpretation': result.metadata['interpretation']
                })

        return combined

    def clear(self) -> None:
        """Clear the method chain."""
        self.methods = []
        self.results = []
