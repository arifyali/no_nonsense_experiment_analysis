"""
Method registry for experimental methods.

This module provides a centralized registry for all experimental analysis methods.
"""

from typing import Dict, Type, List, Any
from .base import ExperimentalMethod
from .ab_test import ABTest
from .anova import OneWayANOVA
from .chi_square import ChiSquareTest
from .regression import LinearRegressionAnalysis, LogisticRegressionAnalysis


class MethodRegistry:
    """Registry for managing experimental methods."""
    
    def __init__(self):
        """Initialize the method registry with default methods."""
        self._methods: Dict[str, Type[ExperimentalMethod]] = {}
        self._register_default_methods()
    
    def _register_default_methods(self):
        """Register all default experimental methods."""
        self.register_method("ab_test", ABTest)
        self.register_method("one_way_anova", OneWayANOVA)
        self.register_method("chi_square_independence", 
                           lambda **kwargs: ChiSquareTest(test_type='independence', **kwargs))
        self.register_method("chi_square_goodness_of_fit", 
                           lambda **kwargs: ChiSquareTest(test_type='goodness_of_fit', **kwargs))
        self.register_method("linear_regression", LinearRegressionAnalysis)
        self.register_method("logistic_regression", LogisticRegressionAnalysis)
    
    def register_method(self, name: str, method_class: Type[ExperimentalMethod]):
        """Register a new experimental method.
        
        Args:
            name: Unique name for the method
            method_class: Class or factory function that creates method instances
        """
        self._methods[name] = method_class
    
    def get_method(self, name: str, **kwargs) -> ExperimentalMethod:
        """Get an instance of a registered method.
        
        Args:
            name: Name of the method to retrieve
            **kwargs: Parameters to pass to method constructor
            
        Returns:
            Instance of the requested method
            
        Raises:
            KeyError: If method name is not registered
        """
        if name not in self._methods:
            raise KeyError(f"Method '{name}' not found. Available methods: {self.list_available_methods()}")
        
        method_class = self._methods[name]
        return method_class(**kwargs)
    
    def list_available_methods(self) -> List[str]:
        """List all available method names.
        
        Returns:
            List of registered method names
        """
        return list(self._methods.keys())
    
    def get_method_info(self, name: str) -> Dict[str, Any]:
        """Get information about a specific method.
        
        Args:
            name: Name of the method
            
        Returns:
            Dictionary containing method information
            
        Raises:
            KeyError: If method name is not registered
        """
        if name not in self._methods:
            raise KeyError(f"Method '{name}' not found")
        
        # Create a temporary instance to get parameter info
        try:
            temp_instance = self.get_method(name)
            parameters = temp_instance.get_parameters()
            method_class = self._methods[name]
            
            return {
                'name': name,
                'class': method_class.__name__ if hasattr(method_class, '__name__') else str(method_class),
                'parameters': parameters,
                'description': method_class.__doc__ if hasattr(method_class, '__doc__') else "No description available"
            }
        except Exception as e:
            return {
                'name': name,
                'class': str(self._methods[name]),
                'parameters': {},
                'description': f"Error getting info: {str(e)}"
            }
    
    def get_all_methods_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all registered methods.
        
        Returns:
            Dictionary mapping method names to their information
        """
        return {name: self.get_method_info(name) for name in self.list_available_methods()}


# Global registry instance
default_registry = MethodRegistry()