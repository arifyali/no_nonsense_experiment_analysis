"""
Method registry for experimental methods.

This module will be implemented in task 5.1.
"""

from typing import Dict, Type, List
from .base import ExperimentalMethod


class MethodRegistry:
    """Registry for managing experimental methods."""
    
    def __init__(self):
        """Initialize the method registry."""
        self._methods: Dict[str, Type[ExperimentalMethod]] = {}
    
    def register_method(self, name: str, method_class: Type[ExperimentalMethod]):
        """Register a new experimental method."""
        # TODO: Implement in task 5.1
        pass
    
    def get_method(self, name: str) -> ExperimentalMethod:
        """Get an instance of a registered method."""
        # TODO: Implement in task 5.1
        pass
    
    def list_available_methods(self) -> List[str]:
        """List all available method names."""
        # TODO: Implement in task 5.1
        pass