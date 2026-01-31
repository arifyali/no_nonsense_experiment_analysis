"""
Experimental methods module for statistical analysis.

This module provides the framework for experimental methods, method registration,
and standardized result handling.
"""

from .base import ExperimentalMethod
from .registry import MethodRegistry
from .result import MethodResult

__all__ = [
    "ExperimentalMethod",
    "MethodRegistry", 
    "MethodResult"
]