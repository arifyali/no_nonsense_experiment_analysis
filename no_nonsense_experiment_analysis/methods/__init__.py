"""
Experimental methods module for statistical analysis.

This module provides the framework for experimental methods, method registration,
and standardized result handling, along with implementations of common
experimental design methods.
"""

from .base import ExperimentalMethod
from .registry import MethodRegistry, default_registry
from .result import MethodResult
from .ab_test import ABTest
from .anova import OneWayANOVA
from .chi_square import ChiSquareTest
from .regression import LinearRegressionAnalysis, LogisticRegressionAnalysis

__all__ = [
    "ExperimentalMethod",
    "MethodRegistry", 
    "default_registry",
    "MethodResult",
    "ABTest",
    "OneWayANOVA", 
    "ChiSquareTest",
    "LinearRegressionAnalysis",
    "LogisticRegressionAnalysis"
]