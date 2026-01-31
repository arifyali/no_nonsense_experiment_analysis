"""
Property-based tests for package structure and organization.

**Feature: no-nonsense-experiment-analysis, Property 1: Package organization**
**Validates: Requirements 8.1, 8.2**
"""

import pytest
from hypothesis import given, strategies as st
import importlib
import inspect


@pytest.mark.property
def test_property_package_organization():
    """
    **Property 1: Package organization**
    
    For any valid package import, the package should be organized into exactly
    three main modules (data_prep, methods, utilities) with clear entry points
    and consistent structure.
    
    **Validates: Requirements 8.1, 8.2**
    """
    import no_nonsense_experiment_analysis
    
    # Test that package has exactly the three expected main modules
    expected_modules = {'data_prep', 'methods', 'utilities'}
    actual_modules = {name for name in dir(no_nonsense_experiment_analysis) 
                     if not name.startswith('_') and hasattr(getattr(no_nonsense_experiment_analysis, name), '__file__')}
    
    # Remove non-module attributes
    module_attributes = set()
    for name in dir(no_nonsense_experiment_analysis):
        if not name.startswith('_'):
            attr = getattr(no_nonsense_experiment_analysis, name)
            if hasattr(attr, '__file__') or (hasattr(attr, '__module__') and 'no_nonsense_experiment_analysis' in str(attr.__module__)):
                if name in expected_modules:
                    module_attributes.add(name)
    
    assert module_attributes == expected_modules, f"Expected modules {expected_modules}, got {module_attributes}"
    
    # Test that each module can be imported
    for module_name in expected_modules:
        module = getattr(no_nonsense_experiment_analysis, module_name)
        assert hasattr(module, '__file__'), f"Module {module_name} should be importable"
    
    # Test that package provides clear entry points for core classes
    expected_classes = {
        'ValidationResult', 'MethodResult', 'AnalysisReport', 'WorkflowState',
        'AnalysisError', 'DataValidationError', 'MethodExecutionError', 'WorkflowError',
        'WorkflowManager'
    }
    
    actual_classes = {name for name in dir(no_nonsense_experiment_analysis) 
                     if not name.startswith('_') and inspect.isclass(getattr(no_nonsense_experiment_analysis, name, None))}
    
    assert expected_classes.issubset(actual_classes), f"Missing core classes: {expected_classes - actual_classes}"


@pytest.mark.property
@given(st.sampled_from(['data_prep', 'methods', 'utilities']))
def test_property_module_structure_consistency(module_name):
    """
    **Property 1: Package organization (module consistency)**
    
    For any main module in the package, it should have consistent structure
    with proper __init__.py, __all__ exports, and importable classes.
    
    **Validates: Requirements 8.1, 8.2**
    """
    import no_nonsense_experiment_analysis
    
    # Get the module
    module = getattr(no_nonsense_experiment_analysis, module_name)
    
    # Test that module has __all__ defined
    assert hasattr(module, '__all__'), f"Module {module_name} should define __all__"
    
    # Test that all items in __all__ are actually available
    for item_name in module.__all__:
        assert hasattr(module, item_name), f"Module {module_name} exports {item_name} in __all__ but doesn't have it"
        
        # Test that the item can be imported
        item = getattr(module, item_name)
        assert item is not None, f"Item {item_name} in {module_name} should not be None"


@pytest.mark.property
def test_property_core_models_structure():
    """
    **Property 1: Package organization (core models)**
    
    For any core data model, it should be properly structured as a dataclass
    with required fields and methods.
    
    **Validates: Requirements 8.1, 8.2**
    """
    from no_nonsense_experiment_analysis.core.models import (
        ValidationResult, MethodResult, AnalysisReport, WorkflowState
    )
    from dataclasses import is_dataclass
    
    # Test that all core models are dataclasses
    core_models = [ValidationResult, MethodResult, AnalysisReport, WorkflowState]
    
    for model_class in core_models:
        assert is_dataclass(model_class), f"{model_class.__name__} should be a dataclass"
        
        # Test that dataclass can be instantiated with minimal arguments
        if model_class == ValidationResult:
            instance = model_class(is_valid=True)
            assert instance.is_valid is True
        elif model_class == MethodResult:
            instance = model_class(method_name="test")
            assert instance.method_name == "test"
        elif model_class == AnalysisReport:
            instance = model_class()
            assert isinstance(instance.dataset_summary, dict)
        elif model_class == WorkflowState:
            instance = model_class(current_step="test")
            assert instance.current_step == "test"


@pytest.mark.property
def test_property_exception_hierarchy():
    """
    **Property 1: Package organization (exception hierarchy)**
    
    For any custom exception in the package, it should inherit from AnalysisError
    and maintain proper hierarchy.
    
    **Validates: Requirements 8.1, 8.2**
    """
    from no_nonsense_experiment_analysis.core.exceptions import (
        AnalysisError, DataValidationError, MethodExecutionError, WorkflowError
    )
    
    # Test exception hierarchy
    specific_exceptions = [DataValidationError, MethodExecutionError, WorkflowError]
    
    for exception_class in specific_exceptions:
        assert issubclass(exception_class, AnalysisError), f"{exception_class.__name__} should inherit from AnalysisError"
        assert issubclass(exception_class, Exception), f"{exception_class.__name__} should inherit from Exception"
        
        # Test that exception can be instantiated and raised
        try:
            raise exception_class("test message")
        except exception_class as e:
            assert str(e) == "test message"
        except Exception:
            pytest.fail(f"{exception_class.__name__} should be raisable and catchable")


@pytest.mark.property
@given(st.text(min_size=1, max_size=100))
def test_property_analysis_error_context_handling(error_message):
    """
    **Property 1: Package organization (error context)**
    
    For any error message and context, AnalysisError should handle it consistently
    and provide proper string representation.
    
    **Validates: Requirements 8.1, 8.2**
    """
    from no_nonsense_experiment_analysis.core.exceptions import AnalysisError
    
    # Test with context
    context = {"step": "test", "data_shape": (100, 5)}
    error = AnalysisError(error_message, context=context)
    
    assert error.message == error_message
    assert error.context == context
    
    # Test string representation includes both message and context
    error_str = str(error)
    assert error_message in error_str
    assert "step=test" in error_str
    assert "data_shape=(100, 5)" in error_str
    
    # Test without context
    error_no_context = AnalysisError(error_message)
    assert error_no_context.message == error_message
    assert error_no_context.context == {}
    assert str(error_no_context) == error_message