"""
Tests for package structure and basic imports.

This module tests that the package is properly structured and that
all core components can be imported successfully.
"""

import pytest
import pandas as pd
from hypothesis import given, strategies as st

# Test basic imports
def test_package_imports():
    """Test that all main package components can be imported."""
    import no_nonsense_experiment_analysis
    
    # Test main package attributes
    assert hasattr(no_nonsense_experiment_analysis, '__version__')
    assert hasattr(no_nonsense_experiment_analysis, 'data_prep')
    assert hasattr(no_nonsense_experiment_analysis, 'methods')
    assert hasattr(no_nonsense_experiment_analysis, 'utilities')


def test_core_model_imports():
    """Test that core data models can be imported."""
    from no_nonsense_experiment_analysis.core.models import (
        ValidationResult, MethodResult, AnalysisReport, WorkflowState
    )
    
    # Test that classes can be instantiated
    validation_result = ValidationResult(is_valid=True)
    assert validation_result.is_valid is True
    
    method_result = MethodResult(method_name="test_method")
    assert method_result.method_name == "test_method"
    
    analysis_report = AnalysisReport()
    assert isinstance(analysis_report.dataset_summary, dict)
    
    workflow_state = WorkflowState(current_step="test")
    assert workflow_state.current_step == "test"


def test_exception_imports():
    """Test that custom exceptions can be imported."""
    from no_nonsense_experiment_analysis.core.exceptions import (
        AnalysisError, DataValidationError, MethodExecutionError, WorkflowError
    )
    
    # Test exception hierarchy
    assert issubclass(DataValidationError, AnalysisError)
    assert issubclass(MethodExecutionError, AnalysisError)
    assert issubclass(WorkflowError, AnalysisError)


def test_workflow_manager_import():
    """Test that WorkflowManager can be imported and instantiated."""
    from no_nonsense_experiment_analysis.core.workflow import WorkflowManager
    import pandas as pd
    
    # Test with valid DataFrame
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    workflow = WorkflowManager(df)
    
    assert workflow.original_data.equals(df)
    assert workflow.current_data.equals(df)
    assert workflow.state.current_step == "initialized"


def test_module_structure():
    """Test that all expected modules and classes exist."""
    # Test data_prep module
    from no_nonsense_experiment_analysis.data_prep import (
        DataValidator, DataCleaner, Preprocessor
    )
    
    # Test methods module  
    from no_nonsense_experiment_analysis.methods import (
        ExperimentalMethod, MethodRegistry, MethodResult
    )
    
    # Test utilities module
    from no_nonsense_experiment_analysis.utilities import (
        StatisticalFunctions, VisualizationTools, DataTransformers
    )
    
    # Verify classes can be instantiated (where applicable)
    validator = DataValidator()
    cleaner = DataCleaner()
    preprocessor = Preprocessor()
    registry = MethodRegistry()
    stats = StatisticalFunctions()
    viz = VisualizationTools()
    transformers = DataTransformers()


@pytest.mark.property
@given(st.text())
def test_analysis_error_with_context(error_message):
    """Property test: AnalysisError should handle any string message."""
    from no_nonsense_experiment_analysis.core.exceptions import AnalysisError
    
    error = AnalysisError(error_message, context={"test": "value"})
    assert error.message == error_message
    assert error.context == {"test": "value"}
    
    # Test string representation includes context
    error_str = str(error)
    assert error_message in error_str
    assert "test=value" in error_str


def test_analysis_report_serialization():
    """Test that AnalysisReport can be serialized to JSON."""
    from no_nonsense_experiment_analysis.core.models import AnalysisReport, MethodResult
    
    # Create a report with some data
    result = MethodResult(
        method_name="test_method",
        statistics={"mean": 5.0, "std": 1.5}
    )
    
    report = AnalysisReport(
        dataset_summary={"shape": (100, 5)},
        methods_applied=["test_method"],
        results=[result]
    )
    
    # Test JSON serialization
    json_str = report.to_json()
    assert isinstance(json_str, str)
    assert "test_method" in json_str
    assert "shape" in json_str
    
    # Test LLM prompt generation
    llm_prompt = report.to_llm_prompt()
    assert isinstance(llm_prompt, str)
    assert "Experimental Analysis Report" in llm_prompt
    assert "test_method" in llm_prompt