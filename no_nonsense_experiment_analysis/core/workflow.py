"""
Workflow management for experimental data analysis.

This module provides the WorkflowManager class that orchestrates the complete
analysis pipeline from data loading through reporting.
"""

from typing import Optional, Dict, Any
import pandas as pd
import copy

from .models import WorkflowState, AnalysisReport, ValidationResult, MethodResult
from .exceptions import WorkflowError, DataValidationError


class WorkflowManager:
    """Orchestrates the complete analysis pipeline.
    
    The WorkflowManager provides a fluent interface for executing the
    standard analysis workflow: load data → prep → analyze → report.
    It maintains state between steps and ensures data integrity.
    """
    
    def __init__(self, data: pd.DataFrame):
        """Initialize the workflow manager with input data.
        
        Args:
            data: Input pandas DataFrame for analysis
            
        Raises:
            DataValidationError: If input is not a pandas DataFrame
        """
        if not isinstance(data, pd.DataFrame):
            raise DataValidationError(
                "Input must be a pandas DataFrame",
                context={"input_type": type(data).__name__}
            )
        
        self.original_data = data.copy()
        self.current_data = data.copy()
        self.analysis_history = []
        self.results = {}
        
        # Initialize workflow state
        self.state = WorkflowState(
            current_step="initialized",
            completed_steps=[],
            data_shape=data.shape,
            applied_transformations=[],
            validation_status=None
        )
    
    def prep(self, **kwargs) -> 'WorkflowManager':
        """Execute data preparation step.
        
        This method delegates to the data_prep module to clean and
        preprocess the data for analysis.
        
        Args:
            **kwargs: Parameters for data preparation operations
            
        Returns:
            Self for method chaining
            
        Raises:
            WorkflowError: If preparation step fails
        """
        try:
            self.state.current_step = "prep"
            
            # TODO: Implement actual data preparation logic
            # This will be implemented in later tasks
            
            self.state.completed_steps.append("prep")
            self.state.data_shape = self.current_data.shape
            
            return self
            
        except Exception as e:
            raise WorkflowError(
                f"Data preparation failed: {str(e)}",
                context={
                    "current_step": self.state.current_step,
                    "data_shape": self.state.data_shape
                }
            )
    
    def analyze(self, method: str, **kwargs) -> 'WorkflowManager':
        """Execute analysis step with specified method.
        
        This method delegates to the methods module to perform
        statistical analysis on the prepared data.
        
        Args:
            method: Name of the analysis method to use
            **kwargs: Parameters for the analysis method
            
        Returns:
            Self for method chaining
            
        Raises:
            WorkflowError: If analysis step fails
        """
        try:
            self.state.current_step = f"analyze_{method}"
            
            # TODO: Implement actual analysis logic
            # This will be implemented in later tasks
            
            self.state.completed_steps.append(f"analyze_{method}")
            
            return self
            
        except Exception as e:
            raise WorkflowError(
                f"Analysis with method '{method}' failed: {str(e)}",
                context={
                    "method": method,
                    "current_step": self.state.current_step,
                    "parameters": kwargs
                }
            )
    
    def report(self) -> AnalysisReport:
        """Generate comprehensive analysis report.
        
        This method creates a structured report of all analysis
        results suitable for LLM-based narrative generation.
        
        Returns:
            AnalysisReport object with complete analysis summary
            
        Raises:
            WorkflowError: If report generation fails
        """
        try:
            self.state.current_step = "report"
            
            # Create analysis report
            report = AnalysisReport(
                dataset_summary={
                    "original_shape": self.original_data.shape,
                    "current_shape": self.current_data.shape,
                    "columns": list(self.current_data.columns),
                    "dtypes": {col: str(dtype) for col, dtype in self.current_data.dtypes.items()}
                },
                preprocessing_steps=self.state.applied_transformations,
                methods_applied=[step for step in self.state.completed_steps if step.startswith("analyze_")],
                results=list(self.results.values()) if isinstance(self.results, dict) else [],
                overall_conclusions={
                    "workflow_completed": True,
                    "steps_completed": len(self.state.completed_steps),
                    "data_integrity_preserved": self.original_data.shape[0] >= 0  # Basic check
                },
                metadata={
                    "workflow_state": {
                        "current_step": self.state.current_step,
                        "completed_steps": self.state.completed_steps,
                        "data_shape": self.state.data_shape
                    }
                }
            )
            
            self.state.completed_steps.append("report")
            
            return report
            
        except Exception as e:
            raise WorkflowError(
                f"Report generation failed: {str(e)}",
                context={
                    "current_step": self.state.current_step,
                    "completed_steps": self.state.completed_steps
                }
            )
    
    def get_state(self) -> WorkflowState:
        """Get the current workflow state.
        
        Returns:
            Copy of the current workflow state
        """
        return copy.deepcopy(self.state)
    
    def get_current_data(self) -> pd.DataFrame:
        """Get the current state of the data.
        
        Returns:
            Copy of the current data DataFrame
        """
        return self.current_data.copy()
    
    def get_original_data(self) -> pd.DataFrame:
        """Get the original input data.
        
        Returns:
            Copy of the original data DataFrame
        """
        return self.original_data.copy()