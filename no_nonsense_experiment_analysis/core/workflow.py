"""
Workflow management for experimental data analysis.

This module provides the WorkflowManager class that orchestrates the complete
analysis pipeline from data loading through reporting.
"""

from typing import Optional, Dict, Any, List, Union, TYPE_CHECKING
import pandas as pd
import copy

from .models import WorkflowState, AnalysisReport, ValidationResult, MethodResult
from .exceptions import WorkflowError, DataValidationError, MethodExecutionError

if TYPE_CHECKING:
    from ..methods.base import ExperimentalMethod


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
        self.analysis_history: List[str] = []
        self.results: Dict[str, MethodResult] = {}
        self.warnings: List[str] = []

        # Initialize workflow state
        self.state = WorkflowState(
            current_step="initialized",
            completed_steps=[],
            data_shape=data.shape,
            applied_transformations=[],
            validation_status=None
        )
    
    def prep(
        self,
        validate: bool = True,
        clean_missing: Optional[str] = None,
        remove_duplicates: bool = False,
        normalize_columns: Optional[List[str]] = None,
        normalize_method: str = "minmax",
        encode_categorical: Optional[List[str]] = None,
        encode_method: str = "onehot",
        **kwargs
    ) -> 'WorkflowManager':
        """Execute data preparation step.

        This method delegates to the data_prep module to clean and
        preprocess the data for analysis.

        Args:
            validate: Whether to validate the data first (default: True)
            clean_missing: Strategy for handling missing values
                ('drop', 'fill_mean', 'fill_median', 'fill_mode', etc.)
            remove_duplicates: Whether to remove duplicate rows
            normalize_columns: List of columns to normalize
            normalize_method: Normalization method ('minmax', 'zscore', 'robust')
            encode_categorical: List of categorical columns to encode
            encode_method: Encoding method ('onehot', 'label', 'ordinal')
            **kwargs: Additional parameters for specific operations

        Returns:
            Self for method chaining

        Raises:
            WorkflowError: If preparation step fails
        """
        try:
            self.state.current_step = "prep"

            # Lazy imports to avoid circular dependency
            from ..data_prep import DataValidator, DataCleaner, Preprocessor

            # Step 1: Validate data
            if validate:
                validator = DataValidator(strict_mode=False)
                validation_result = validator.validate_dataframe(self.current_data)
                self.state.validation_status = validation_result

                # Collect warnings without stopping
                if validation_result.warnings:
                    self.warnings.extend(validation_result.warnings)

                # If validation has errors, raise but with guidance
                if not validation_result.is_valid:
                    raise WorkflowError(
                        f"Data validation failed: {'; '.join(validation_result.errors)}",
                        context={
                            "current_step": "prep.validate",
                            "errors": validation_result.errors,
                            "data_shape": self.state.data_shape
                        }
                    )
                self.state.applied_transformations.append("validated")

            # Step 2: Clean missing values
            if clean_missing:
                cleaner = DataCleaner(strict_mode=True)
                self.current_data = cleaner.handle_missing_values(
                    self.current_data,
                    strategy=clean_missing,
                    columns=kwargs.get('missing_columns'),
                    fill_value=kwargs.get('fill_value')
                )
                self.state.applied_transformations.append(f"clean_missing:{clean_missing}")

            # Step 3: Remove duplicates
            if remove_duplicates:
                cleaner = DataCleaner(strict_mode=True)
                self.current_data = cleaner.remove_duplicates(
                    self.current_data,
                    subset=kwargs.get('duplicate_subset'),
                    keep=kwargs.get('duplicate_keep', 'first')
                )
                self.state.applied_transformations.append("remove_duplicates")

            # Step 4: Normalize columns
            if normalize_columns:
                preprocessor = Preprocessor(strict_mode=True)
                self.current_data = preprocessor.normalize_columns(
                    self.current_data,
                    columns=normalize_columns,
                    method=normalize_method
                )
                self.state.applied_transformations.append(
                    f"normalize:{normalize_method}:{','.join(normalize_columns)}"
                )

            # Step 5: Encode categorical columns
            if encode_categorical:
                preprocessor = Preprocessor(strict_mode=True)
                self.current_data = preprocessor.encode_categorical(
                    self.current_data,
                    columns=encode_categorical,
                    method=encode_method
                )
                self.state.applied_transformations.append(
                    f"encode:{encode_method}:{','.join(encode_categorical)}"
                )

            self.state.completed_steps.append("prep")
            self.state.data_shape = self.current_data.shape

            return self

        except WorkflowError:
            raise
        except DataValidationError as e:
            raise WorkflowError(
                f"Data preparation failed during validation: {str(e)}",
                context={
                    "current_step": self.state.current_step,
                    "data_shape": self.state.data_shape,
                    "original_error": str(e)
                }
            )
        except Exception as e:
            raise WorkflowError(
                f"Data preparation failed: {str(e)}",
                context={
                    "current_step": self.state.current_step,
                    "data_shape": self.state.data_shape,
                    "completed_steps": self.state.completed_steps
                }
            )
    
    def analyze(
        self,
        method: Union[str, "ExperimentalMethod"],
        **kwargs
    ) -> 'WorkflowManager':
        """Execute analysis step with specified method.

        This method delegates to the methods module to perform
        statistical analysis on the prepared data.

        Args:
            method: Name of the analysis method (string) or ExperimentalMethod instance
                Available methods: 'ab_test', 'one_way_anova', 'chi_square_independence',
                'chi_square_goodness_of_fit', 'linear_regression', 'logistic_regression'
            **kwargs: Parameters for the analysis method (e.g., group_col, metric_col)

        Returns:
            Self for method chaining

        Raises:
            WorkflowError: If analysis step fails
        """
        # Lazy imports to avoid circular dependency
        from ..methods import default_registry
        from ..methods.base import ExperimentalMethod

        method_name = method if isinstance(method, str) else method.method_name

        try:
            self.state.current_step = f"analyze_{method_name}"

            # Get method instance
            if isinstance(method, str):
                try:
                    method_instance = default_registry.get_method(method)
                except KeyError as e:
                    available = default_registry.list_available_methods()
                    raise WorkflowError(
                        f"Unknown method '{method}'. Available methods: {available}",
                        context={
                            "method": method,
                            "available_methods": available,
                            "current_step": self.state.current_step
                        }
                    )
            else:
                method_instance = method

            # Execute the method
            result = method_instance.validate_and_execute(self.current_data, **kwargs)

            # Store result
            result_key = f"{method_name}_{len(self.results)}"
            self.results[result_key] = result
            self.analysis_history.append(method_name)

            self.state.completed_steps.append(f"analyze_{method_name}")

            return self

        except WorkflowError:
            raise
        except MethodExecutionError as e:
            raise WorkflowError(
                f"Analysis with method '{method_name}' failed: {str(e)}",
                context={
                    "method": method_name,
                    "current_step": self.state.current_step,
                    "parameters": kwargs,
                    "original_error": str(e)
                }
            )
        except Exception as e:
            raise WorkflowError(
                f"Analysis with method '{method_name}' failed: {str(e)}",
                context={
                    "method": method_name,
                    "current_step": self.state.current_step,
                    "parameters": kwargs,
                    "completed_steps": self.state.completed_steps
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

            # Summarize significant findings
            significant_results = []
            for _, result in self.results.items():
                # Check if any p-value is significant (< 0.05)
                for p_name, p_val in result.p_values.items():
                    if p_val < 0.05:
                        significant_results.append({
                            "method": result.method_name,
                            "test": p_name,
                            "p_value": p_val,
                            "effect_sizes": result.effect_sizes
                        })

            # Create analysis report
            report = AnalysisReport(
                dataset_summary={
                    "original_shape": self.original_data.shape,
                    "current_shape": self.current_data.shape,
                    "columns": list(self.current_data.columns),
                    "dtypes": {col: str(dtype) for col, dtype in self.current_data.dtypes.items()},
                    "rows_removed": self.original_data.shape[0] - self.current_data.shape[0]
                },
                preprocessing_steps=self.state.applied_transformations,
                methods_applied=self.analysis_history,
                results=list(self.results.values()),
                overall_conclusions={
                    "workflow_completed": True,
                    "steps_completed": len(self.state.completed_steps),
                    "analyses_performed": len(self.results),
                    "significant_findings": len(significant_results),
                    "significant_results_summary": significant_results,
                    "data_integrity_preserved": True
                },
                metadata={
                    "workflow_state": {
                        "current_step": self.state.current_step,
                        "completed_steps": self.state.completed_steps,
                        "data_shape": self.state.data_shape
                    },
                    "warnings": self.warnings,
                    "validation_status": (
                        {
                            "is_valid": self.state.validation_status.is_valid,
                            "errors": self.state.validation_status.errors,
                            "warnings": self.state.validation_status.warnings
                        }
                        if self.state.validation_status else None
                    )
                }
            )

            self.state.completed_steps.append("report")

            return report

        except Exception as e:
            raise WorkflowError(
                f"Report generation failed: {str(e)}",
                context={
                    "current_step": self.state.current_step,
                    "completed_steps": self.state.completed_steps,
                    "results_count": len(self.results)
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