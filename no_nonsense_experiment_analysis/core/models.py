"""
Core data models for the experiment analysis package.

This module defines the primary data structures used throughout the package
for validation results, method results, analysis reports, and workflow state.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional
import json
import pandas as pd


@dataclass
class ValidationResult:
    """Result of data validation operations.
    
    Attributes:
        is_valid: Whether the data passed validation
        errors: List of validation error messages
        warnings: List of validation warning messages  
        data_summary: Summary statistics and metadata about the data
    """
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    data_summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MethodResult:
    """Standardized result container for experimental methods.
    
    Attributes:
        method_name: Name of the method that produced this result
        parameters: Parameters used in the method execution
        statistics: Statistical measures computed by the method
        p_values: P-values from statistical tests
        confidence_intervals: Confidence intervals for estimates
        effect_sizes: Effect size measures
        metadata: Additional metadata about the method execution
    """
    method_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, float] = field(default_factory=dict)
    p_values: Dict[str, float] = field(default_factory=dict)
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    effect_sizes: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowState:
    """Tracks the current state of analysis workflow.
    
    Attributes:
        current_step: Name of the current workflow step
        completed_steps: List of completed workflow steps
        data_shape: Shape of the current dataset (rows, columns)
        applied_transformations: List of transformations applied to data
        validation_status: Current validation status of the data
    """
    current_step: str
    completed_steps: List[str] = field(default_factory=list)
    data_shape: Tuple[int, int] = (0, 0)
    applied_transformations: List[str] = field(default_factory=list)
    validation_status: Optional[ValidationResult] = None


@dataclass
class AnalysisReport:
    """LLM-ready structured report of analysis results.
    
    This class provides a comprehensive summary of the analysis process
    and results in a format suitable for LLM-based narrative generation.
    
    Attributes:
        dataset_summary: Summary statistics and characteristics of the dataset
        preprocessing_steps: List of preprocessing operations performed
        methods_applied: List of analysis methods used
        results: List of method results from the analysis
        overall_conclusions: High-level conclusions and insights
        metadata: Additional metadata about the analysis process
    """
    dataset_summary: Dict[str, Any] = field(default_factory=dict)
    preprocessing_steps: List[str] = field(default_factory=list)
    methods_applied: List[str] = field(default_factory=list)
    results: List[MethodResult] = field(default_factory=list)
    overall_conclusions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self) -> str:
        """Serialize the report to JSON format.
        
        Returns:
            JSON string representation of the report
        """
        # Convert dataclass to dictionary, handling nested dataclasses
        def convert_to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: convert_to_dict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [convert_to_dict(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_to_dict(v) for k, v in obj.items()}
            else:
                return obj
        
        report_dict = convert_to_dict(self)
        return json.dumps(report_dict, indent=2, default=str)
    
    def to_llm_prompt(self) -> str:
        """Generate an LLM-ready prompt from the analysis report.
        
        Returns:
            Formatted string suitable for LLM narrative generation
        """
        prompt_parts = [
            "# Experimental Analysis Report",
            "",
            "## Dataset Summary",
            f"- Shape: {self.dataset_summary.get('shape', 'Unknown')}",
            f"- Columns: {self.dataset_summary.get('columns', [])}",
            f"- Data types: {self.dataset_summary.get('dtypes', {})}",
            "",
            "## Preprocessing Steps",
        ]
        
        for step in self.preprocessing_steps:
            prompt_parts.append(f"- {step}")
        
        prompt_parts.extend([
            "",
            "## Analysis Methods Applied",
        ])
        
        for method in self.methods_applied:
            prompt_parts.append(f"- {method}")
        
        prompt_parts.extend([
            "",
            "## Results Summary",
        ])
        
        for result in self.results:
            prompt_parts.extend([
                f"### {result.method_name}",
                f"- Parameters: {result.parameters}",
                f"- Key statistics: {result.statistics}",
                f"- P-values: {result.p_values}",
                f"- Effect sizes: {result.effect_sizes}",
                ""
            ])
        
        prompt_parts.extend([
            "## Overall Conclusions",
            str(self.overall_conclusions),
            "",
            "Please generate a narrative summary of this experimental analysis."
        ])
        
        return "\n".join(prompt_parts)