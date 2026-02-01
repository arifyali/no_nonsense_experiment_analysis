"""
Data validation functionality for experimental data analysis.

This module provides comprehensive validation for pandas DataFrames used in
experimental analysis, ensuring data quality and structure requirements are met.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from ..core.models import ValidationResult
from ..core.exceptions import DataValidationError


class DataValidator:
    """Validates pandas DataFrames for experimental analysis.
    
    This class provides comprehensive validation functionality to ensure
    data quality and structure requirements are met before analysis.
    """
    
    def __init__(self, strict_mode: bool = False):
        """Initialize the data validator.
        
        Args:
            strict_mode: If True, raises exceptions on validation failures.
                        If False, returns ValidationResult with errors/warnings.
        """
        self.strict_mode = strict_mode
    
    def validate_dataframe(self, df: Any, schema: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate DataFrame structure and content.
        
        Performs comprehensive validation including type checking, structure
        validation, and optional schema validation.
        
        Args:
            df: Input data to validate (should be pandas DataFrame)
            schema: Optional schema dictionary with validation rules
                   Format: {
                       'required_columns': ['col1', 'col2'],
                       'column_types': {'col1': 'numeric', 'col2': 'categorical'},
                       'min_rows': 10,
                       'max_rows': 10000,
                       'allow_missing': True
                   }
        
        Returns:
            ValidationResult with validation status and details
            
        Raises:
            DataValidationError: If strict_mode=True and validation fails
        """
        errors = []
        warnings = []
        data_summary = {}
        
        # Check if input is a pandas DataFrame
        if not isinstance(df, pd.DataFrame):
            error_msg = f"Input must be a pandas DataFrame, got {type(df).__name__}"
            errors.append(error_msg)
            if self.strict_mode:
                raise DataValidationError(error_msg)
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                data_summary=data_summary
            )
        
        # Basic structure validation
        data_summary['shape'] = df.shape
        data_summary['columns'] = list(df.columns)
        data_summary['dtypes'] = df.dtypes.to_dict()
        data_summary['memory_usage'] = df.memory_usage(deep=True).sum()
        
        # Check for empty DataFrame
        if df.empty:
            error_msg = "DataFrame is empty (no rows or columns)"
            errors.append(error_msg)
        elif len(df) == 0:
            error_msg = "DataFrame has no rows"
            errors.append(error_msg)
        elif len(df.columns) == 0:
            error_msg = "DataFrame has no columns"
            errors.append(error_msg)
        
        # Check for duplicate column names
        if len(df.columns) != len(set(df.columns)):
            duplicates = df.columns[df.columns.duplicated()].tolist()
            error_msg = f"Duplicate column names found: {duplicates}"
            errors.append(error_msg)
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]
        if len(missing_cols) > 0:
            data_summary['missing_values'] = missing_cols.to_dict()
            missing_pct = (missing_cols / len(df) * 100).round(2)
            
            # Warn about high missing value percentages
            high_missing = missing_pct[missing_pct > 50]
            if len(high_missing) > 0:
                warning_msg = f"Columns with >50% missing values: {high_missing.to_dict()}"
                warnings.append(warning_msg)
        else:
            data_summary['missing_values'] = {}
        
        # Check for completely null columns
        null_cols = df.columns[df.isnull().all()].tolist()
        if null_cols:
            warning_msg = f"Columns with all null values: {null_cols}"
            warnings.append(warning_msg)
        
        # Check for single-value columns (no variance)
        single_value_cols = []
        for col in df.columns:
            try:
                unique_count = df[col].nunique(dropna=False)
                if unique_count <= 1:
                    single_value_cols.append(col)
            except (ValueError, TypeError):
                # Skip columns that can't be processed (e.g., due to duplicate column names)
                continue
        if single_value_cols:
            warning_msg = f"Columns with single unique value: {single_value_cols}"
            warnings.append(warning_msg)
        
        # Schema validation if provided
        if schema:
            schema_errors, schema_warnings = self._validate_schema(df, schema)
            errors.extend(schema_errors)
            warnings.extend(schema_warnings)
        
        # Determine overall validation status
        is_valid = len(errors) == 0
        
        if self.strict_mode and not is_valid:
            raise DataValidationError(f"DataFrame validation failed: {'; '.join(errors)}")
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            data_summary=data_summary
        )
    
    def check_required_columns(self, df: pd.DataFrame, columns: List[str]) -> ValidationResult:
        """Check if required columns are present in the DataFrame.
        
        Args:
            df: DataFrame to check
            columns: List of required column names
            
        Returns:
            ValidationResult indicating if all required columns are present
            
        Raises:
            DataValidationError: If strict_mode=True and required columns are missing
        """
        errors = []
        warnings = []
        
        if not isinstance(df, pd.DataFrame):
            error_msg = f"Input must be a pandas DataFrame, got {type(df).__name__}"
            errors.append(error_msg)
        else:
            missing_columns = [col for col in columns if col not in df.columns]
            
            if missing_columns:
                error_msg = f"Required columns missing: {missing_columns}"
                errors.append(error_msg)
            
            # Check for case-sensitive issues
            df_cols_lower = [col.lower() for col in df.columns]
            for col in missing_columns:
                if col.lower() in df_cols_lower:
                    actual_col = df.columns[df_cols_lower.index(col.lower())]
                    warning_msg = f"Column '{col}' not found, but '{actual_col}' exists (case mismatch)"
                    warnings.append(warning_msg)
        
        is_valid = len(errors) == 0
        
        if self.strict_mode and not is_valid:
            raise DataValidationError(f"Required column check failed: {'; '.join(errors)}")
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            data_summary={
                'required_columns': columns,
                'present_columns': list(df.columns) if isinstance(df, pd.DataFrame) else [],
                'missing_columns': errors
            }
        )
    
    def validate_data_types(self, df: pd.DataFrame, type_map: Dict[str, str]) -> ValidationResult:
        """Validate data types against expected types.
        
        Args:
            df: DataFrame to validate
            type_map: Dictionary mapping column names to expected types
                     Supported types: 'numeric', 'categorical', 'datetime', 'boolean', 'string'
            
        Returns:
            ValidationResult with type validation details
            
        Raises:
            DataValidationError: If strict_mode=True and type validation fails
        """
        errors = []
        warnings = []
        data_summary = {}
        
        if not isinstance(df, pd.DataFrame):
            error_msg = f"Input must be a pandas DataFrame, got {type(df).__name__}"
            errors.append(error_msg)
            if self.strict_mode:
                raise DataValidationError(error_msg)
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings, data_summary=data_summary)
        
        type_validation_results = {}
        
        for column, expected_type in type_map.items():
            if column not in df.columns:
                error_msg = f"Column '{column}' not found in DataFrame"
                errors.append(error_msg)
                continue
            
            col_data = df[column]
            actual_dtype = str(col_data.dtype)
            is_valid_type = False
            
            # Validate based on expected type
            if expected_type == 'numeric':
                is_valid_type = pd.api.types.is_numeric_dtype(col_data)
                if not is_valid_type:
                    # Check if it can be converted to numeric
                    try:
                        converted = pd.to_numeric(col_data, errors='coerce')
                        if converted.notna().any():  # If any values can be converted
                            warning_msg = f"Column '{column}' is not numeric but may be convertible"
                            warnings.append(warning_msg)
                        else:
                            error_msg = f"Column '{column}' expected numeric, got {actual_dtype}"
                            errors.append(error_msg)
                    except:
                        error_msg = f"Column '{column}' expected numeric, got {actual_dtype}"
                        errors.append(error_msg)
                        
            elif expected_type == 'categorical':
                is_valid_type = (pd.api.types.is_categorical_dtype(col_data) or 
                               pd.api.types.is_object_dtype(col_data) or
                               pd.api.types.is_string_dtype(col_data))
                if not is_valid_type:
                    error_msg = f"Column '{column}' expected categorical, got {actual_dtype}"
                    errors.append(error_msg)
                    
            elif expected_type == 'datetime':
                is_valid_type = pd.api.types.is_datetime64_any_dtype(col_data)
                if not is_valid_type:
                    # Check if it can be converted to datetime
                    try:
                        converted = pd.to_datetime(col_data, errors='coerce')
                        if converted.notna().any():  # If any values can be converted
                            warning_msg = f"Column '{column}' is not datetime but may be convertible"
                            warnings.append(warning_msg)
                        else:
                            error_msg = f"Column '{column}' expected datetime, got {actual_dtype}"
                            errors.append(error_msg)
                    except:
                        error_msg = f"Column '{column}' expected datetime, got {actual_dtype}"
                        errors.append(error_msg)
                        
            elif expected_type == 'boolean':
                is_valid_type = pd.api.types.is_bool_dtype(col_data)
                if not is_valid_type:
                    # Check if it's binary numeric (0/1)
                    if pd.api.types.is_numeric_dtype(col_data):
                        unique_vals = col_data.dropna().unique()
                        if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                            warning_msg = f"Column '{column}' is numeric 0/1 but expected boolean"
                            warnings.append(warning_msg)
                            is_valid_type = True
                    
                    if not is_valid_type:
                        error_msg = f"Column '{column}' expected boolean, got {actual_dtype}"
                        errors.append(error_msg)
                        
            elif expected_type == 'string':
                is_valid_type = (pd.api.types.is_object_dtype(col_data) or 
                               pd.api.types.is_string_dtype(col_data))
                if not is_valid_type:
                    error_msg = f"Column '{column}' expected string, got {actual_dtype}"
                    errors.append(error_msg)
            else:
                warning_msg = f"Unknown expected type '{expected_type}' for column '{column}'"
                warnings.append(warning_msg)
            
            type_validation_results[column] = {
                'expected_type': expected_type,
                'actual_dtype': actual_dtype,
                'is_valid': is_valid_type,
                'unique_values': col_data.nunique(),
                'null_count': col_data.isnull().sum()
            }
        
        data_summary['type_validation'] = type_validation_results
        is_valid = len(errors) == 0
        
        if self.strict_mode and not is_valid:
            raise DataValidationError(f"Data type validation failed: {'; '.join(errors)}")
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            data_summary=data_summary
        )
    
    def _validate_schema(self, df: pd.DataFrame, schema: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Internal method to validate DataFrame against a schema.
        
        Args:
            df: DataFrame to validate
            schema: Schema dictionary with validation rules
            
        Returns:
            Tuple of (errors, warnings) lists
        """
        errors = []
        warnings = []
        
        # Check required columns
        if 'required_columns' in schema:
            result = self.check_required_columns(df, schema['required_columns'])
            errors.extend(result.errors)
            warnings.extend(result.warnings)
        
        # Check column types
        if 'column_types' in schema:
            result = self.validate_data_types(df, schema['column_types'])
            errors.extend(result.errors)
            warnings.extend(result.warnings)
        
        # Check row count constraints
        if 'min_rows' in schema:
            if len(df) < schema['min_rows']:
                errors.append(f"DataFrame has {len(df)} rows, minimum required: {schema['min_rows']}")
        
        if 'max_rows' in schema:
            if len(df) > schema['max_rows']:
                warnings.append(f"DataFrame has {len(df)} rows, maximum recommended: {schema['max_rows']}")
        
        # Check missing value policy
        if 'allow_missing' in schema and not schema['allow_missing']:
            if df.isnull().any().any():
                errors.append("Schema does not allow missing values, but missing values found")
        
        return errors, warnings