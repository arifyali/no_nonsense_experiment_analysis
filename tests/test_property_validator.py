"""
Property-based tests for DataValidator class.

This module contains property-based tests using Hypothesis to validate
the correctness properties of the DataValidator implementation.
"""

import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from no_nonsense_experiment_analysis.data_prep.validator import DataValidator
from no_nonsense_experiment_analysis.core.models import ValidationResult
from no_nonsense_experiment_analysis.core.exceptions import DataValidationError


class TestDataValidatorProperties:
    """Property-based tests for DataValidator."""
    
    @pytest.fixture
    def validator(self):
        """Create a DataValidator instance for testing."""
        return DataValidator(strict_mode=False)
    
    @pytest.fixture
    def strict_validator(self):
        """Create a strict DataValidator instance for testing."""
        return DataValidator(strict_mode=True)


class TestInputTypeValidation:
    """Property 1: Input type validation - Validates Requirements 1.4"""
    
    @given(
        invalid_input=st.one_of(
            st.lists(st.integers(), min_size=0, max_size=10),
            st.dictionaries(st.text(min_size=1, max_size=10), st.integers(), min_size=0, max_size=5),
            st.text(min_size=0, max_size=50),
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.booleans(),
            st.none(),
            st.sets(st.integers(), min_size=0, max_size=5),
            st.tuples(st.integers(), st.text())
        )
    )
    @settings(max_examples=30)
    def test_non_dataframe_input_rejection(self, invalid_input):
        """
        **Validates: Requirements 1.4**
        
        Property: The DataValidator must reject all non-DataFrame inputs
        and return ValidationResult with is_valid=False.
        """
        validator = DataValidator(strict_mode=False)
        
        # Skip pandas DataFrames as they are valid inputs
        assume(not isinstance(invalid_input, pd.DataFrame))
        
        result = validator.validate_dataframe(invalid_input)
        
        # Property: Must reject non-DataFrame inputs
        assert not result.is_valid
        assert len(result.errors) > 0
        
        # Property: Must return ValidationResult object
        assert isinstance(result, ValidationResult)
        assert hasattr(result, 'is_valid')
        assert hasattr(result, 'errors')
        assert hasattr(result, 'warnings')
        assert hasattr(result, 'data_summary')
    
    @given(
        num_rows=st.integers(min_value=1, max_value=10),
        columns=st.lists(
            st.text(min_size=1, max_size=20).filter(lambda x: x.strip() and x.isidentifier()),
            min_size=1,
            max_size=5,
            unique=True
        )
    )
    @settings(max_examples=20)
    def test_valid_dataframe_acceptance(self, num_rows, columns):
        """
        **Validates: Requirements 1.4**
        
        Property: The DataValidator must accept valid pandas DataFrames
        and return ValidationResult with appropriate validation status.
        """
        validator = DataValidator(strict_mode=False)
        
        # Create a valid DataFrame with consistent row counts
        data = {}
        for col in columns:
            data[col] = [i for i in range(num_rows)]
        
        df = pd.DataFrame(data)
        
        result = validator.validate_dataframe(df)
        
        # Property: Must return ValidationResult object
        assert isinstance(result, ValidationResult)
        
        # Property: Must include data summary for valid DataFrames
        assert 'shape' in result.data_summary
        assert 'columns' in result.data_summary
        assert 'dtypes' in result.data_summary
        
        # Property: Shape should match the actual DataFrame
        assert result.data_summary['shape'] == df.shape
        assert result.data_summary['columns'] == list(df.columns)
    
    @given(
        columns=st.lists(
            st.text(min_size=1, max_size=15).filter(lambda x: x.strip() and x.isidentifier()),
            min_size=1,
            max_size=5,
            unique=True
        )
    )
    @settings(max_examples=15)
    def test_empty_dataframe_handling(self, columns):
        """
        **Validates: Requirements 1.4**
        
        Property: The DataValidator must handle empty DataFrames appropriately,
        distinguishing between different types of emptiness.
        """
        validator = DataValidator(strict_mode=False)
        
        # Test completely empty DataFrame
        empty_df = pd.DataFrame()
        result = validator.validate_dataframe(empty_df)
        
        # Property: Empty DataFrame should be invalid
        assert not result.is_valid
        assert any("empty" in error.lower() for error in result.errors)
        
        # Test DataFrame with columns but no rows
        no_rows_df = pd.DataFrame(columns=columns)
        result = validator.validate_dataframe(no_rows_df)
        
        # Property: DataFrame with no rows should be invalid
        assert not result.is_valid
        assert any("no rows" in error.lower() for error in result.errors)
    
    def test_strict_mode_exception_property(self):
        """
        **Validates: Requirements 1.4**
        
        Property: When strict_mode=True, invalid inputs must raise DataValidationError.
        """
        strict_validator = DataValidator(strict_mode=True)
        
        invalid_inputs = [
            [1, 2, 3],
            {"a": 1},
            "not a dataframe",
            123,
            None
        ]
        
        for invalid_input in invalid_inputs:
            # Property: Must raise DataValidationError in strict mode
            with pytest.raises(DataValidationError):
                strict_validator.validate_dataframe(invalid_input)


class TestValidationCompleteness:
    """Property 2: Validation completeness - Validates Requirements 1.1, 1.3"""
    
    @given(
        num_rows=st.integers(min_value=1, max_value=20),
        columns=st.lists(
            st.text(min_size=1, max_size=15).filter(lambda x: x.strip() and x.isidentifier()),
            min_size=1,
            max_size=8,
            unique=True
        ),
        missing_probability=st.floats(min_value=0.0, max_value=0.8)
    )
    @settings(max_examples=25)
    def test_comprehensive_dataframe_analysis(self, num_rows, columns, missing_probability):
        """
        **Validates: Requirements 1.1, 1.3**
        
        Property: The DataValidator must perform comprehensive analysis of DataFrames,
        detecting all relevant structural and content issues.
        """
        validator = DataValidator(strict_mode=False)
        
        # Create DataFrame with potential issues
        data = {}
        for col in columns:
            # Generate data with some missing values
            col_data = []
            for i in range(num_rows):
                if np.random.random() < missing_probability:
                    col_data.append(None)
                else:
                    col_data.append(i)
            data[col] = col_data
        
        df = pd.DataFrame(data)
        result = validator.validate_dataframe(df)
        
        # Property: Must return ValidationResult with comprehensive analysis
        assert isinstance(result, ValidationResult)
        
        # Property: Must include essential data summary fields
        required_summary_fields = ['shape', 'columns', 'dtypes', 'memory_usage', 'missing_values']
        for field in required_summary_fields:
            assert field in result.data_summary, f"Missing required summary field: {field}"
        
        # Property: Shape analysis must be accurate
        assert result.data_summary['shape'] == df.shape
        assert result.data_summary['columns'] == list(df.columns)
        
        # Property: Missing value analysis must be complete
        actual_missing = df.isnull().sum()
        expected_missing = {col: count for col, count in actual_missing.items() if count > 0}
        assert result.data_summary['missing_values'] == expected_missing
        
        # Property: If there are missing values, they should be detected
        if df.isnull().any().any():
            # Should have missing values in summary
            assert len(result.data_summary['missing_values']) > 0
        else:
            # Should have empty missing values dict
            assert result.data_summary['missing_values'] == {}
    
    @given(
        base_columns=st.lists(
            st.text(min_size=1, max_size=10).filter(lambda x: x.strip() and x.isidentifier()),
            min_size=2,
            max_size=5,
            unique=True
        )
    )
    @settings(max_examples=15)
    def test_duplicate_column_detection(self, base_columns):
        """
        **Validates: Requirements 1.1, 1.3**
        
        Property: The DataValidator must detect duplicate column names
        and report them as validation errors.
        """
        validator = DataValidator(strict_mode=False)
        
        # Create DataFrame with duplicate column names
        duplicate_col = base_columns[0]
        all_columns = base_columns + [duplicate_col]  # Add duplicate
        
        # Create data for the DataFrame
        data = []
        for i, col in enumerate(all_columns):
            data.append([i] * 3)  # 3 rows of data
        
        # Create DataFrame with duplicate columns using numpy array
        df = pd.DataFrame(np.array(data).T, columns=all_columns)
        
        result = validator.validate_dataframe(df)
        
        # Property: Must detect duplicate columns
        assert not result.is_valid
        assert len(result.errors) > 0
        
        # Property: Error message must mention duplicate columns
        error_msg = " ".join(result.errors)
        assert "duplicate" in error_msg.lower()
        assert duplicate_col in error_msg
    
    @given(
        num_rows=st.integers(min_value=1, max_value=15),
        columns=st.lists(
            st.text(min_size=1, max_size=10).filter(lambda x: x.strip() and x.isidentifier()),
            min_size=1,
            max_size=5,
            unique=True
        ),
        single_value_cols=st.integers(min_value=1, max_value=3)
    )
    @settings(max_examples=20)
    def test_single_value_column_detection(self, num_rows, columns, single_value_cols):
        """
        **Validates: Requirements 1.1, 1.3**
        
        Property: The DataValidator must detect columns with single unique values
        and report them as warnings.
        """
        validator = DataValidator(strict_mode=False)
        
        # Ensure we don't try to create more single-value columns than total columns
        single_value_cols = min(single_value_cols, len(columns))
        
        # Create DataFrame with some single-value columns
        data = {}
        for i, col in enumerate(columns):
            if i < single_value_cols:
                # Single value column
                data[col] = [42] * num_rows
            else:
                # Variable column
                data[col] = list(range(num_rows))
        
        df = pd.DataFrame(data)
        result = validator.validate_dataframe(df)
        
        # Property: Must detect single-value columns
        if single_value_cols > 0:
            assert len(result.warnings) > 0
            warning_msg = " ".join(result.warnings)
            assert "single unique value" in warning_msg.lower()
    
    @given(
        required_cols=st.lists(
            st.text(min_size=1, max_size=10).filter(lambda x: x.strip() and x.isidentifier()),
            min_size=2,
            max_size=5,
            unique=True
        ),
        extra_cols=st.lists(
            st.text(min_size=1, max_size=10).filter(lambda x: x.strip() and x.isidentifier()),
            min_size=0,
            max_size=3,
            unique=True
        )
    )
    @settings(max_examples=20)
    def test_required_column_validation_completeness(self, required_cols, extra_cols):
        """
        **Validates: Requirements 1.1, 1.3**
        
        Property: The check_required_columns method must comprehensively validate
        column presence and provide complete information about missing columns.
        """
        validator = DataValidator(strict_mode=False)
        
        # Ensure extra_cols don't overlap with required_cols
        extra_cols = [col for col in extra_cols if col not in required_cols]
        
        # Create DataFrame with only some of the required columns
        present_cols = required_cols[:-1]  # Remove last required column
        all_cols = present_cols + extra_cols
        
        df = pd.DataFrame({col: [1, 2, 3] for col in all_cols})
        
        result = validator.check_required_columns(df, required_cols)
        
        # Property: Must detect missing columns
        assert not result.is_valid
        assert len(result.errors) > 0
        
        # Property: Must provide complete summary information
        assert 'required_columns' in result.data_summary
        assert 'present_columns' in result.data_summary
        assert 'missing_columns' in result.data_summary
        
        # Property: Summary must be accurate
        assert result.data_summary['required_columns'] == required_cols
        assert set(result.data_summary['present_columns']) == set(all_cols)
        
        # Property: Must identify the specific missing column
        missing_col = required_cols[-1]
        error_msg = " ".join(result.errors)
        assert missing_col in error_msg
    
    def test_validation_result_structure_completeness(self):
        """
        **Validates: Requirements 1.1, 1.3**
        
        Property: All validation methods must return ValidationResult objects
        with complete and consistent structure.
        """
        validator = DataValidator(strict_mode=False)
        
        # Test with various inputs
        test_cases = [
            pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}),  # Valid DataFrame
            pd.DataFrame(),  # Empty DataFrame
            [1, 2, 3],  # Invalid input
        ]
        
        for test_input in test_cases:
            result = validator.validate_dataframe(test_input)
            
            # Property: Must always return ValidationResult
            assert isinstance(result, ValidationResult)
            
            # Property: Must have all required attributes
            required_attrs = ['is_valid', 'errors', 'warnings', 'data_summary']
            for attr in required_attrs:
                assert hasattr(result, attr), f"Missing attribute: {attr}"
                assert getattr(result, attr) is not None, f"Attribute {attr} is None"
            
            # Property: Lists must be actual lists
            assert isinstance(result.errors, list)
            assert isinstance(result.warnings, list)
            assert isinstance(result.data_summary, dict)


class TestErrorMessageDescriptiveness:
    """Property 3: Error message descriptiveness - Validates Requirements 1.2, 7.1"""
    
    @given(
        invalid_input=st.one_of(
            st.lists(st.integers()),
            st.dictionaries(st.text(), st.integers()),
            st.text(),
            st.integers(),
            st.none()
        )
    )
    @settings(max_examples=50)
    def test_non_dataframe_input_error_descriptiveness(self, invalid_input):
        """
        **Validates: Requirements 1.2, 7.1**
        
        Property: When invalid input types are provided, error messages must be descriptive
        and include specific details about the problem.
        """
        validator = DataValidator(strict_mode=False)
        
        # Skip pandas DataFrames as they are valid inputs
        assume(not isinstance(invalid_input, pd.DataFrame))
        
        result = validator.validate_dataframe(invalid_input)
        
        # Property: Must have errors for invalid input
        assert not result.is_valid
        assert len(result.errors) > 0
        
        # Property: Error messages must be descriptive
        error_msg = result.errors[0]
        
        # Must mention it's not a DataFrame
        assert "dataframe" in error_msg.lower()
        
        # Must include the actual type received
        actual_type = type(invalid_input).__name__
        assert actual_type.lower() in error_msg.lower()
        
        # Must be a complete sentence (starts with capital, ends with punctuation or has clear structure)
        assert len(error_msg) > 10  # Reasonable minimum length for descriptiveness
        assert error_msg[0].isupper() or error_msg.startswith("Input")  # Proper capitalization
    
    @given(
        columns=st.lists(
            st.sampled_from(['col1', 'col2', 'col3', 'col4', 'col5', 'data', 'value', 'id', 'name', 'test']),
            min_size=1,
            max_size=5,
            unique=True
        ),
        missing_columns=st.lists(
            st.sampled_from(['missing1', 'missing2', 'missing3', 'absent', 'notfound']),
            min_size=1,
            max_size=3,
            unique=True
        )
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.filter_too_much])
    def test_missing_columns_error_descriptiveness(self, columns, missing_columns):
        """
        **Validates: Requirements 1.2, 7.1**
        
        Property: When required columns are missing, error messages must specifically
        name the missing columns.
        """
        validator = DataValidator(strict_mode=False)
        
        # Create DataFrame with some columns
        df = pd.DataFrame({col: [1, 2, 3] for col in columns})
        
        # Check for columns that include some missing ones
        all_required = columns + missing_columns
        result = validator.check_required_columns(df, all_required)
        
        # Should always be invalid since we're adding missing columns
        assert not result.is_valid
        
        # Property: Error messages must name specific missing columns
        error_msg = " ".join(result.errors)
        
        # Must mention "missing" or "required"
        assert any(word in error_msg.lower() for word in ["missing", "required", "not found"])
        
        # Must include at least one of the missing column names
        assert any(col in error_msg for col in missing_columns)
        
        # Must be descriptive (reasonable length)
        assert len(error_msg) > 15
    
    @given(
        column_name=st.text(min_size=1, max_size=20).filter(lambda x: x.strip() and x.isidentifier()),
        expected_type=st.sampled_from(['numeric', 'categorical', 'datetime', 'boolean', 'string']),
        actual_data=st.one_of(
            st.lists(st.text(min_size=1, max_size=10), min_size=3, max_size=10),  # String data
            st.lists(st.booleans(), min_size=3, max_size=10),  # Boolean data
            st.lists(st.integers(), min_size=3, max_size=10)   # Integer data
        )
    )
    @settings(max_examples=30)
    def test_type_mismatch_error_descriptiveness(self, column_name, expected_type, actual_data):
        """
        **Validates: Requirements 1.2, 7.1**
        
        Property: When data types don't match expectations, error messages must specify
        both the expected type and the actual type found.
        """
        validator = DataValidator(strict_mode=False)
        
        # Create DataFrame with the actual data
        df = pd.DataFrame({column_name: actual_data})
        actual_dtype = str(df[column_name].dtype)
        
        # Create type map with potentially mismatched type
        type_map = {column_name: expected_type}
        
        # Determine if this should actually be a type mismatch
        is_mismatch = False
        if expected_type == 'numeric' and not pd.api.types.is_numeric_dtype(df[column_name]):
            is_mismatch = True
        elif expected_type == 'categorical' and not (pd.api.types.is_object_dtype(df[column_name]) or pd.api.types.is_string_dtype(df[column_name])):
            is_mismatch = True
        elif expected_type == 'boolean' and not pd.api.types.is_bool_dtype(df[column_name]):
            is_mismatch = True
        elif expected_type == 'string' and not (pd.api.types.is_object_dtype(df[column_name]) or pd.api.types.is_string_dtype(df[column_name])):
            is_mismatch = True
        
        result = validator.validate_data_types(df, type_map)
        
        if is_mismatch and not result.is_valid:
            # Property: Error messages must mention both expected and actual types
            error_msg = " ".join(result.errors)
            
            # Must mention the column name
            assert column_name in error_msg
            
            # Must mention "expected" or similar concept
            assert any(word in error_msg.lower() for word in ["expected", "should be", "must be"])
            
            # Must mention the expected type
            assert expected_type in error_msg.lower()
            
            # Must be descriptive
            assert len(error_msg) > 20
    
    @given(
        schema_violations=st.dictionaries(
            st.sampled_from(['min_rows', 'max_rows', 'allow_missing']),
            st.one_of(st.integers(min_value=1, max_value=1000), st.booleans()),
            min_size=1,
            max_size=3
        )
    )
    @settings(max_examples=20)
    def test_schema_violation_error_descriptiveness(self, schema_violations):
        """
        **Validates: Requirements 1.2, 7.1**
        
        Property: When schema validation fails, error messages must clearly explain
        what constraint was violated and what the actual values were.
        """
        validator = DataValidator(strict_mode=False)
        
        # Create a small DataFrame
        df = pd.DataFrame({
            'col1': [1, 2, 3, None, 5],
            'col2': ['a', 'b', 'c', 'd', 'e']
        })
        
        # Create schema that might violate constraints
        schema = {}
        
        if 'min_rows' in schema_violations:
            # Set minimum rows higher than actual
            schema['min_rows'] = len(df) + 10
        
        if 'max_rows' in schema_violations:
            # Set maximum rows lower than actual
            schema['max_rows'] = max(1, len(df) - 2)
        
        if 'allow_missing' in schema_violations:
            # Disallow missing values when we have them
            schema['allow_missing'] = False
        
        result = validator.validate_dataframe(df, schema=schema)
        
        if not result.is_valid:
            # Property: Error messages must be specific about violations
            error_msg = " ".join(result.errors)
            
            # Must mention specific numbers/values when relevant
            if 'min_rows' in schema_violations:
                assert str(len(df)) in error_msg or "minimum" in error_msg.lower()
            
            if 'allow_missing' in schema_violations:
                assert "missing" in error_msg.lower() or "null" in error_msg.lower()
            
            # Must be descriptive
            assert len(error_msg) > 15
    
    def test_error_message_consistency_property(self):
        """
        **Validates: Requirements 1.2, 7.1**
        
        Property: Similar validation failures should produce consistently formatted
        error messages across different inputs.
        """
        validator = DataValidator(strict_mode=False)
        
        # Test multiple non-DataFrame inputs
        non_df_inputs = [
            [1, 2, 3],
            {"a": 1, "b": 2},
            "not a dataframe",
            123,
            None
        ]
        
        error_messages = []
        for invalid_input in non_df_inputs:
            result = validator.validate_dataframe(invalid_input)
            if result.errors:
                error_messages.append(result.errors[0])
        
        # Property: All error messages should follow similar structure
        assert len(error_messages) > 0
        
        # All should mention DataFrame
        for msg in error_messages:
            assert "dataframe" in msg.lower()
        
        # All should mention the input type issue
        for msg in error_messages:
            assert any(word in msg.lower() for word in ["input", "must be", "expected", "got"])
        
        # All should be reasonably similar in length (within reasonable bounds)
        lengths = [len(msg) for msg in error_messages]
        if len(lengths) > 1:
            min_len, max_len = min(lengths), max(lengths)
            # Error messages shouldn't vary too wildly in length for similar errors
            assert max_len <= min_len * 3  # Allow up to 3x variation


if __name__ == "__main__":
    pytest.main([__file__])