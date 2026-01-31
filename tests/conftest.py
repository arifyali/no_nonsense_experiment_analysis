"""
Pytest configuration and shared fixtures for the test suite.
"""

import pytest
import pandas as pd
import numpy as np
from hypothesis import settings, Verbosity

# Configure Hypothesis for property-based testing
settings.register_profile("default", max_examples=100, verbosity=Verbosity.normal)
settings.load_profile("default")


@pytest.fixture
def sample_dataframe():
    """Provide a sample DataFrame for testing."""
    np.random.seed(42)  # For reproducible tests
    return pd.DataFrame({
        'id': range(1, 101),
        'group': np.random.choice(['A', 'B', 'C'], 100),
        'value': np.random.normal(10, 2, 100),
        'category': np.random.choice(['X', 'Y'], 100),
        'score': np.random.uniform(0, 100, 100)
    })


@pytest.fixture
def empty_dataframe():
    """Provide an empty DataFrame for testing edge cases."""
    return pd.DataFrame()


@pytest.fixture
def dataframe_with_missing():
    """Provide a DataFrame with missing values for testing."""
    np.random.seed(42)
    df = pd.DataFrame({
        'id': range(1, 51),
        'value': np.random.normal(10, 2, 50),
        'category': np.random.choice(['A', 'B', None], 50)
    })
    # Introduce some NaN values
    df.loc[df.index[:5], 'value'] = np.nan
    return df


@pytest.fixture
def invalid_input_types():
    """Provide various invalid input types for testing validation."""
    return [
        "not_a_dataframe",
        123,
        [],
        {},
        None,
        np.array([1, 2, 3])
    ]