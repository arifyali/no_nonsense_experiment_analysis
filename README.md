# no_nonsense_experiment_analysis

A Python companion library for experimental data analysis, designed to work alongside [The No-Nonsense Guide to Experimental Design](https://github.com/mustafaysir/no_nonsense_experimental_design).

## Overview

This package provides data scientists and analysts with a streamlined toolkit for experimental data analysis, implementing the practical methodologies outlined in [The No-Nonsense Guide to Experimental Design](https://github.com/mustafaysir/no_nonsense_experimental_design). The package follows a clear workflow from data preparation through analysis to reporting, accepting pandas DataFrames as input and focusing on tabular datasets.

## Related Resources

- **[The No-Nonsense Guide to Experimental Design](https://github.com/mustafaysir/no_nonsense_experimental_design)** - The companion guide that provides the theoretical foundation and practical methodology for experimental design that this library implements.

## Features

- **Data Preparation**: Comprehensive data validation, cleaning, and preprocessing
- **Experimental Methods**: Standardized interface for statistical analysis methods
- **Shared Utilities**: Common statistical functions, visualization tools, and data transformers
- **Workflow Management**: Structured pipeline from data loading to LLM-ready reporting
- **Property-Based Testing**: Rigorous correctness validation using Hypothesis

## Architecture

The package follows a modular three-component architecture:

- **data_prep**: Data validation, cleaning, and preprocessing
- **methods**: Experimental analysis methods and workflow management  
- **utilities**: Shared statistical functions, visualization, and data transformers

## Installation

```bash
pip install no_nonsense_experiment_analysis
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import pandas as pd
from no_nonsense_experiment_analysis import WorkflowManager

# Load your experimental data
data = pd.read_csv("your_experiment_data.csv")

# Create workflow manager
workflow = WorkflowManager(data)

# Execute analysis pipeline
report = (workflow
    .prep(strategy="drop_missing")
    .analyze("t_test", groups=["control", "treatment"])
    .report())

# Generate LLM-ready narrative prompt
narrative_prompt = report.to_llm_prompt()
print(narrative_prompt)
```

## Development

This package is under active development. The current implementation provides the core structure and interfaces. Individual components will be implemented incrementally.

### Running Tests

```bash
pytest tests/ -v
```

### Project Structure

```
no_nonsense_experiment_analysis/
├── core/                 # Core data models and workflow management
├── data_prep/           # Data validation, cleaning, preprocessing
├── methods/             # Experimental methods and registry
├── utilities/           # Statistical functions, visualization, transformers
└── tests/               # Test suite with unit and property-based tests
```

## Requirements

- Python >= 3.8
- pandas >= 1.5.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- scipy >= 1.9.0

## License

MIT License - see LICENSE file for details.

## Contributing

This project follows property-based testing principles and emphasizes correctness through formal verification. All contributions should include appropriate tests and maintain the existing architecture patterns.

For experimental design methodology and theoretical background, refer to [The No-Nonsense Guide to Experimental Design](https://github.com/mustafaysir/no_nonsense_experimental_design).