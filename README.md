# no_nonsense_experiment_analysis

A Python companion library for experimental data analysis, designed to work alongside [The No-Nonsense Guide to Experimental Design](https://github.com/mustafaysir/no_nonsense_experimental_design).

## Overview

This package provides data scientists and analysts with a streamlined toolkit for experimental data analysis, implementing the practical methodologies outlined in [The No-Nonsense Guide to Experimental Design](https://github.com/mustafaysir/no_nonsense_experimental_design). The package follows a clear workflow from data preparation through analysis to reporting, accepting pandas DataFrames as input and focusing on tabular datasets.

## Related Resources

- **[The No-Nonsense Guide to Experimental Design](https://github.com/mustafaysir/no_nonsense_experimental_design)** - The companion guide that provides the theoretical foundation and practical methodology for experimental design that this library implements.

## Features

- **Data Preparation**: Comprehensive data validation, cleaning, and preprocessing
- **Experimental Methods**: Standardized interface for statistical analysis methods including:
  - **A/B Testing**: Two-group comparison with effect size calculation
  - **ANOVA**: Multi-group analysis with post-hoc testing
  - **Chi-Square Tests**: Independence and goodness-of-fit testing for categorical data
  - **Regression Analysis**: Linear and logistic regression with comprehensive statistics
- **Shared Utilities**: Common statistical functions, visualization tools, and data transformers
- **Workflow Management**: Structured pipeline from data loading to LLM-ready reporting
- **Property-Based Testing**: Rigorous correctness validation using Hypothesis

## Architecture

The package follows a modular three-component architecture:

- **data_prep**: Data validation, cleaning, and preprocessing
- **methods**: Experimental analysis methods and workflow management  
- **utilities**: Shared statistical functions, visualization, and data transformers

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/arifyali/no_nonsense_experiment_analysis.git
```

Or clone and install locally:

```bash
git clone https://github.com/arifyali/no_nonsense_experiment_analysis.git
cd no_nonsense_experiment_analysis
pip install .
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import pandas as pd
from no_nonsense_experiment_analysis import WorkflowManager
from no_nonsense_experiment_analysis.methods import ABTest, default_registry

# Load your experimental data
data = pd.read_csv("your_experiment_data.csv")

# Option 1: Use individual methods directly
ab_test = ABTest(alpha=0.05)
result = ab_test.execute(data, group_col="treatment_group", metric_col="conversion_rate")

print(f"P-value: {result.p_values['two_sample_ttest']:.4f}")
print(f"Effect size: {result.effect_sizes['cohens_d']:.3f}")
print(f"Significant: {result.metadata['significant']}")

# Option 2: Use the method registry
method = default_registry.get_method('ab_test', alpha=0.05)
result = method.execute(data, group_col="treatment_group", metric_col="conversion_rate")

# Option 3: Use workflow manager (coming soon)
workflow = WorkflowManager(data)
report = (workflow
    .prep(strategy="drop_missing")
    .analyze("ab_test", group_col="treatment_group", metric_col="conversion_rate")
    .report())

# Generate LLM-ready narrative prompt
narrative_prompt = report.to_llm_prompt()
print(narrative_prompt)
```

### Available Methods

- **`ab_test`**: A/B testing for two-group comparisons
- **`one_way_anova`**: One-way ANOVA for multi-group comparisons  
- **`chi_square_independence`**: Chi-square test of independence
- **`chi_square_goodness_of_fit`**: Chi-square goodness of fit test
- **`linear_regression`**: Linear regression analysis
- **`logistic_regression`**: Logistic regression for binary outcomes

For detailed documentation of each method, including when to use them, implementation details, and examples, see [METHODS.md](docs/METHODS.md).

**Theoretical Background**: For comprehensive explanations of experimental design principles and methodologies, refer to the [Experimentation Primer](https://github.com/mustafaysir/no_nonsense_experimental_design/blob/main/Experimentation%20Primer.md) in the companion repository.

## Method Selection Quick Reference

| Data Type | Number of Groups | Method | Use Case |
|-----------|------------------|--------|----------|
| Continuous | 2 groups | `ab_test` | A/B testing, treatment vs control |
| Continuous | 3+ groups | `one_way_anova` | Multi-arm experiments |
| Continuous | Multiple predictors | `linear_regression` | Modeling relationships, controlling confounders |
| Categorical | 2+ groups | `chi_square_independence` | Testing associations |
| Categorical | Distribution test | `chi_square_goodness_of_fit` | Validating expected distributions |
| Binary | Multiple predictors | `logistic_regression` | Predicting binary outcomes |

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
- scikit-learn >= 1.1.0

## License

MIT License - see LICENSE file for details.

## Contributing

This project follows property-based testing principles and emphasizes correctness through formal verification. All contributions should include appropriate tests and maintain the existing architecture patterns.

For experimental design methodology and theoretical background, refer to [The No-Nonsense Guide to Experimental Design](https://github.com/mustafaysir/no_nonsense_experimental_design).