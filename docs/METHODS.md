# Experimental Methods Documentation

This document provides detailed information about the experimental design methods implemented in the `no_nonsense_experiment_analysis` package. These methods implement the practical methodologies outlined in [The No-Nonsense Guide to Experimental Design](https://github.com/mustafaysir/no_nonsense_experimental_design).

For theoretical background and detailed explanations of experimental design principles, please refer to the [Experimentation Primer](https://github.com/mustafaysir/no_nonsense_experimental_design/blob/main/Experimentation%20Primer.md) in the companion repository.

## Available Methods

### 1. A/B Testing (`ab_test`)

**Purpose**: Compare two groups (control vs treatment) to determine if there's a statistically significant difference in a continuous metric.

**When to Use**:
- Comparing two versions of a product, feature, or treatment
- Testing the effect of a single intervention
- Simple randomized controlled experiments

**Implementation Details**:
- Uses Welch's t-test (unequal variances assumed)
- Calculates Cohen's d for effect size
- Provides confidence intervals for the difference in means
- Includes statistical power considerations

**Parameters**:
- `group_col`: Column containing group assignments ('A', 'B' or any two values)
- `metric_col`: Column containing the continuous outcome metric
- `alpha`: Significance level (default: 0.05)
- `alternative`: Type of test ('two-sided', 'less', 'greater')

**Output Statistics**:
- Mean and standard deviation for each group
- T-statistic and p-value
- Cohen's d effect size
- Confidence interval for difference in means
- Statistical significance interpretation

**Example**:
```python
from no_nonsense_experiment_analysis.methods import ABTest

ab_test = ABTest(alpha=0.05)
result = ab_test.execute(data, group_col='treatment', metric_col='conversion_rate')
```

---

### 2. One-Way ANOVA (`one_way_anova`)

**Purpose**: Compare means across multiple groups (3 or more) to test if at least one group differs significantly from the others.

**When to Use**:
- Comparing multiple treatments or conditions
- Testing the effect of a categorical variable with multiple levels
- Multi-arm experiments

**Implementation Details**:
- Uses F-test for overall significance
- Calculates eta-squared (η²) for effect size
- Optional post-hoc pairwise comparisons with Bonferroni correction
- Assumes normality and homogeneity of variance

**Parameters**:
- `group_col`: Column containing group assignments (3+ groups)
- `metric_col`: Column containing the continuous outcome metric
- `alpha`: Significance level (default: 0.05)
- `post_hoc`: Whether to perform pairwise comparisons (default: True)

**Output Statistics**:
- F-statistic and p-value
- Degrees of freedom (between and within groups)
- Eta-squared effect size
- Group means, standard deviations, and sample sizes
- Post-hoc pairwise comparison results (if requested)

**Example**:
```python
from no_nonsense_experiment_analysis.methods import OneWayANOVA

anova = OneWayANOVA(alpha=0.05, post_hoc=True)
result = anova.execute(data, group_col='treatment', metric_col='outcome')
```

---

### 3. Chi-Square Test of Independence (`chi_square_independence`)

**Purpose**: Test whether two categorical variables are independent of each other.

**When to Use**:
- Testing association between treatment and categorical outcomes
- Analyzing contingency tables
- Testing if proportions differ across groups

**Implementation Details**:
- Uses Pearson's chi-square test
- Calculates Cramér's V for effect size
- Provides standardized residuals for cell-level analysis
- Checks assumptions (expected frequencies ≥ 5)

**Parameters**:
- `group_col`: Column containing group assignments (categorical)
- `outcome_col`: Column containing categorical outcomes
- `alpha`: Significance level (default: 0.05)

**Output Statistics**:
- Chi-square statistic and p-value
- Degrees of freedom
- Cramér's V effect size
- Contingency table and expected frequencies
- Standardized residuals
- Assumption validation

**Example**:
```python
from no_nonsense_experiment_analysis.methods import ChiSquareTest

chi_test = ChiSquareTest(test_type='independence')
result = chi_test.execute(data, group_col='treatment', outcome_col='success')
```

---

### 4. Chi-Square Goodness of Fit Test (`chi_square_goodness_of_fit`)

**Purpose**: Test whether observed categorical data follows an expected distribution.

**When to Use**:
- Testing if data follows a uniform distribution
- Comparing observed frequencies to theoretical expectations
- Validating randomization procedures

**Implementation Details**:
- Compares observed vs expected frequencies
- Defaults to uniform distribution if no expected frequencies provided
- Calculates effect size measure

**Parameters**:
- `observed_col`: Column containing categorical observations
- `expected`: List of expected frequencies (optional, defaults to uniform)
- `alpha`: Significance level (default: 0.05)

**Output Statistics**:
- Chi-square statistic and p-value
- Observed vs expected frequencies
- Effect size measure
- Assumption validation

**Example**:
```python
from no_nonsense_experiment_analysis.methods import ChiSquareTest

chi_test = ChiSquareTest(test_type='goodness_of_fit')
result = chi_test.execute(data, observed_col='category')
```

---

### 5. Linear Regression (`linear_regression`)

**Purpose**: Model the relationship between continuous predictors and a continuous outcome variable.

**When to Use**:
- Understanding relationships between variables
- Controlling for confounding variables
- Predicting continuous outcomes
- Testing multiple factors simultaneously

**Implementation Details**:
- Uses ordinary least squares (OLS) regression
- Calculates R² and adjusted R²
- Provides coefficient statistics with standard errors
- Includes F-test for overall model significance
- Calculates confidence intervals for coefficients

**Parameters**:
- `target_col`: Column containing the continuous dependent variable
- `feature_cols`: List of columns containing independent variables
- `alpha`: Significance level (default: 0.05)
- `include_intercept`: Whether to include intercept term (default: True)

**Output Statistics**:
- R² and adjusted R²
- F-statistic for overall model
- Coefficient estimates with standard errors
- T-statistics and p-values for each coefficient
- Confidence intervals for coefficients
- RMSE and other fit statistics

**Example**:
```python
from no_nonsense_experiment_analysis.methods import LinearRegressionAnalysis

regression = LinearRegressionAnalysis()
result = regression.execute(data, target_col='outcome', feature_cols=['x1', 'x2', 'x3'])
```

---

### 6. Logistic Regression (`logistic_regression`)

**Purpose**: Model the relationship between predictors and a binary outcome variable.

**When to Use**:
- Predicting binary outcomes (success/failure, yes/no)
- Understanding factors affecting probability of an event
- Classification problems
- Analyzing odds ratios

**Implementation Details**:
- Uses maximum likelihood estimation
- Calculates odds ratios for interpretability
- Provides McFadden's pseudo R²
- Includes model accuracy metrics

**Parameters**:
- `target_col`: Column containing the binary dependent variable
- `feature_cols`: List of columns containing independent variables
- `alpha`: Significance level (default: 0.05)
- `max_iter`: Maximum iterations for convergence (default: 1000)

**Output Statistics**:
- Model accuracy
- McFadden's pseudo R²
- Coefficient estimates and odds ratios
- Log-likelihood statistics
- Classification performance metrics

**Example**:
```python
from no_nonsense_experiment_analysis.methods import LogisticRegressionAnalysis

logistic = LogisticRegressionAnalysis()
result = logistic.execute(data, target_col='success', feature_cols=['x1', 'x2'])
```

---

## Method Selection Guide

### For Continuous Outcomes:
- **2 groups**: Use A/B Testing
- **3+ groups**: Use One-Way ANOVA
- **Multiple predictors**: Use Linear Regression

### For Categorical Outcomes:
- **Testing association**: Use Chi-Square Independence Test
- **Testing distribution**: Use Chi-Square Goodness of Fit Test
- **Binary prediction**: Use Logistic Regression

### Effect Size Interpretation:

| Method | Effect Size Measure | Small | Medium | Large |
|--------|-------------------|-------|--------|-------|
| A/B Test | Cohen's d | 0.2 | 0.5 | 0.8 |
| ANOVA | Eta-squared (η²) | 0.01 | 0.06 | 0.14 |
| Chi-Square | Cramér's V | 0.1 | 0.3 | 0.5 |
| Linear Regression | R² | 0.02 | 0.13 | 0.26 |

## Assumptions and Limitations

### A/B Testing & ANOVA:
- **Normality**: Data should be approximately normally distributed
- **Independence**: Observations should be independent
- **Homogeneity**: Equal variances across groups (for ANOVA)

### Chi-Square Tests:
- **Expected frequencies**: All expected cell counts should be ≥ 5
- **Independence**: Observations should be independent
- **Categorical data**: Variables should be truly categorical

### Regression Methods:
- **Linearity**: Relationship should be linear (for linear regression)
- **Independence**: Residuals should be independent
- **Homoscedasticity**: Constant variance of residuals
- **No multicollinearity**: Predictors should not be highly correlated

## References

- [The No-Nonsense Guide to Experimental Design](https://github.com/mustafaysir/no_nonsense_experimental_design)
- [Experimentation Primer](https://github.com/mustafaysir/no_nonsense_experimental_design/blob/main/Experimentation%20Primer.md)
- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences
- Field, A. (2013). Discovering Statistics Using IBM SPSS Statistics
- Montgomery, D. C. (2017). Design and Analysis of Experiments