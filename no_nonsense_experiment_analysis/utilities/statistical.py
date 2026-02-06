"""
Statistical utility functions for experimental analysis.

This module provides common statistical operations including effect size calculations,
confidence interval estimation, and multiple comparison corrections.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Callable, List, Tuple, Optional, Union, Literal


class StatisticalFunctions:
    """Common statistical operations for experimental analysis.

    This class provides static methods for statistical calculations commonly
    used in experimental analysis, including effect sizes, confidence intervals,
    and p-value corrections.
    """

    # Supported effect size types
    EFFECT_SIZE_TYPES = ["cohens_d", "hedges_g", "glass_delta", "cohens_h"]

    # Supported correction methods
    CORRECTION_METHODS = ["bonferroni", "holm", "sidak", "benjamini_hochberg", "benjamini_yekutieli"]

    @staticmethod
    def calculate_effect_size(
        group1: Union[pd.Series, np.ndarray, List[float]],
        group2: Union[pd.Series, np.ndarray, List[float]],
        effect_type: str = "cohens_d"
    ) -> float:
        """Calculate effect size between two groups.

        Args:
            group1: Data for the first group
            group2: Data for the second group
            effect_type: Type of effect size to calculate:
                - "cohens_d": Cohen's d (pooled standard deviation)
                - "hedges_g": Hedges' g (bias-corrected Cohen's d)
                - "glass_delta": Glass's delta (uses control group SD)
                - "cohens_h": Cohen's h (for proportions)

        Returns:
            The calculated effect size value

        Raises:
            ValueError: If effect_type is not supported or data is invalid
        """
        # Convert to numpy arrays
        g1 = np.asarray(group1, dtype=float)
        g2 = np.asarray(group2, dtype=float)

        # Remove NaN values
        g1 = g1[~np.isnan(g1)]
        g2 = g2[~np.isnan(g2)]

        if len(g1) < 2 or len(g2) < 2:
            raise ValueError("Each group must have at least 2 non-null values")

        n1, n2 = len(g1), len(g2)
        mean1, mean2 = np.mean(g1), np.mean(g2)
        var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)

        if effect_type == "cohens_d":
            # Pooled standard deviation
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            if pooled_std == 0:
                return 0.0
            return (mean1 - mean2) / pooled_std

        elif effect_type == "hedges_g":
            # Cohen's d with bias correction
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            if pooled_std == 0:
                return 0.0
            d = (mean1 - mean2) / pooled_std
            # Hedges' correction factor
            correction = 1 - (3 / (4 * (n1 + n2) - 9))
            return d * correction

        elif effect_type == "glass_delta":
            # Uses group2 (control) standard deviation
            std2 = np.sqrt(var2)
            if std2 == 0:
                return 0.0
            return (mean1 - mean2) / std2

        elif effect_type == "cohens_h":
            # For proportions (assumes values are proportions between 0 and 1)
            # Clamp values to valid proportion range
            p1 = np.clip(mean1, 0.001, 0.999)
            p2 = np.clip(mean2, 0.001, 0.999)
            phi1 = 2 * np.arcsin(np.sqrt(p1))
            phi2 = 2 * np.arcsin(np.sqrt(p2))
            return phi1 - phi2

        else:
            raise ValueError(
                f"Unknown effect type '{effect_type}'. "
                f"Supported types: {StatisticalFunctions.EFFECT_SIZE_TYPES}"
            )

    @staticmethod
    def interpret_effect_size(effect_size: float, effect_type: str = "cohens_d") -> str:
        """Interpret the magnitude of an effect size.

        Args:
            effect_size: The effect size value
            effect_type: The type of effect size (affects interpretation thresholds)

        Returns:
            String interpretation: "negligible", "small", "medium", or "large"
        """
        abs_effect = abs(effect_size)

        if effect_type in ["cohens_d", "hedges_g", "glass_delta"]:
            # Cohen's conventions for d
            if abs_effect < 0.2:
                return "negligible"
            elif abs_effect < 0.5:
                return "small"
            elif abs_effect < 0.8:
                return "medium"
            else:
                return "large"
        elif effect_type == "cohens_h":
            # Cohen's conventions for h
            if abs_effect < 0.2:
                return "negligible"
            elif abs_effect < 0.5:
                return "small"
            elif abs_effect < 0.8:
                return "medium"
            else:
                return "large"
        else:
            # Default interpretation
            if abs_effect < 0.2:
                return "negligible"
            elif abs_effect < 0.5:
                return "small"
            elif abs_effect < 0.8:
                return "medium"
            else:
                return "large"

    @staticmethod
    def bootstrap_confidence_interval(
        data: Union[pd.Series, np.ndarray, List[float]],
        statistic: Callable[[np.ndarray], float] = np.mean,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
        random_state: Optional[int] = None
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for a statistic.

        Uses the percentile method to compute confidence intervals via
        bootstrap resampling.

        Args:
            data: The data to bootstrap
            statistic: Function that computes the statistic of interest.
                       Should accept a numpy array and return a scalar.
            confidence_level: Confidence level (default 0.95 for 95% CI)
            n_bootstrap: Number of bootstrap samples (default 1000)
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (lower_bound, upper_bound) for the confidence interval

        Raises:
            ValueError: If data is empty or confidence_level is invalid
        """
        # Convert to numpy array and remove NaN
        arr = np.asarray(data, dtype=float)
        arr = arr[~np.isnan(arr)]

        if len(arr) == 0:
            raise ValueError("Data must contain at least one non-null value")

        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")

        if n_bootstrap < 1:
            raise ValueError("n_bootstrap must be at least 1")

        # Set random state
        rng = np.random.default_rng(random_state)

        # Generate bootstrap samples and compute statistics
        bootstrap_stats = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            sample = rng.choice(arr, size=len(arr), replace=True)
            bootstrap_stats[i] = statistic(sample)

        # Calculate percentile-based confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower_bound = np.percentile(bootstrap_stats, lower_percentile)
        upper_bound = np.percentile(bootstrap_stats, upper_percentile)

        return (lower_bound, upper_bound)

    @staticmethod
    def bootstrap_two_sample_difference(
        group1: Union[pd.Series, np.ndarray, List[float]],
        group2: Union[pd.Series, np.ndarray, List[float]],
        statistic: Callable[[np.ndarray], float] = np.mean,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
        random_state: Optional[int] = None
    ) -> Tuple[float, float, float]:
        """Calculate bootstrap confidence interval for difference between two groups.

        Args:
            group1: Data for the first group
            group2: Data for the second group
            statistic: Function that computes the statistic (default: mean)
            confidence_level: Confidence level (default 0.95)
            n_bootstrap: Number of bootstrap samples
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (point_estimate, lower_bound, upper_bound)
        """
        g1 = np.asarray(group1, dtype=float)
        g2 = np.asarray(group2, dtype=float)
        g1 = g1[~np.isnan(g1)]
        g2 = g2[~np.isnan(g2)]

        if len(g1) == 0 or len(g2) == 0:
            raise ValueError("Both groups must contain at least one non-null value")

        rng = np.random.default_rng(random_state)

        # Point estimate
        point_estimate = statistic(g1) - statistic(g2)

        # Bootstrap differences
        bootstrap_diffs = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            sample1 = rng.choice(g1, size=len(g1), replace=True)
            sample2 = rng.choice(g2, size=len(g2), replace=True)
            bootstrap_diffs[i] = statistic(sample1) - statistic(sample2)

        alpha = 1 - confidence_level
        lower = np.percentile(bootstrap_diffs, (alpha / 2) * 100)
        upper = np.percentile(bootstrap_diffs, (1 - alpha / 2) * 100)

        return (point_estimate, lower, upper)

    @staticmethod
    def multiple_comparison_correction(
        p_values: Union[List[float], np.ndarray],
        method: str = "bonferroni",
        alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply multiple comparison correction to p-values.

        Args:
            p_values: List or array of p-values to correct
            method: Correction method:
                - "bonferroni": Bonferroni correction (conservative)
                - "holm": Holm-Bonferroni step-down method
                - "sidak": Sidak correction
                - "benjamini_hochberg": Benjamini-Hochberg FDR control
                - "benjamini_yekutieli": Benjamini-Yekutieli FDR control
            alpha: Significance level for determining significant results

        Returns:
            Tuple of (corrected_p_values, reject_null) where:
                - corrected_p_values: Array of adjusted p-values
                - reject_null: Boolean array indicating which hypotheses to reject

        Raises:
            ValueError: If method is not supported or p_values is empty
        """
        p_arr = np.asarray(p_values, dtype=float)

        if len(p_arr) == 0:
            raise ValueError("p_values cannot be empty")

        if np.any((p_arr < 0) | (p_arr > 1)):
            raise ValueError("All p-values must be between 0 and 1")

        n = len(p_arr)

        if method == "bonferroni":
            # Simple Bonferroni: multiply by n, cap at 1
            corrected = np.minimum(p_arr * n, 1.0)
            reject = corrected <= alpha

        elif method == "holm":
            # Holm-Bonferroni step-down procedure
            sorted_indices = np.argsort(p_arr)
            sorted_p = p_arr[sorted_indices]

            corrected = np.zeros(n)
            for i, (idx, p) in enumerate(zip(sorted_indices, sorted_p)):
                corrected[idx] = min(p * (n - i), 1.0)

            # Ensure monotonicity (later p-values can't be smaller)
            corrected_sorted = corrected[sorted_indices]
            for i in range(1, n):
                corrected_sorted[i] = max(corrected_sorted[i], corrected_sorted[i-1])
            corrected[sorted_indices] = corrected_sorted

            reject = corrected <= alpha

        elif method == "sidak":
            # Sidak correction: 1 - (1 - p)^n
            corrected = 1 - (1 - p_arr) ** n
            reject = corrected <= alpha

        elif method == "benjamini_hochberg":
            # Benjamini-Hochberg FDR control
            sorted_indices = np.argsort(p_arr)
            sorted_p = p_arr[sorted_indices]

            corrected = np.zeros(n)
            for i, (idx, p) in enumerate(zip(sorted_indices, sorted_p)):
                corrected[idx] = min(p * n / (i + 1), 1.0)

            # Ensure monotonicity (from largest to smallest)
            corrected_sorted = corrected[sorted_indices]
            for i in range(n - 2, -1, -1):
                corrected_sorted[i] = min(corrected_sorted[i], corrected_sorted[i+1])
            corrected[sorted_indices] = corrected_sorted

            reject = corrected <= alpha

        elif method == "benjamini_yekutieli":
            # Benjamini-Yekutieli (more conservative FDR for dependent tests)
            sorted_indices = np.argsort(p_arr)
            sorted_p = p_arr[sorted_indices]

            # Correction factor for dependence
            c_n = np.sum(1.0 / np.arange(1, n + 1))

            corrected = np.zeros(n)
            for i, (idx, p) in enumerate(zip(sorted_indices, sorted_p)):
                corrected[idx] = min(p * n * c_n / (i + 1), 1.0)

            # Ensure monotonicity
            corrected_sorted = corrected[sorted_indices]
            for i in range(n - 2, -1, -1):
                corrected_sorted[i] = min(corrected_sorted[i], corrected_sorted[i+1])
            corrected[sorted_indices] = corrected_sorted

            reject = corrected <= alpha

        else:
            raise ValueError(
                f"Unknown correction method '{method}'. "
                f"Supported methods: {StatisticalFunctions.CORRECTION_METHODS}"
            )

        return corrected, reject

    @staticmethod
    def calculate_power(
        effect_size: float,
        sample_size: int,
        alpha: float = 0.05,
        test_type: Literal["two_sample", "one_sample", "paired"] = "two_sample"
    ) -> float:
        """Calculate statistical power for a given effect size and sample size.

        Args:
            effect_size: The expected effect size (Cohen's d)
            sample_size: Sample size per group (for two_sample) or total (for one_sample)
            alpha: Significance level
            test_type: Type of test

        Returns:
            Statistical power (probability of detecting the effect)
        """
        if test_type == "two_sample":
            # Two-sample t-test
            df = 2 * sample_size - 2
            ncp = effect_size * np.sqrt(sample_size / 2)  # Non-centrality parameter
        elif test_type == "one_sample":
            df = sample_size - 1
            ncp = effect_size * np.sqrt(sample_size)
        elif test_type == "paired":
            df = sample_size - 1
            ncp = effect_size * np.sqrt(sample_size)
        else:
            raise ValueError(f"Unknown test_type: {test_type}")

        # Critical value for two-tailed test
        t_crit = stats.t.ppf(1 - alpha / 2, df)

        # Power is P(reject H0 | H1 is true) using non-central t distribution
        power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)

        return power

    @staticmethod
    def calculate_required_sample_size(
        effect_size: float,
        power: float = 0.8,
        alpha: float = 0.05,
        test_type: Literal["two_sample", "one_sample", "paired"] = "two_sample"
    ) -> int:
        """Calculate required sample size for desired power.

        Args:
            effect_size: The expected effect size (Cohen's d)
            power: Desired statistical power (default 0.8)
            alpha: Significance level (default 0.05)
            test_type: Type of test

        Returns:
            Required sample size per group (for two_sample) or total (for one_sample)
        """
        if effect_size == 0:
            raise ValueError("Effect size cannot be zero")

        # Binary search for sample size
        low, high = 2, 10000

        while low < high:
            mid = (low + high) // 2
            calculated_power = StatisticalFunctions.calculate_power(
                effect_size, mid, alpha, test_type
            )

            if calculated_power < power:
                low = mid + 1
            else:
                high = mid

        return low

    @staticmethod
    def calculate_confidence_interval(
        data: Union[pd.Series, np.ndarray, List[float]],
        confidence_level: float = 0.95,
        method: Literal["t", "normal", "bootstrap"] = "t"
    ) -> Tuple[float, float]:
        """Calculate confidence interval for the mean.

        Args:
            data: The data to analyze
            confidence_level: Confidence level (default 0.95)
            method: Method for CI calculation:
                - "t": t-distribution (recommended for small samples)
                - "normal": Normal approximation
                - "bootstrap": Bootstrap percentile method

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        arr = np.asarray(data, dtype=float)
        arr = arr[~np.isnan(arr)]

        if len(arr) < 2:
            raise ValueError("Data must have at least 2 non-null values")

        n = len(arr)
        mean = np.mean(arr)
        se = np.std(arr, ddof=1) / np.sqrt(n)

        if method == "t":
            t_crit = stats.t.ppf((1 + confidence_level) / 2, n - 1)
            margin = t_crit * se
            return (mean - margin, mean + margin)

        elif method == "normal":
            z_crit = stats.norm.ppf((1 + confidence_level) / 2)
            margin = z_crit * se
            return (mean - margin, mean + margin)

        elif method == "bootstrap":
            return StatisticalFunctions.bootstrap_confidence_interval(
                arr, np.mean, confidence_level
            )

        else:
            raise ValueError(f"Unknown method: {method}")
