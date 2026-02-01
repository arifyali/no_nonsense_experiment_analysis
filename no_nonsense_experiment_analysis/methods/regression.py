"""
Regression analysis method implementation.

This module provides regression analysis functionality for experimental designs
including linear regression and logistic regression.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error
from typing import Dict, Any, List, Optional
from .base import ExperimentalMethod
from ..core.models import MethodResult


class LinearRegressionAnalysis(ExperimentalMethod):
    """Linear regression analysis for continuous outcomes."""
    
    def __init__(self, alpha: float = 0.05, include_intercept: bool = True):
        """Initialize linear regression parameters.
        
        Args:
            alpha: Significance level for statistical tests
            include_intercept: Whether to include intercept in the model
        """
        self.alpha = alpha
        self.include_intercept = include_intercept
    
    def validate_inputs(self, data: pd.DataFrame, **kwargs) -> bool:
        """Validate inputs for linear regression.
        
        Args:
            data: DataFrame containing experimental data
            **kwargs: Additional parameters including:
                - target_col: Column name for the dependent variable
                - feature_cols: List of column names for independent variables
        
        Returns:
            True if inputs are valid, False otherwise
        """
        target_col = kwargs.get('target_col')
        feature_cols = kwargs.get('feature_cols')
        
        if target_col is None or feature_cols is None:
            return False
        
        if target_col not in data.columns:
            return False
        
        if not all(col in data.columns for col in feature_cols):
            return False
        
        # Check if target is numeric
        if not pd.api.types.is_numeric_dtype(data[target_col]):
            return False
        
        # Check if features are numeric
        for col in feature_cols:
            if not pd.api.types.is_numeric_dtype(data[col]):
                return False
        
        return True
    
    def execute(self, data: pd.DataFrame, **kwargs) -> MethodResult:
        """Execute linear regression analysis.
        
        Args:
            data: DataFrame containing experimental data
            **kwargs: Parameters including target_col and feature_cols
        
        Returns:
            MethodResult containing regression statistics and results
        """
        target_col = kwargs['target_col']
        feature_cols = kwargs['feature_cols']
        
        # Prepare data
        clean_data = data[[target_col] + feature_cols].dropna()
        X = clean_data[feature_cols]
        y = clean_data[target_col]
        
        # Fit regression model
        model = LinearRegression(fit_intercept=self.include_intercept)
        model.fit(X, y)
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate statistics
        n = len(y)
        k = len(feature_cols)
        
        # R-squared and adjusted R-squared
        r2 = r2_score(y, y_pred)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else r2
        
        # Mean squared error and RMSE
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        
        # Calculate coefficient statistics
        # For proper statistical inference, we need to calculate standard errors
        residuals = y - y_pred
        mse_residual = np.sum(residuals**2) / (n - k - 1)
        
        # Design matrix (add intercept if needed)
        if self.include_intercept:
            X_design = np.column_stack([np.ones(n), X])
            coef_names = ['intercept'] + feature_cols
            coefficients = np.concatenate([[model.intercept_], model.coef_])
        else:
            X_design = X.values
            coef_names = feature_cols
            coefficients = model.coef_
        
        # Calculate standard errors
        try:
            cov_matrix = mse_residual * np.linalg.inv(X_design.T @ X_design)
            std_errors = np.sqrt(np.diag(cov_matrix))
            
            # Calculate t-statistics and p-values
            t_stats = coefficients / std_errors
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k - 1))
            
            # Calculate confidence intervals
            t_critical = stats.t.ppf(1 - self.alpha/2, n - k - 1)
            ci_lower = coefficients - t_critical * std_errors
            ci_upper = coefficients + t_critical * std_errors
            
        except np.linalg.LinAlgError:
            # Handle singular matrix (multicollinearity)
            std_errors = np.full_like(coefficients, np.nan)
            t_stats = np.full_like(coefficients, np.nan)
            p_values = np.full_like(coefficients, np.nan)
            ci_lower = np.full_like(coefficients, np.nan)
            ci_upper = np.full_like(coefficients, np.nan)
        
        # F-statistic for overall model significance
        ss_total = np.sum((y - y.mean())**2)
        ss_residual = np.sum(residuals**2)
        ss_regression = ss_total - ss_residual
        
        f_stat = (ss_regression / k) / (ss_residual / (n - k - 1)) if n > k + 1 else np.nan
        f_p_value = 1 - stats.f.cdf(f_stat, k, n - k - 1) if not np.isnan(f_stat) else np.nan
        
        # Prepare results
        coef_stats = {}
        coef_p_values = {}
        coef_ci = {}
        
        for i, name in enumerate(coef_names):
            coef_stats[f'coef_{name}'] = coefficients[i]
            coef_stats[f'std_err_{name}'] = std_errors[i]
            coef_stats[f't_stat_{name}'] = t_stats[i]
            coef_p_values[f'coef_{name}'] = p_values[i]
            coef_ci[f'coef_{name}'] = (ci_lower[i], ci_upper[i])
        
        return MethodResult(
            method_name="Linear Regression",
            parameters={
                'target_col': target_col,
                'feature_cols': feature_cols,
                'alpha': self.alpha,
                'include_intercept': self.include_intercept
            },
            statistics={
                'r_squared': r2,
                'adjusted_r_squared': adj_r2,
                'mse': mse,
                'rmse': rmse,
                'f_statistic': f_stat,
                'sample_size': n,
                'num_features': k,
                **coef_stats
            },
            p_values={
                'f_test': f_p_value,
                **coef_p_values
            },
            confidence_intervals=coef_ci,
            effect_sizes={
                'r_squared': r2  # R² can be considered an effect size measure
            },
            metadata={
                'model_significant': f_p_value < self.alpha if not np.isnan(f_p_value) else False,
                'interpretation': self._interpret_results(r2, f_p_value, coefficients, p_values, coef_names),
                'assumptions_notes': "Check residual plots for linearity, homoscedasticity, and normality"
            }
        )
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get method parameters and descriptions."""
        return {
            'target_col': 'Column name for the dependent variable (continuous)',
            'feature_cols': 'List of column names for independent variables',
            'alpha': f'Significance level (default: {self.alpha})',
            'include_intercept': f'Include intercept term (default: {self.include_intercept})'
        }
    
    def _interpret_results(self, r2: float, f_p_value: float, coefficients: np.ndarray, 
                          p_values: np.ndarray, coef_names: List[str]) -> str:
        """Interpret the regression results."""
        # Model significance
        if not np.isnan(f_p_value):
            model_sig = "significant" if f_p_value < self.alpha else "not significant"
            model_text = f"The overall model is {model_sig} (F-test p={f_p_value:.4f})"
        else:
            model_text = "Unable to calculate F-test"
        
        # R-squared interpretation
        if r2 < 0.1:
            r2_interp = "weak"
        elif r2 < 0.3:
            r2_interp = "moderate"
        elif r2 < 0.7:
            r2_interp = "strong"
        else:
            r2_interp = "very strong"
        
        # Significant predictors
        sig_predictors = []
        for i, (name, p_val) in enumerate(zip(coef_names, p_values)):
            if not np.isnan(p_val) and p_val < self.alpha:
                sig_predictors.append(name)
        
        sig_text = f"Significant predictors: {', '.join(sig_predictors)}" if sig_predictors else "No significant predictors"
        
        return f"{model_text}. The model explains {r2:.1%} of the variance ({r2_interp} relationship). {sig_text}."


class LogisticRegressionAnalysis(ExperimentalMethod):
    """Logistic regression analysis for binary outcomes."""
    
    def __init__(self, alpha: float = 0.05, max_iter: int = 1000):
        """Initialize logistic regression parameters.
        
        Args:
            alpha: Significance level for statistical tests
            max_iter: Maximum iterations for convergence
        """
        self.alpha = alpha
        self.max_iter = max_iter
    
    def validate_inputs(self, data: pd.DataFrame, **kwargs) -> bool:
        """Validate inputs for logistic regression.
        
        Args:
            data: DataFrame containing experimental data
            **kwargs: Additional parameters including:
                - target_col: Column name for the dependent variable (binary)
                - feature_cols: List of column names for independent variables
        
        Returns:
            True if inputs are valid, False otherwise
        """
        target_col = kwargs.get('target_col')
        feature_cols = kwargs.get('feature_cols')
        
        if target_col is None or feature_cols is None:
            return False
        
        if target_col not in data.columns:
            return False
        
        if not all(col in data.columns for col in feature_cols):
            return False
        
        # Check if target is binary
        unique_values = data[target_col].nunique()
        if unique_values != 2:
            return False
        
        # Check if features are numeric
        for col in feature_cols:
            if not pd.api.types.is_numeric_dtype(data[col]):
                return False
        
        return True
    
    def execute(self, data: pd.DataFrame, **kwargs) -> MethodResult:
        """Execute logistic regression analysis.
        
        Args:
            data: DataFrame containing experimental data
            **kwargs: Parameters including target_col and feature_cols
        
        Returns:
            MethodResult containing logistic regression statistics and results
        """
        target_col = kwargs['target_col']
        feature_cols = kwargs['feature_cols']
        
        # Prepare data
        clean_data = data[[target_col] + feature_cols].dropna()
        X = clean_data[feature_cols]
        y = clean_data[target_col]
        
        # Ensure binary encoding (0, 1)
        unique_vals = sorted(y.unique())
        y_binary = (y == unique_vals[1]).astype(int)
        
        # Fit logistic regression model
        model = LogisticRegression(max_iter=self.max_iter, fit_intercept=True)
        model.fit(X, y_binary)
        
        # Make predictions
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)
        
        # Calculate statistics
        n = len(y_binary)
        k = len(feature_cols)
        
        # Accuracy and other metrics
        accuracy = np.mean(y_pred == y_binary)
        
        # Pseudo R-squared (McFadden's)
        null_log_likelihood = -n * (np.mean(y_binary) * np.log(np.mean(y_binary) + 1e-15) + 
                                   (1 - np.mean(y_binary)) * np.log(1 - np.mean(y_binary) + 1e-15))
        
        log_likelihood = np.sum(y_binary * np.log(y_pred_proba + 1e-15) + 
                               (1 - y_binary) * np.log(1 - y_pred_proba + 1e-15))
        
        mcfadden_r2 = 1 - (log_likelihood / null_log_likelihood)
        
        # Coefficients and odds ratios
        coefficients = np.concatenate([[model.intercept_[0]], model.coef_[0]])
        coef_names = ['intercept'] + feature_cols
        odds_ratios = np.exp(coefficients)
        
        # For proper inference, we'd need the Hessian matrix, but sklearn doesn't provide it
        # We'll provide basic statistics
        coef_stats = {}
        odds_ratio_stats = {}
        
        for i, name in enumerate(coef_names):
            coef_stats[f'coef_{name}'] = coefficients[i]
            odds_ratio_stats[f'odds_ratio_{name}'] = odds_ratios[i]
        
        return MethodResult(
            method_name="Logistic Regression",
            parameters={
                'target_col': target_col,
                'feature_cols': feature_cols,
                'alpha': self.alpha,
                'max_iter': self.max_iter
            },
            statistics={
                'accuracy': accuracy,
                'mcfadden_r2': mcfadden_r2,
                'log_likelihood': log_likelihood,
                'null_log_likelihood': null_log_likelihood,
                'sample_size': n,
                'num_features': k,
                **coef_stats,
                **odds_ratio_stats
            },
            p_values={
                # Note: sklearn doesn't provide p-values, would need statsmodels for proper inference
            },
            confidence_intervals={
                # Note: would need proper standard errors for confidence intervals
            },
            effect_sizes={
                'mcfadden_r2': mcfadden_r2
            },
            metadata={
                'target_classes': unique_vals,
                'interpretation': self._interpret_results(accuracy, mcfadden_r2, odds_ratios, coef_names),
                'note': "For statistical inference (p-values, CIs), consider using statsmodels"
            }
        )
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get method parameters and descriptions."""
        return {
            'target_col': 'Column name for the dependent variable (binary)',
            'feature_cols': 'List of column names for independent variables',
            'alpha': f'Significance level (default: {self.alpha})',
            'max_iter': f'Maximum iterations for convergence (default: {self.max_iter})'
        }
    
    def _interpret_results(self, accuracy: float, mcfadden_r2: float, 
                          odds_ratios: np.ndarray, coef_names: List[str]) -> str:
        """Interpret the logistic regression results."""
        # Model performance
        if accuracy < 0.6:
            acc_interp = "poor"
        elif accuracy < 0.7:
            acc_interp = "fair"
        elif accuracy < 0.8:
            acc_interp = "good"
        else:
            acc_interp = "excellent"
        
        # Pseudo R-squared interpretation
        if mcfadden_r2 < 0.1:
            r2_interp = "weak"
        elif mcfadden_r2 < 0.2:
            r2_interp = "moderate"
        elif mcfadden_r2 < 0.4:
            r2_interp = "strong"
        else:
            r2_interp = "very strong"
        
        return f"Model accuracy: {accuracy:.1%} ({acc_interp}). McFadden's R²: {mcfadden_r2:.3f} ({r2_interp} fit)."