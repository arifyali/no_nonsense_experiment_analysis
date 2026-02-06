"""
Visualization utility functions for experimental analysis.

This module provides plotting utilities for distributions, group comparisons,
and analysis result summaries.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Union, Tuple, Any, Literal
from ..core.models import MethodResult


class VisualizationTools:
    """Plotting utilities for experimental analysis.

    This class provides static methods for creating visualizations commonly
    used in experimental data analysis, including distributions, comparisons,
    and result summaries.
    """

    # Default color palette
    DEFAULT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    @staticmethod
    def plot_distribution(
        data: Union[pd.Series, np.ndarray, List[float]],
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        plot_type: Literal["histogram", "kde", "both"] = "both",
        bins: Union[int, str] = "auto",
        color: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
        show_stats: bool = True,
        ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
        """Plot distribution of a data series.

        Args:
            data: The data to plot
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            plot_type: Type of plot ("histogram", "kde", or "both")
            bins: Number of bins for histogram (or "auto")
            color: Color for the plot
            figsize: Figure size as (width, height)
            show_stats: Whether to show descriptive statistics
            ax: Optional existing axes to plot on

        Returns:
            The matplotlib Figure object
        """
        # Convert to numpy array
        arr = np.asarray(data, dtype=float)
        arr = arr[~np.isnan(arr)]

        if len(arr) == 0:
            raise ValueError("Data must contain at least one non-null value")

        # Create figure if no axes provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        color = color or VisualizationTools.DEFAULT_COLORS[0]

        if plot_type in ["histogram", "both"]:
            ax.hist(arr, bins=bins, density=True, alpha=0.7, color=color,
                    edgecolor='white', label='Histogram')

        if plot_type in ["kde", "both"]:
            from scipy import stats
            kde = stats.gaussian_kde(arr)
            x_range = np.linspace(arr.min(), arr.max(), 200)
            ax.plot(x_range, kde(x_range), color=color, linewidth=2, label='KDE')

        # Add statistics annotation
        if show_stats:
            stats_text = (
                f"n = {len(arr)}\n"
                f"Mean = {np.mean(arr):.3f}\n"
                f"Std = {np.std(arr):.3f}\n"
                f"Median = {np.median(arr):.3f}"
            )
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=9)

        ax.set_title(title or "Distribution")
        ax.set_xlabel(xlabel or "Value")
        ax.set_ylabel(ylabel or "Density")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_comparison(
        groups: Dict[str, Union[pd.Series, np.ndarray, List[float]]],
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        plot_type: Literal["box", "violin", "bar", "strip"] = "box",
        colors: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 6),
        show_means: bool = True,
        ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
        """Plot comparison between groups.

        Args:
            groups: Dictionary mapping group names to data
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            plot_type: Type of comparison plot
            colors: List of colors for groups
            figsize: Figure size
            show_means: Whether to show mean markers
            ax: Optional existing axes

        Returns:
            The matplotlib Figure object
        """
        if len(groups) == 0:
            raise ValueError("groups cannot be empty")

        # Create figure if no axes provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        colors = colors or VisualizationTools.DEFAULT_COLORS

        group_names = list(groups.keys())
        group_data = [np.asarray(groups[name], dtype=float) for name in group_names]
        group_data = [arr[~np.isnan(arr)] for arr in group_data]

        positions = np.arange(len(group_names))

        if plot_type == "box":
            bp = ax.boxplot(group_data, positions=positions, patch_artist=True,
                           widths=0.6, labels=group_names)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            if show_means:
                means = [np.mean(d) for d in group_data]
                ax.scatter(positions, means, marker='D', color='red',
                          s=50, zorder=3, label='Mean')
                ax.legend()

        elif plot_type == "violin":
            parts = ax.violinplot(group_data, positions=positions, widths=0.7,
                                  showmeans=show_means, showmedians=True)
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i % len(colors)])
                pc.set_alpha(0.7)
            ax.set_xticks(positions)
            ax.set_xticklabels(group_names)

        elif plot_type == "bar":
            means = [np.mean(d) for d in group_data]
            stds = [np.std(d) for d in group_data]
            bars = ax.bar(positions, means, yerr=stds, capsize=5,
                         color=[colors[i % len(colors)] for i in range(len(group_names))],
                         alpha=0.7, edgecolor='white')
            ax.set_xticks(positions)
            ax.set_xticklabels(group_names)

        elif plot_type == "strip":
            for i, (name, data) in enumerate(zip(group_names, group_data)):
                x = np.random.normal(i, 0.08, size=len(data))
                ax.scatter(x, data, alpha=0.5, color=colors[i % len(colors)],
                          s=30, label=name)
            if show_means:
                means = [np.mean(d) for d in group_data]
                ax.scatter(positions, means, marker='D', color='red',
                          s=80, zorder=3, edgecolor='white', linewidth=2)
            ax.set_xticks(positions)
            ax.set_xticklabels(group_names)

        ax.set_title(title or "Group Comparison")
        ax.set_xlabel(xlabel or "Group")
        ax.set_ylabel(ylabel or "Value")
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_results_summary(
        results: List[MethodResult],
        plot_type: Literal["p_values", "effect_sizes", "combined"] = "combined",
        alpha: float = 0.05,
        figsize: Tuple[int, int] = (12, 6),
        title: Optional[str] = None
    ) -> plt.Figure:
        """Plot summary of analysis results.

        Args:
            results: List of MethodResult objects
            plot_type: Type of summary plot:
                - "p_values": Bar chart of p-values with significance threshold
                - "effect_sizes": Bar chart of effect sizes
                - "combined": Both p-values and effect sizes side by side
            alpha: Significance threshold for p-values
            figsize: Figure size
            title: Overall title

        Returns:
            The matplotlib Figure object
        """
        if len(results) == 0:
            raise ValueError("results cannot be empty")

        if plot_type == "combined":
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            VisualizationTools._plot_p_values(results, alpha, ax1)
            VisualizationTools._plot_effect_sizes(results, ax2)
        elif plot_type == "p_values":
            fig, ax = plt.subplots(figsize=figsize)
            VisualizationTools._plot_p_values(results, alpha, ax)
        elif plot_type == "effect_sizes":
            fig, ax = plt.subplots(figsize=figsize)
            VisualizationTools._plot_effect_sizes(results, ax)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")

        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')

        plt.tight_layout()
        return fig

    @staticmethod
    def _plot_p_values(results: List[MethodResult], alpha: float, ax: plt.Axes) -> None:
        """Helper to plot p-values from results."""
        method_names = []
        p_values = []

        for result in results:
            for test_name, p_val in result.p_values.items():
                if isinstance(p_val, (int, float)) and not np.isnan(p_val):
                    method_names.append(f"{result.method_name}\n({test_name})")
                    p_values.append(p_val)

        if not p_values:
            ax.text(0.5, 0.5, "No p-values available", ha='center', va='center',
                   transform=ax.transAxes)
            return

        positions = np.arange(len(method_names))
        colors = ['#2ca02c' if p < alpha else '#d62728' for p in p_values]

        ax.barh(positions, p_values, color=colors, alpha=0.7, edgecolor='white')
        ax.axvline(x=alpha, color='red', linestyle='--', linewidth=2, label=f'α = {alpha}')
        ax.set_yticks(positions)
        ax.set_yticklabels(method_names)
        ax.set_xlabel('P-value')
        ax.set_title('P-values by Test')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')

        # Add significance annotation
        for i, (p, name) in enumerate(zip(p_values, method_names)):
            sig_text = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            ax.text(p + 0.01, i, sig_text, va='center', fontsize=9)

    @staticmethod
    def _plot_effect_sizes(results: List[MethodResult], ax: plt.Axes) -> None:
        """Helper to plot effect sizes from results."""
        method_names = []
        effect_sizes = []

        for result in results:
            for effect_name, effect_val in result.effect_sizes.items():
                if isinstance(effect_val, (int, float)) and not np.isnan(effect_val):
                    method_names.append(f"{result.method_name}\n({effect_name})")
                    effect_sizes.append(effect_val)

        if not effect_sizes:
            ax.text(0.5, 0.5, "No effect sizes available", ha='center', va='center',
                   transform=ax.transAxes)
            return

        positions = np.arange(len(method_names))
        colors = [VisualizationTools.DEFAULT_COLORS[i % len(VisualizationTools.DEFAULT_COLORS)]
                  for i in range(len(method_names))]

        ax.barh(positions, effect_sizes, color=colors, alpha=0.7, edgecolor='white')
        ax.set_yticks(positions)
        ax.set_yticklabels(method_names)
        ax.set_xlabel('Effect Size')
        ax.set_title('Effect Sizes by Test')
        ax.grid(True, alpha=0.3, axis='x')

        # Add Cohen's d thresholds
        ax.axvline(x=0.2, color='gray', linestyle=':', alpha=0.5, label='Small (0.2)')
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Medium (0.5)')
        ax.axvline(x=0.8, color='gray', linestyle='-', alpha=0.5, label='Large (0.8)')
        ax.legend(loc='lower right', fontsize=8)

    @staticmethod
    def plot_before_after(
        before: Union[pd.Series, np.ndarray, List[float]],
        after: Union[pd.Series, np.ndarray, List[float]],
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
        show_change: bool = True
    ) -> plt.Figure:
        """Plot paired before-after comparison.

        Args:
            before: Data before intervention
            after: Data after intervention
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size
            show_change: Whether to show individual change lines

        Returns:
            The matplotlib Figure object
        """
        before_arr = np.asarray(before, dtype=float)
        after_arr = np.asarray(after, dtype=float)

        if len(before_arr) != len(after_arr):
            raise ValueError("Before and after arrays must have the same length")

        fig, ax = plt.subplots(figsize=figsize)

        positions = np.array([0, 1])

        # Plot individual lines if requested
        if show_change:
            for b, a in zip(before_arr, after_arr):
                color = 'green' if a > b else 'red' if a < b else 'gray'
                ax.plot(positions, [b, a], color=color, alpha=0.3, linewidth=1)

        # Plot points
        ax.scatter(np.zeros_like(before_arr), before_arr,
                   color=VisualizationTools.DEFAULT_COLORS[0], alpha=0.6,
                   s=50, label='Before')
        ax.scatter(np.ones_like(after_arr), after_arr,
                   color=VisualizationTools.DEFAULT_COLORS[1], alpha=0.6,
                   s=50, label='After')

        # Plot means
        ax.scatter([0], [np.mean(before_arr)], color=VisualizationTools.DEFAULT_COLORS[0],
                   marker='D', s=150, edgecolor='white', linewidth=2, zorder=5)
        ax.scatter([1], [np.mean(after_arr)], color=VisualizationTools.DEFAULT_COLORS[1],
                   marker='D', s=150, edgecolor='white', linewidth=2, zorder=5)

        ax.set_xticks(positions)
        ax.set_xticklabels(['Before', 'After'])
        ax.set_title(title or "Before-After Comparison")
        ax.set_xlabel(xlabel or "")
        ax.set_ylabel(ylabel or "Value")
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Add change statistics
        change = after_arr - before_arr
        stats_text = (
            f"Mean change: {np.mean(change):+.3f}\n"
            f"Median change: {np.median(change):+.3f}\n"
            f"% improved: {100 * np.mean(change > 0):.1f}%"
        )
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=9)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_correlation_matrix(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: Literal["pearson", "spearman", "kendall"] = "pearson",
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = "RdBu_r",
        annotate: bool = True
    ) -> plt.Figure:
        """Plot correlation matrix heatmap.

        Args:
            df: DataFrame to analyze
            columns: Columns to include (defaults to all numeric)
            method: Correlation method
            title: Plot title
            figsize: Figure size
            cmap: Colormap name
            annotate: Whether to show correlation values

        Returns:
            The matplotlib Figure object
        """
        # Select numeric columns
        if columns:
            numeric_df = df[columns].select_dtypes(include=[np.number])
        else:
            numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            raise ValueError("No numeric columns found")

        corr_matrix = numeric_df.corr(method=method)

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1, aspect='auto')

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Correlation')

        # Set ticks
        ax.set_xticks(np.arange(len(corr_matrix.columns)))
        ax.set_yticks(np.arange(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr_matrix.columns)

        # Add annotations
        if annotate:
            for i in range(len(corr_matrix)):
                for j in range(len(corr_matrix)):
                    value = corr_matrix.iloc[i, j]
                    text_color = 'white' if abs(value) > 0.5 else 'black'
                    ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                           color=text_color, fontsize=8)

        ax.set_title(title or f"Correlation Matrix ({method.title()})")

        plt.tight_layout()
        return fig

    @staticmethod
    def save_figure(
        fig: plt.Figure,
        filepath: str,
        dpi: int = 150,
        format: Optional[str] = None,
        transparent: bool = False
    ) -> None:
        """Save a figure to file.

        Args:
            fig: The figure to save
            filepath: Path to save the figure
            dpi: Resolution in dots per inch
            format: Output format (inferred from filepath if not specified)
            transparent: Whether to use transparent background
        """
        fig.savefig(filepath, dpi=dpi, format=format, transparent=transparent,
                    bbox_inches='tight', facecolor='white' if not transparent else 'none')
