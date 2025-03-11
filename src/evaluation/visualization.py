import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from pathlib import Path

from ..utils.io import ensure_dir

def set_plotting_style():
    """Set the plotting style for consistent visualizations."""
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12

def plot_metrics(df, metric='accuracy', output_dir=None, prefix=''):
    """
    Plot metrics for model variants.
    
    Args:
        df: DataFrame with metrics
        metric: Column name for metric to plot
        output_dir: Directory to save plots
        prefix: Prefix for output filenames
    """
    if output_dir:
        ensure_dir(output_dir)
    
    # Set up the plot
    plt.figure(figsize=(12, 6))
    
    # Create barplot of accuracy by variant
    if 'variant' in df.columns:
        # Sort by the metric value
        plot_df = df.sort_values(metric, ascending=False)
        
        # Plot
        ax = sns.barplot(x='variant', y=metric, data=plot_df)
        
        # Add value labels on top of bars
        for i, v in enumerate(plot_df[metric]):
            ax.text(i, v + 0.01, f"{v:.2f}", ha='center')
        
        # Customize appearance
        plt.title(f"{metric.replace('_', ' ').title()} by Variant")
        plt.ylabel(metric.replace('_', ' ').title())
        plt.xlabel('Variant')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot
        if output_dir:
            output_file = os.path.join(output_dir, f"{prefix}_{metric}_by_variant.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_file}")
    
    # Close plot to avoid memory issues
    plt.close()

def plot_metrics_by_topic(df, metric='accuracy', top_n=10, title=None, output_file=None):
    """
    Plot metrics broken down by topic/category.
    
    Args:
        df: DataFrame with metrics and topic/category column
        metric: Column name for metric to plot
        top_n: Number of top topics to show
        title: Plot title
        output_file: Path to save plot
    """
    # Determine the category column (topic or category)
    category_col = next((col for col in ['topic', 'category'] if col in df.columns), None)
    if not category_col:
        print("No topic or category column found in DataFrame")
        return
    
    # Sort and take top N
    plot_df = df.sort_values(metric, ascending=False).head(top_n)
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Create barplot
    ax = sns.barplot(x=metric, y=category_col, data=plot_df, orient='h')
    
    # Add value labels
    for i, v in enumerate(plot_df[metric]):
        ax.text(v + 0.01, i, f"{v:.2f}", va='center')
    
    # Customize appearance
    if title:
        plt.title(title)
    else:
        plt.title(f"Top {top_n} by {metric.replace('_', ' ').title()}")
    
    plt.ylabel(category_col.title())
    plt.xlabel(metric.replace('_', ' ').title())
    plt.tight_layout()
    
    # Save plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    # Close plot
    plt.close()

def plot_variant_comparison(summary_df, output_dir=None, prefix=''):
    """
    Create comparison plots for multiple metrics across variants.
    
    Args:
        summary_df: DataFrame with metrics by variant
        output_dir: Directory to save plots
        prefix: Prefix for output filenames
    """
    if output_dir:
        ensure_dir(output_dir)
    
    # Get metric columns (exclude non-numeric columns)
    metric_cols = [col for col in summary_df.columns 
                   if col not in ['variant', 'topic', 'category'] 
                   and pd.api.types.is_numeric_dtype(summary_df[col])]
    
    if not metric_cols:
        print("No metric columns found for plotting")
        return
    
    # Set up a grouped bar chart for all metrics
    plt.figure(figsize=(14, 8))
    
    # Prepare data
    plot_df = summary_df.copy()
    if 'variant' not in plot_df.columns:
        # If no variant column, create a dummy one
        plot_df['variant'] = 'default'
    
    # Sort variants by the first metric
    sorted_variants = plot_df.sort_values(metric_cols[0], ascending=False)['variant'].unique()
    
    # Set up grouped bar positions
    x = np.arange(len(sorted_variants))
    width = 0.8 / len(metric_cols)
    
    # Plot each metric as a group
    for i, metric in enumerate(metric_cols):
        offsets = width * i - width * (len(metric_cols) - 1) / 2
        heights = [plot_df[plot_df['variant'] == variant][metric].mean() for variant in sorted_variants]
        bars = plt.bar(x + offsets, heights, width, label=metric.replace('_', ' ').title())
        
        # Add value labels
        for j, v in enumerate(heights):
            plt.text(x[j] + offsets, v + 0.01, f"{v:.2f}", ha='center', fontsize=8)
    
    # Customize appearance
    plt.title("Comparison of Metrics Across Variants")
    plt.xlabel('Variant')
    plt.ylabel('Score')
    plt.xticks(x, sorted_variants, rotation=45, ha='right')
    plt.legend()
    plt.ylim(0, 1.1)  # Assuming metrics are between 0 and 1
    plt.tight_layout()
    
    # Save plot
    if output_dir:
        output_file = os.path.join(output_dir, f"{prefix}_metric_comparison.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {output_file}")
    
    # Close plot
    plt.close()

def plot_topic_comparison(topic_metrics, variants, metric='exact_match', top_n=10, output_file=None):
    """Plot a comparison of topics across multiple variants."""
    set_plotting_style()
    
    # Prepare data for plotting
    plot_data = []
    for variant, metrics in variants.items():
        for topic, topic_metrics in metrics.iterrows():
            plot_data.append({
                'variant': variant,
                'topic': topic,
                'metric': topic_metrics[metric],
                'count': topic_metrics['count']
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Get top N topics by count
    top_topics = plot_df.groupby('topic')['count'].sum().nlargest(top_n).index.tolist()
    plot_df = plot_df[plot_df['topic'].isin(top_topics)]
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    ax = sns.barplot(x='topic', y='metric', hue='variant', data=plot_df)
    
    # Set labels and title
    plt.xlabel('Topic')
    plt.ylabel(f'{metric.replace("_", " ").title()} Accuracy')
    plt.title(f'Comparison of Variants by Topic ({metric.replace("_", " ").title()} Accuracy)')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output_file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

if __name__ == "__main__":
    # Example usage
    import argparse
    from ..utils.io import load_json
    
    parser = argparse.ArgumentParser(description="Generate visualization for experiment results")
    parser.add_argument("--results_file", required=True, help="Path to results CSV or JSON file")
    parser.add_argument("--output_dir", required=True, help="Directory to save visualizations")
    parser.add_argument("--prefix", default="", help="Prefix for output filenames")
    args = parser.parse_args()
    
    # Load results
    if args.results_file.endswith('.csv'):
        results_df = pd.read_csv(args.results_file)
    else:
        # Assume JSON
        results = load_json(args.results_file)
        results_df = pd.DataFrame(results)
    
    # Create visualizations
    plot_variant_comparison(results_df, args.output_dir, args.prefix)
    
    # Plot individual metrics
    metric_cols = [col for col in results_df.columns 
                   if col not in ['variant', 'topic', 'category'] 
                   and pd.api.types.is_numeric_dtype(results_df[col])]
    
    for metric in metric_cols:
        plot_metrics(results_df, metric, args.output_dir, args.prefix) 