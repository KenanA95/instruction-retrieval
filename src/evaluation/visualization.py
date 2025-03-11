import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def set_plotting_style():
    """Set the plotting style for consistent visualizations."""
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12

def plot_variant_comparison(summary_df, metric='exact_match', title=None, output_file=None):
    """Plot a comparison of variants for a specific metric."""
    set_plotting_style()
    
    # Sort by the metric
    sorted_df = summary_df.sort_values(by=metric, ascending=False)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='variant', y=metric, data=sorted_df)
    
    # Add value labels on top of bars
    for i, v in enumerate(sorted_df[metric]):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # Set labels and title
    plt.xlabel('Variant')
    plt.ylabel(f'{metric.replace("_", " ").title()} Accuracy')
    plt.title(title or f'Comparison of Variants by {metric.replace("_", " ").title()} Accuracy')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output_file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def plot_metrics_by_topic(topic_metrics, metric='exact_match', top_n=10, title=None, output_file=None):
    """Plot metrics by topic, showing the top N topics by sample count."""
    set_plotting_style()
    
    # Sort by count and take top N
    sorted_df = topic_metrics.sort_values(by='count', ascending=False).head(top_n)
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x=sorted_df.index, y=metric, data=sorted_df)
    
    # Add value labels on top of bars
    for i, v in enumerate(sorted_df[metric]):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # Set labels and title
    plt.xlabel('Topic')
    plt.ylabel(f'{metric.replace("_", " ").title()} Accuracy')
    plt.title(title or f'Top {top_n} Topics by {metric.replace("_", " ").title()} Accuracy')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output_file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def plot_all_metrics(summary_df, output_dir=None, prefix=None):
    """Plot all metrics for variant comparison."""
    metrics = ['exact_match', 'answer_in_extracted', 'answer_in_raw']
    
    figures = {}
    for metric in metrics:
        if metric in summary_df.columns:
            title = f'Comparison of Variants by {metric.replace("_", " ").title()} Accuracy'
            
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f'{prefix or ""}_{metric}.png')
            else:
                output_file = None
            
            fig = plot_variant_comparison(
                summary_df, 
                metric=metric, 
                title=title,
                output_file=output_file
            )
            
            figures[metric] = fig
    
    return figures

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