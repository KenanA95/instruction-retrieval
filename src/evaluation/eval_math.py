import os
import json
import pandas as pd
from pathlib import Path

from ..utils.io import load_json, save_json
from .metrics import calculate_accuracy

def evaluate_variants(results_file):
    """
    Evaluate model outputs for math tasks.
    
    Args:
        results_file: Path to JSON file with model outputs
        
    Returns:
        Tuple of (variant_metrics, summary_df)
    """
    # Load results
    if isinstance(results_file, str) or isinstance(results_file, Path):
        results = load_json(results_file)
    else:
        # Assume results_file is already the loaded results
        results = results_file
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Calculate metrics
    metrics, df_with_metrics = calculate_accuracy(
        df, 
        answer_col='answer',
        raw_response_col='raw_response',
        extracted_answer_col='extracted_answer'
    )
    
    # Return the metrics
    return metrics, df_with_metrics

def evaluate_all_variants(results_dir):
    """
    Evaluate all variant JSON files in the results directory.
    
    Args:
        results_dir: Directory containing JSON result files
        
    Returns:
        DataFrame with evaluation summary
    """
    # Get all JSON files
    json_files = list(Path(results_dir).glob('*.json'))
    
    # Store all results
    all_results = []
    
    for json_file in json_files:
        # Skip the combined results file if it exists
        if 'all_variants' in json_file.name:
            continue
            
        print(f"Processing {json_file.name}")
        
        # Load the JSON file
        results = load_json(json_file)
            
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Calculate metrics
        metrics, df_with_metrics = calculate_accuracy(
            df, 
            answer_col='answer',
            raw_response_col='raw_response',
            extracted_answer_col='extracted_answer'
        )
        
        # Add metrics to each row along with variant name
        variant_name = json_file.stem  # filename without extension
        for _, row in df_with_metrics.iterrows():
            all_results.append({
                'variant': variant_name,
                'topic': row['topic'],
                'exact_match': row.get('exact_match', False),
                'answer_in_extracted': row.get('answer_in_extracted', False),
                'answer_in_raw': row.get('answer_in_raw', False)
            })
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(all_results)
    
    # Calculate averages by variant and topic
    summary_df = results_df.groupby(['variant', 'topic']).agg({
        'exact_match': 'mean',
        'answer_in_extracted': 'mean',
        'answer_in_raw': 'mean'
    }).reset_index()
    
    return summary_df

def calculate_topic_accuracies(results_df):
    """
    Calculate topic-level accuracy metrics.
    
    Args:
        results_df: DataFrame with evaluation results
        
    Returns:
        DataFrame with topic-level metrics
    """
    # Group by topic and calculate mean for each metric
    topic_df = results_df.groupby('topic').agg({
        'exact_match': 'mean',
        'answer_in_extracted': 'mean',
        'answer_in_raw': 'mean'
    }).reset_index()
    
    # Sort by exact match accuracy
    topic_df = topic_df.sort_values('exact_match', ascending=False)
    
    return topic_df

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate math model outputs")
    parser.add_argument("--results_dir", required=True, help="Directory containing JSON result files")
    parser.add_argument("--output_file", default=None, help="Output CSV file path")
    args = parser.parse_args()
    
    # Run the evaluation
    results_dir = args.results_dir
    summary_df = evaluate_all_variants(results_dir)
    
    # Save to CSV if output file specified
    if args.output_file:
        summary_df.to_csv(args.output_file, index=False)
        print(f"Results saved to {args.output_file}")
    
    # Print summary statistics
    print("\nSummary by variant:")
    variant_summary = summary_df.groupby('variant').mean(numeric_only=True)
    print(variant_summary) 