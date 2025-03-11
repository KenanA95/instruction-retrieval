import os
import json
import pandas as pd
from pathlib import Path

from ..utils.io import load_json, save_json

def evaluate_variants(results_file):
    """
    Evaluate model outputs for MedQA multiple-choice tasks.
    
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
    
    # Check if required columns exist
    if 'answer' not in df.columns or 'extracted_answer' not in df.columns:
        raise ValueError("Results must contain 'answer' and 'extracted_answer' columns")
    
    # Calculate exact match accuracy
    df['correct'] = df.apply(lambda row: 
        row['extracted_answer'].strip().upper() == row['answer'].strip().upper() 
        if pd.notna(row['extracted_answer']) else False, 
        axis=1
    )
    
    # Calculate overall accuracy
    accuracy = df['correct'].mean()
    
    # If topics exist, calculate per-topic accuracy
    topic_accuracies = {}
    if 'topic' in df.columns:
        topic_groups = df.groupby('topic')
        for topic, group in topic_groups:
            topic_accuracies[topic] = group['correct'].mean()
    
    # Create summary dataframe
    if 'variant' in df.columns:
        # If variants exist, group by variant
        summary_df = df.groupby('variant')['correct'].mean().reset_index()
        summary_df.rename(columns={'correct': 'accuracy'}, inplace=True)
    else:
        # Otherwise, create a single row summary
        summary_df = pd.DataFrame([{'accuracy': accuracy}])
    
    # Add topic breakdown if topics exist
    if topic_accuracies:
        topic_df = pd.DataFrame([(topic, acc) for topic, acc in topic_accuracies.items()],
                                columns=['topic', 'accuracy'])
        topic_df = topic_df.sort_values('accuracy', ascending=False)
    else:
        topic_df = None
    
    # Create metrics dict
    metrics = {
        'accuracy': accuracy,
        'topic_accuracies': topic_accuracies
    }
    
    return metrics, summary_df

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
            
        # Get variant name from filename
        variant_name = json_file.stem  # filename without extension
        
        # Evaluate this variant
        metrics, _ = evaluate_variants(results)
        
        # Store results
        all_results.append({
            'variant': variant_name,
            'accuracy': metrics['accuracy']
        })
    
    # Convert to DataFrame
    summary_df = pd.DataFrame(all_results)
    summary_df = summary_df.sort_values('accuracy', ascending=False)
    
    return summary_df

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate MedQA model outputs")
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
    print(summary_df) 