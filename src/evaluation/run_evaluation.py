import os
import json
import pandas as pd
from pathlib import Path
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.evaluation.metrics import calculate_accuracy
from src.utils.io import load_json, save_json

def evaluate_all_variants(results_dir):
    """Evaluate all variant JSON files in the results directory."""
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
    
    # Save to CSV
    output_file = os.path.join(results_dir, 'variant_evaluation_results.csv')
    summary_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    return summary_df

if __name__ == "__main__":
    # Run the evaluation
    results_dir = "/home/kalkiek/projects/instruction-retrieval/results/math_instruction_retrieval/inference_results"
    summary_df = evaluate_all_variants(results_dir)

    # Print summary statistics
    print("\nSummary by variant:")
    variant_summary = summary_df.groupby('variant').mean(numeric_only=True)
    print(variant_summary)
