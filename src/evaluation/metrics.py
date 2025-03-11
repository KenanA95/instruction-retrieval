import os
import pandas as pd
import re
import json
from ..utils.io import load_json, save_json

def normalize_answer(answer):
    """Normalize answer string for comparison."""
    if pd.isna(answer):
        return ""
    
    # Convert to string
    answer = str(answer).strip()
    
    # Remove spaces between numbers and operators
    answer = re.sub(r'(\d)\s+([+\-*/])\s+(\d)', r'\1\2\3', answer)
    
    # Replace multiple spaces with single space
    answer = re.sub(r'\s+', ' ', answer)
    
    return answer

def exact_match(predicted, reference):
    """Check if predicted answer exactly matches reference answer."""
    if pd.isna(predicted) or pd.isna(reference):
        return False
    
    predicted = normalize_answer(predicted)
    reference = normalize_answer(reference)
    
    return predicted == reference

def contains_answer(text, answer):
    """Check if text contains the answer."""
    if pd.isna(text) or pd.isna(answer):
        return False
    
    text = normalize_answer(text)
    answer = normalize_answer(answer)
    
    return answer in text

def calculate_accuracy(results, answer_col='answer', raw_response_col='raw_response', extracted_answer_col=None):
    """
    Calculate accuracy at three levels:
    1. Exact match between extracted answer and reference answer
    2. Reference answer contained in extracted answer
    3. Reference answer contained in raw response
    
    Args:
        results: List of result dictionaries or DataFrame
        answer_col: Column/key name for reference answers
        raw_response_col: Column/key name for raw model responses
        extracted_answer_col: Column/key name for extracted answers (if available)
        
    Returns:
        dict: Dictionary with accuracy metrics
        DataFrame: DataFrame with added accuracy columns
    """
    # Convert to DataFrame if not already
    if not isinstance(results, pd.DataFrame):
        df = pd.DataFrame(results)
    else:
        df = results.copy()
    
    metrics = {}
    
    # Level 3: Answer contained in raw response
    df['answer_in_raw'] = df.apply(
        lambda row: contains_answer(row[raw_response_col], row[answer_col]), 
        axis=1
    )
    metrics['answer_in_raw'] = df['answer_in_raw'].mean()
    
    # Check if extracted answer column exists
    if extracted_answer_col and extracted_answer_col in df.columns:
        # Level 1: Exact match between extracted answer and reference answer
        df['exact_match'] = df.apply(
            lambda row: exact_match(row[extracted_answer_col], row[answer_col]), 
            axis=1
        )
        metrics['exact_match'] = df['exact_match'].mean()
        
        # Level 2: Answer contained in extracted answer
        df['answer_in_extracted'] = df.apply(
            lambda row: contains_answer(row[extracted_answer_col], row[answer_col]), 
            axis=1
        )
        metrics['answer_in_extracted'] = df['answer_in_extracted'].mean()
    else:
        metrics['exact_match'] = None
        metrics['answer_in_extracted'] = None
    
    return metrics, df

def calculate_topic_accuracies(results, topic_col='topic'):
    """Calculate accuracy metrics by topic."""
    if not isinstance(results, pd.DataFrame):
        df = pd.DataFrame(results)
    else:
        df = results.copy()
    
    agg_dict = {topic_col: 'count'}
    
    if 'exact_match' in df.columns:
        agg_dict['exact_match'] = 'mean'
    
    if 'answer_in_extracted' in df.columns:
        agg_dict['answer_in_extracted'] = 'mean'
    
    if 'answer_in_raw' in df.columns:
        agg_dict['answer_in_raw'] = 'mean'
    
    topic_accuracies = df.groupby(topic_col).agg(agg_dict)
    topic_accuracies = topic_accuracies.rename(columns={topic_col: 'count'})
    
    return topic_accuracies

def evaluate_variants(results_file):
    """Evaluate all variants in a combined results file or individual variant files."""
    # Load results
    if isinstance(results_file, str):
        # Check if file exists
        if not os.path.exists(results_file):
            print(f"Combined results file not found: {results_file}")
            print("Looking for individual variant files instead...")
            
            # Extract directory and experiment name from the combined file path
            output_dir = os.path.dirname(results_file)
            experiment_name = os.path.basename(results_file).split('_all_variants_')[0]
            model_name = os.path.basename(results_file).split('_')[-1].replace('.json', '')
            
            # Find all individual variant result files
            variant_files = []
            for file in os.listdir(output_dir):
                if file.endswith('.json') and file != os.path.basename(results_file):
                    variant_files.append(os.path.join(output_dir, file))
            
            if not variant_files:
                raise FileNotFoundError(f"No variant result files found in {output_dir}")
            
            print(f"Found {len(variant_files)} variant result files")
            
            # Load each variant file and combine results
            combined_results = []
            variant_data = {}
            
            for variant_file in variant_files:
                try:
                    variant_name = os.path.basename(variant_file).split('_')[0]
                    print(f"Loading variant: {variant_name}")
                    
                    variant_results = load_json(variant_file)
                    variant_data[variant_name] = variant_results
                    
                    # Add results to combined data
                    for item in variant_results:
                        # Find or create a row for this topic-question pair
                        existing = next((r for r in combined_results 
                                         if r.get('topic') == item.get('topic') and 
                                            r.get('question') == item.get('question')), None)
                        
                        if existing is None:
                            existing = {
                                'topic': item.get('topic'),
                                'question': item.get('question'),
                                'answer': item.get('answer')
                            }
                            combined_results.append(existing)
                        
                        # Add variant-specific columns
                        existing[f'{variant_name}_raw_response'] = item.get('raw_response', '')
                        existing[f'{variant_name}_extracted_answer'] = item.get('extracted_answer', '')
                except Exception as e:
                    print(f"Error loading variant file {variant_file}: {e}")
            
            # Save the combined results
            if combined_results:
                save_json(combined_results, results_file)
                print(f"Created combined results file: {results_file}")
                results = combined_results
            else:
                raise ValueError("Failed to create combined results from variant files")
        else:
            # Load the existing combined results file
            results = load_json(results_file)
    else:
        # Results already provided as a data structure
        results = results_file
    
    # Rest of the function remains the same
    df = pd.DataFrame(results)
    
    # Find all variant columns
    variant_prefixes = set()
    for col in df.columns:
        if col.endswith('_raw_response') or col.endswith('_extracted_answer'):
            prefix = col.rsplit('_', 2)[0]
            variant_prefixes.add(prefix)
    
    print(f"Found {len(variant_prefixes)} variants in results: {', '.join(variant_prefixes)}")
    
    # Calculate accuracy for each variant
    variant_results = {}
    for prefix in variant_prefixes:
        raw_col = f"{prefix}_raw_response"
        extracted_col = f"{prefix}_extracted_answer"
        
        if raw_col in df.columns and extracted_col in df.columns:
            print(f"Calculating metrics for variant: {prefix}")
            metrics, _ = calculate_accuracy(df, 'answer', raw_col, extracted_col)
            variant_results[prefix] = metrics
    
    # Create a summary DataFrame
    summary_data = []
    for variant, metrics in variant_results.items():
        row = {'variant': variant}
        row.update(metrics)
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    return variant_results, summary_df

def save_evaluation(metrics, topic_metrics, output_dir, experiment_name):
    """Save evaluation metrics to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save overall metrics
    metrics_file = os.path.join(output_dir, f"{experiment_name}_metrics.json")
    save_json(metrics, metrics_file)
    
    # Save topic metrics
    topic_file = os.path.join(output_dir, f"{experiment_name}_topic_metrics.csv")
    topic_metrics.to_csv(topic_file)
    
    return metrics_file, topic_file 