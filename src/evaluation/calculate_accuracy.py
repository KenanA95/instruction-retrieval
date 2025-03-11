import pandas as pd
import re
import json
import os

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

def load_results(results_file):
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    return pd.DataFrame(data)

def calculate_accuracy(df, answer_col='answer', raw_response_col='raw_response', extracted_answer_col=None):
    """
    Calculate accuracy at three levels:
    1. Exact match between extracted answer and reference answer
    2. Reference answer contained in extracted answer
    3. Reference answer contained in raw response
    
    Args:
        df: DataFrame containing the data
        answer_col: Column name for reference answers
        raw_response_col: Column name for raw model responses
        extracted_answer_col: Column name for extracted answers (if available)
        
    Returns:
        dict: Dictionary with accuracy metrics
        DataFrame: Original DataFrame with added accuracy columns
    """
    result_df = df.copy()
    results = {}
    
    # Level 3: Answer contained in raw response
    result_df['answer_in_raw'] = result_df.apply(
        lambda row: contains_answer(row[raw_response_col], row[answer_col]), 
        axis=1
    )
    results['answer_in_raw'] = result_df['answer_in_raw'].mean()
    
    # Check if extracted answer column exists
    if extracted_answer_col and extracted_answer_col in result_df.columns:
        # Level 1: Exact match between extracted answer and reference answer
        result_df['exact_match'] = result_df.apply(
            lambda row: exact_match(row[extracted_answer_col], row[answer_col]), 
            axis=1
        )
        results['exact_match'] = result_df['exact_match'].mean()
        
        # Level 2: Answer contained in extracted answer
        result_df['answer_in_extracted'] = result_df.apply(
            lambda row: contains_answer(row[extracted_answer_col], row[answer_col]), 
            axis=1
        )
        results['answer_in_extracted'] = result_df['answer_in_extracted'].mean()
    else:
        results['exact_match'] = None
        results['answer_in_extracted'] = None
    
    return results, result_df

def calculate_topic_accuracies(df, topic_col='topic'):
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

def calculate_variant_accuracies(results_file):
    df = load_results(results_file)
    
    # Find all variant columns
    variant_prefixes = set()
    for col in df.columns:
        if col.endswith('_raw_response') or col.endswith('_extracted_answer'):
            prefix = col.rsplit('_', 2)[0]
            variant_prefixes.add(prefix)
    
    # Calculate accuracy for each variant
    variant_results = {}
    for prefix in variant_prefixes:
        raw_col = f"{prefix}_raw_response"
        extracted_col = f"{prefix}_extracted_answer"
        
        if raw_col in df.columns and extracted_col in df.columns:
            results, _ = calculate_accuracy(df, 'answer', raw_col, extracted_col)
            variant_results[prefix] = results
    
    # Create a summary DataFrame
    summary_data = []
    for variant, metrics in variant_results.items():
        row = {'variant': variant}
        row.update(metrics)
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    return variant_results, summary_df
