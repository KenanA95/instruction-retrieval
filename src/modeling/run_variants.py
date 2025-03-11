import os
import argparse
import pandas as pd
import sys
from pathlib import Path
from run_prompts import process_file, get_clean_model_name

def find_variant_files(variants_dir):
    """Find all variant JSON files in the variants directory structure"""
    variant_files = []
    for root, dirs, files in os.walk(variants_dir):
        for file in files:
            if file.endswith('.json') and not file == 'all_variants.json':
                variant_files.append(os.path.join(root, file))
    return variant_files

def main():
    parser = argparse.ArgumentParser(description="Run inference on all prompt variants and extract answers")
    parser.add_argument("--variants-dir", "-v", required=True, 
                        help="Directory containing prompt variant files")
    parser.add_argument("--model", "-m", default="meta-llama/Llama-3.2-1B-Instruct",
                        help="Model name to use for inference")
    parser.add_argument("--extract-model", "-e", default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Model name to use for answer extraction")
    parser.add_argument("--output-dir", "-o", default="/shared/3/projects/instruction-retrieval/results/",
                        help="Output directory for results")
    parser.add_argument("--batch-size", "-b", type=int, default=32, 
                        help="Batch size for processing")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all variant files
    variant_files = find_variant_files(args.variants_dir)
    print(f"Found {len(variant_files)} variant files to process")
    
    # Process each variant file
    results = {}
    for variant_file in variant_files:
        try:
            variant_name = os.path.basename(os.path.dirname(variant_file))
            print(f"\nProcessing variant: {variant_name}")
            
            # Use the process_file function from run_prompts.py
            results[variant_name] = process_file(
                input_file=variant_file, 
                model_name=args.model, 
                extract_model_name=args.extract_model, 
                output_dir=args.output_dir, 
                batch_size=args.batch_size, 
                experiment_name=variant_name
            )
            
            print(f"Completed processing for variant: {variant_name}")
        except Exception as e:
            print(f"Error processing {variant_file}: {e}", file=sys.stderr)
    
    # Create a combined results file
    print("\nCreating combined results file...")
    combined_results = []
    
    # Get a list of all topics and questions
    all_topics_questions = set()
    for df in results.values():
        all_topics_questions.update(zip(df['topic'], df['question']))
    
    # For each topic-question pair, collect results from all variants
    for topic, question in all_topics_questions:
        row = {'topic': topic, 'question': question}
        
        # Find the answer (should be the same across all variants)
        for variant_name, df in results.items():
            match = df[(df['topic'] == topic) & (df['question'] == question)]
            if not match.empty:
                row['answer'] = match.iloc[0]['answer']
                break
        
        # Collect results from each variant
        for variant_name, df in results.items():
            match = df[(df['topic'] == topic) & (df['question'] == question)]
            if not match.empty:
                row[f'{variant_name}_raw_response'] = match.iloc[0]['raw_response']
                row[f'{variant_name}_extracted_answer'] = match.iloc[0]['extracted_answer']
        
        combined_results.append(row)
    
    # Save combined results
    combined_df = pd.DataFrame(combined_results)
    clean_model_name = get_clean_model_name(args.model)
    combined_file = os.path.join(args.output_dir, f"all_variants_{clean_model_name}.json")
    combined_df.to_json(combined_file, orient='records', indent=2)
    print(f"Saved combined results to {combined_file}")
    
    print("\nAll variants processed successfully")

if __name__ == "__main__":
    main()
