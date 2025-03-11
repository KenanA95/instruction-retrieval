import argparse
import pandas as pd
from tqdm import tqdm
import os
import sys
from llama_inference import LlamaInference
from extract_answer import extract_answer, load_llama_model

def validate_dataframe(df):
    required_columns = ['prompt', 'question', 'answer']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError("Input file missing required columns: " + ", ".join(missing))
    return True

def get_clean_model_name(model_name):
    return model_name.split('/')[-1].replace('-', '_')

def process_in_batches(data_list, process_func, batch_size, desc):
    results = []
    for i in tqdm(range(0, len(data_list), batch_size), desc=desc):
        batch = data_list[i: i + batch_size]
        results.extend(process_func(batch))
    return results

def process_file(input_file, model_name, extract_model_name, output_dir, batch_size, experiment_name):
    print(f"Loading data from {input_file}")
    file_ext = os.path.splitext(input_file)[1].lower()
    if file_ext == '.csv':
        df = pd.read_csv(input_file)
    elif file_ext == '.tsv':
        df = pd.read_csv(input_file, sep='\t')
    elif file_ext == '.json':
        df = pd.read_json(input_file, orient='records')
    else:
        raise ValueError(f"Unsupported file extension: {file_ext}. Use .csv, .tsv, or .json")
    
    validate_dataframe(df)
    print("Data validation successful")
    
    clean_model_name = get_clean_model_name(model_name)
    output_file = os.path.join(output_dir, f"{experiment_name}_{clean_model_name}.json")
    
    # Initialize and run the inference model in batches
    llm = LlamaInference(model_name=model_name)
    print(f"Initialized inference model: {model_name}")
    
    # Batch process prompts for raw responses
    prompts = df['prompt'].tolist()
    print("Generating responses for each prompt...")
    raw_responses = process_in_batches(
        prompts,
        lambda batch: llm.generate_responses_batch(batch),
        batch_size,
        "Generating responses"
    )
    df['raw_response'] = raw_responses
    
    intermediate_file = os.path.join(output_dir, f"{experiment_name}_{clean_model_name}_raw.json")
    df.to_json(intermediate_file, orient='records', indent=2)
    print(f"Saved raw responses to {intermediate_file}")
    
    # Load extraction model and extract answers in batches
    print(f"Loading model for answer extraction: {extract_model_name}")
    extract_model, extract_tokenizer = load_llama_model(model_name=extract_model_name)
    
    responses = df['raw_response'].tolist()
    print("Extracting answers from responses...")
    extracted_answers = process_in_batches(
        responses,
        lambda batch: [extract_answer(extract_model, extract_tokenizer, resp) for resp in batch],
        batch_size,
        "Extracting answers"
    )
    df['extracted_answer'] = extracted_answers
    
    df.to_json(output_file, orient='records', indent=2)
    print(f"Saved final results to {output_file}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Run inference on pre-generated prompts and extract answers")
    parser.add_argument("--input", "-i", required=True, help="Input CSV/TSV file with prompts and answers")
    parser.add_argument("--model", "-m", default="meta-llama/Llama-3.2-1B-Instruct",
                        help="Model name to use for inference")
    parser.add_argument("--extract-model", "-e", default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Model name to use for answer extraction")
    parser.add_argument("--output-dir", "-o", default="/shared/3/projects/instruction-retrieval/results/",
                        help="Output directory for results")
    parser.add_argument("--batch-size", "-b", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--experiment-name", "-n", default="experiment",
                        help="Name of the experiment (used in output filenames)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        process_file(args.input, args.model, args.extract_model, args.output_dir, args.batch_size, args.experiment_name)
        print("Processing completed successfully")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()