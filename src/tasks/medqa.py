import os
import json
import pandas as pd
from pathlib import Path

from ..data.datasets.medqa import MedQADataset
from ..modeling.prompt_runner import PromptRunner
from ..modeling.inference import InferenceRunner
from ..modeling.retrieval import TextbookRetriever
from ..evaluation.eval_medqa import evaluate_variants
from ..evaluation.visualization import plot_metrics
from ..utils.logging import get_logger
from ..utils.io import ensure_dir, save_json, load_json

logger = get_logger(__name__)

def run_medqa_experiment(config):
    """
    Run the MedQA experiment with the given configuration.
    
    This orchestrates the complete pipeline:
    1. Load the MedQA dataset
    2. Set up retrieval for textbook passages (if RAG is enabled)
    3. Generate prompt variants (zero-shot, RAG, instruction, RAG+instruction)
    4. Run model inference
    5. Evaluate results
    6. Generate visualizations
    
    Args:
        config: Configuration dict with experiment parameters
        
    Returns:
        Dict containing experiment results and metrics
    """
    # Set up experiment
    experiment_name = config.get('experiment_name', 'medqa')
    logger.info(f"Starting MedQA experiment: {experiment_name}")
    
    # Set up directories
    base_dir = config.get('base_dir', '/shared/3/projects/instruction-retrieval/medqa')
    output_dir = config.get('output_dir', 'results/medqa')
    ensure_dir(output_dir)
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset_config = config.get('dataset', {}).copy()
    if 'base_dir' not in dataset_config:
        dataset_config['base_dir'] = base_dir
        
    dataset = MedQADataset(config=dataset_config)
    dataset.load()
    logger.info(f"Loaded {len(dataset)} examples from MedQA")
    
    # Initialize retriever for textbook passages (if needed)
    use_rag = config.get('use_rag', False)
    retriever = None
    if use_rag:
        logger.info("Initializing textbook retriever...")
        retrieval_config = config.get('retrieval', {})
        retriever = TextbookRetriever(
            textbook_path=retrieval_config.get('textbook_path', f"{base_dir}/textbooks"),
            top_k=retrieval_config.get('top_k', 3)
        )
    
    # Set up prompt generation
    prompt_dirs = config.get('prompt_dirs')
    if not prompt_dirs:
        raise ValueError("No prompt directories specified in config")
    
    variants = config.get('variants', list(prompt_dirs.keys()))
    
    # Generate prompts
    logger.info("Generating prompt variants...")
    prompt_runner = PromptRunner(prompt_dirs=prompt_dirs)
    prompt_variants_dir = os.path.join(output_dir, "prompt_variants")
    
    # For each example, retrieve relevant passages if RAG is enabled
    if use_rag:
        for example in dataset.examples:
            # Add retrieved context to examples
            retrieved_passages = retriever.retrieve(example['question'])
            example['context'] = retrieved_passages
    
    # Generate prompts with or without retrieved context
    prompt_runner.generate_prompt_variants(
        dataset=dataset,
        output_dir=prompt_variants_dir,
        variants=variants,
        use_retrieval=use_rag
    )
    
    # Run inference
    logger.info("Running inference...")
    inference_config = config.get('inference', {})
    inference_runner = InferenceRunner(
        model_name=inference_config.get('model_name', 'meta-llama/Llama-3.2-1B-Instruct'),
        batch_size=inference_config.get('batch_size', 8)
    )
    
    inference_results_dir = os.path.join(output_dir, "inference_results")
    results = inference_runner.run_variant_inference(
        variants_dir=prompt_variants_dir,
        output_dir=inference_results_dir,
        experiment_name=experiment_name
    )
    
    # Evaluate results
    logger.info("Evaluating results...")
    combined_results_file = os.path.join(
        inference_results_dir, 
        f"all_variants_{experiment_name}_{inference_runner.get_model_name()}.json"
    )
    
    # Run evaluation - for MedQA, we're checking multiple-choice accuracy
    variant_metrics, summary_df = evaluate_variants(combined_results_file)
    
    # Save evaluation results
    evaluation_dir = os.path.join(output_dir, "evaluation")
    ensure_dir(evaluation_dir)
    
    summary_file = os.path.join(evaluation_dir, f"{experiment_name}_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    visualization_dir = os.path.join(output_dir, "visualizations")
    ensure_dir(visualization_dir)
    
    # Generate variant comparison plots
    plot_metrics(
        summary_df,
        output_dir=visualization_dir,
        prefix=experiment_name
    )
    
    logger.info("MedQA experiment completed successfully")
    return {
        'variant_metrics': variant_metrics,
        'summary_df': summary_df
    }

if __name__ == "__main__":
    # Example of direct usage
    import argparse
    from ..utils.config import load_config
    
    parser = argparse.ArgumentParser(description="Run MedQA experiment")
    parser.add_argument("--config", "-c", required=True, help="Path to experiment configuration file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    run_medqa_experiment(config) 