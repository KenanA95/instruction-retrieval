import os
import json
import pandas as pd
from pathlib import Path

from ..data.datasets.math import MathDataset
from ..modeling.prompt_runner import PromptRunner
from ..modeling.inference import InferenceRunner
from ..evaluation.eval_math import evaluate_variants, calculate_topic_accuracies
from ..evaluation.visualization import plot_metrics
from ..utils.logging import get_logger
from ..utils.io import ensure_dir, save_json, load_json

logger = get_logger(__name__)

def run_math_experiment(config):
    """
    Run the Math MVP experiment with the given configuration.
    
    This orchestrates the complete pipeline:
    1. Load the math dataset
    2. Generate prompt variants (zero-shot, few-shot, instruction variants)
    3. Run model inference
    4. Evaluate results
    5. Generate visualizations
    
    Args:
        config: Configuration dict with experiment parameters
        
    Returns:
        Dict containing experiment results and metrics
    """
    # Set up experiment
    experiment_name = config.get('experiment_name', 'math_mvp')
    logger.info(f"Starting math experiment: {experiment_name}")
    
    # Set up directories
    base_dir = config.get('base_dir', '/shared/3/projects/instruction-retrieval/mathematics_dataset')
    output_dir = config.get('output_dir', 'results/math')
    ensure_dir(output_dir)
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset_config = config.get('dataset', {}).copy()
    if 'base_dir' not in dataset_config:
        dataset_config['base_dir'] = base_dir
        
    dataset = MathDataset(config=dataset_config)
    dataset.load()
    logger.info(f"Loaded {len(dataset)} examples from {len(dataset.get_topics())} topics")
    
    # Set up prompt generation
    prompt_dirs = config.get('prompt_dirs')
    if not prompt_dirs:
        raise ValueError("No prompt directories specified in config")
    
    variants = config.get('variants', list(prompt_dirs.keys()))
    num_cot_examples = config.get('num_cot_examples', 3)
    
    # Generate prompts
    logger.info("Generating prompt variants...")
    prompt_runner = PromptRunner(prompt_dirs=prompt_dirs)
    prompt_variants_dir = os.path.join(output_dir, "prompt_variants")
    prompt_runner.generate_prompt_variants(
        dataset=dataset,
        output_dir=prompt_variants_dir,
        variants=variants,
        num_cot_examples=num_cot_examples
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
    
    # Run evaluation
    variant_metrics, summary_df = evaluate_variants(combined_results_file)
    
    # Save evaluation results
    evaluation_dir = os.path.join(output_dir, "evaluation")
    ensure_dir(evaluation_dir)
    
    summary_file = os.path.join(evaluation_dir, f"{experiment_name}_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    
    # Calculate and save topic-level metrics
    topic_metrics = {}
    for variant in variants:
        variant_file = os.path.join(inference_results_dir, f"{variant}_{inference_runner.get_model_name()}.json")
        if os.path.exists(variant_file):
            variant_results = load_json(variant_file)
            _, variant_df = evaluate_variants(variant_results)
            topic_metrics[variant] = calculate_topic_accuracies(variant_df)
            
            topic_file = os.path.join(evaluation_dir, f"{variant}_topic_metrics.csv")
            topic_metrics[variant].to_csv(topic_file)
    
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
    
    logger.info("Math experiment completed successfully")
    return {
        'variant_metrics': variant_metrics,
        'summary_df': summary_df,
        'topic_metrics': topic_metrics
    }

if __name__ == "__main__":
    # Example of direct usage
    import argparse
    from ..utils.config import load_config
    
    parser = argparse.ArgumentParser(description="Run Math MVP experiment")
    parser.add_argument("--config", "-c", required=True, help="Path to experiment configuration file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    run_math_experiment(config) 