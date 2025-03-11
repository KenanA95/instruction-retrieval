import os
import argparse
import json
from pathlib import Path
import pandas as pd

from ..data.datasets.math import MathDataset
from ..data.prompts.prompt_manager import PromptManager
from ..models.inference import InferenceRunner
from ..evaluation.metrics import evaluate_variants, calculate_topic_accuracies
from ..evaluation.visualization import plot_all_metrics, plot_metrics_by_topic
from ..utils.config import Config
from ..utils.logging import get_experiment_logger
from ..utils.io import ensure_dir, save_json

def run_math_experiment(config):
    """Run a math experiment with the given configuration."""
    # Setup logging
    experiment_name = config.get('experiment_name', 'math_experiment')
    logger = get_experiment_logger(experiment_name)
    logger.info(f"Starting math experiment: {experiment_name}")
    
    # Set up directories
    base_dir = config.get('base_dir', '/shared/3/projects/instruction-retrieval/mathematics_dataset')
    output_dir = config.get('output_dir', 'results')
    ensure_dir(output_dir)
    
    # Prepare dataset config
    dataset_config = config.get('dataset', {}).copy()  # Use a copy to avoid modifying the original
    if 'base_dir' not in dataset_config:
        dataset_config['base_dir'] = base_dir
        
    # Load dataset
    logger.info("Loading dataset...")
    dataset = MathDataset(config=dataset_config)
    dataset.load()
    logger.info(f"Loaded {len(dataset)} examples from {len(dataset.get_topics())} topics")
    
    # Get prompt directories and variants from config
    prompt_dirs = config.get('prompt_dirs')
    if not prompt_dirs:
        raise ValueError("No prompt directories specified in config")
    
    variants = config.get('variants', list(prompt_dirs.keys()))
    num_cot_examples = config.get('num_cot_examples', 3)
    
    # Generate prompt variants
    logger.info("Generating prompt variants...")
    prompt_manager = PromptManager(prompt_dirs=prompt_dirs)
    prompt_variants_dir = os.path.join(output_dir, "prompt_variants")
    prompt_manager.generate_prompt_variants(
        dataset=dataset,
        output_dir=prompt_variants_dir,
        variants=variants,
        num_cot_examples=num_cot_examples
    )
    logger.info(f"Generated prompt variants in {prompt_variants_dir}")
    
    # Prepare for inference
    logger.info("Running inference...")
    inference_config = config.get('inference', {})
    inference_runner = InferenceRunner(
        model_name=inference_config.get('model_name', 'meta-llama/Llama-3.2-1B-Instruct'),
        extract_model_name=inference_config.get('extract_model_name'),
        batch_size=inference_config.get('batch_size', 8)
    )
    
    # Run inference
    inference_results_dir = os.path.join(output_dir, "inference_results")
    results = inference_runner.run_variant_inference(
        variants_dir=prompt_variants_dir,
        output_dir=inference_results_dir,
        experiment_name=experiment_name
    )
    logger.info(f"Inference completed. Results saved to {inference_results_dir}")
    
    # Evaluate results
    logger.info("Evaluating results...")
    combined_results_file = os.path.join(
        inference_results_dir, 
        f"all_variants_{experiment_name}_{inference_runner.model.get_clean_model_name()}.json"
    )
    
    variant_metrics, summary_df = evaluate_variants(combined_results_file)
    logger.info("Variant metrics calculated")
    
    # Save evaluation results
    evaluation_dir = os.path.join(output_dir, "evaluation")
    ensure_dir(evaluation_dir)
    
    summary_file = os.path.join(evaluation_dir, f"{experiment_name}_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Saved summary metrics to {summary_file}")
    
    # Calculate topic-level metrics for each variant
    topic_metrics = {}
    for variant in variants:
        variant_file = os.path.join(inference_results_dir, f"{variant}_{inference_runner.model.get_clean_model_name()}.json")
        if os.path.exists(variant_file):
            with open(variant_file, 'r') as f:
                variant_results = json.load(f)
            
            _, variant_df = evaluate_variants(variant_results)
            topic_metrics[variant] = calculate_topic_accuracies(variant_df)
            
            # Save topic metrics
            topic_file = os.path.join(evaluation_dir, f"{variant}_topic_metrics.csv")
            topic_metrics[variant].to_csv(topic_file)
            logger.info(f"Saved {variant} topic metrics to {topic_file}")
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    visualization_dir = os.path.join(output_dir, "visualizations")
    ensure_dir(visualization_dir)
    
    # Plot variant comparison
    plot_all_metrics(
        summary_df,
        output_dir=visualization_dir,
        prefix=experiment_name
    )
    logger.info(f"Saved variant comparison plots to {visualization_dir}")
    
    # Plot topic metrics for each variant
    for variant, metrics in topic_metrics.items():
        for metric in ['exact_match', 'answer_in_extracted', 'answer_in_raw']:
            if metric in metrics.columns:
                output_file = os.path.join(visualization_dir, f"{variant}_{metric}_by_topic.png")
                plot_metrics_by_topic(
                    metrics,
                    metric=metric,
                    top_n=10,
                    title=f"{variant} - Top 10 Topics by {metric.replace('_', ' ').title()} Accuracy",
                    output_file=output_file
                )
    
    logger.info("Experiment completed successfully")
    return {
        'variant_metrics': variant_metrics,
        'summary_df': summary_df,
        'topic_metrics': topic_metrics
    }

def main(config_path=None):
    """Run an experiment with the given configuration file path."""
    if not config_path:
        # If no config path provided, parse command line arguments
        parser = argparse.ArgumentParser(description="Run an instruction retrieval experiment")
        parser.add_argument("--config", "-c", required=True, help="Path to experiment configuration file")
        args = parser.parse_args()
        config_path = args.config
    
    # Load configuration
    config = Config(config_path)
    
    # Determine experiment type
    experiment_type = config.get('experiment_type', 'math')
    
    # Run the appropriate experiment
    if experiment_type == 'math':
        results = run_math_experiment(config)
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    print(f"Experiment completed successfully")
    return results

if __name__ == "__main__":
    main() 