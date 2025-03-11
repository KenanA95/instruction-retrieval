#!/usr/bin/env python
"""
CaseHOLD task sanity check script.

This script runs a quick sanity check for the CaseHOLD task to ensure that:
1. Templates and instructions exist
2. The models can run on a small number of examples from each variant
3. Results are properly saved and evaluated
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tasks.casehold import run_casehold_experiment
from src.utils.config import load_config
from src.evaluation.run_evaluation import evaluate_all_variants
from src.utils.logging import get_logger

logger = get_logger(__name__)

def run_casehold_sanity_check(config_path=None, num_examples=2):
    """
    Run a sanity check for the CaseHOLD task.
    
    Args:
        config_path: Path to config file, or None to use default
        num_examples: Number of examples to run per variant
    
    Returns:
        bool: True if sanity check passed, False otherwise
    """
    # Load config
    if not config_path:
        config_path = os.path.join("configs", "casehold", "config.json")
    
    logger.info(f"Running CaseHOLD sanity check with config from {config_path}")
    config = load_config(config_path)
    
    # Create a sanity check output directory
    original_output_dir = config.get("output_dir", "results/casehold")
    sanity_output_dir = os.path.join(original_output_dir, "sanity_check")
    config["output_dir"] = sanity_output_dir
    
    # Set a small sample size
    config["sample_size"] = num_examples
    
    # Run the experiment
    try:
        logger.info(f"Running CaseHOLD experiment with {num_examples} examples per variant")
        run_casehold_experiment(config)
        
        # Check that results were generated
        inference_results_dir = os.path.join(sanity_output_dir, "inference_results")
        if not os.path.exists(inference_results_dir):
            logger.error("No inference results were generated")
            return False
        
        # Check that at least one result file exists for each variant
        variants = config.get("variants", [])
        for variant in variants:
            variant_results = list(Path(inference_results_dir).glob(f"*{variant}*.json"))
            if not variant_results:
                logger.error(f"No results found for variant {variant}")
                return False
            logger.info(f"Found {len(variant_results)} result files for variant {variant}")
        
        # Run evaluation on the results
        logger.info("Running evaluation on sanity check results")
        try:
            evaluate_all_variants(sanity_output_dir)
            logger.info("Evaluation completed successfully")
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return False
        
        logger.info("CaseHOLD sanity check completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error during CaseHOLD sanity check: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run CaseHOLD sanity check")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--num-examples", type=int, default=2, help="Number of examples to run per variant")
    
    args = parser.parse_args()
    
    success = run_casehold_sanity_check(args.config, args.num_examples)
    
    if success:
        logger.info("✅ CaseHOLD sanity check PASSED")
        sys.exit(0)
    else:
        logger.error("❌ CaseHOLD sanity check FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main() 