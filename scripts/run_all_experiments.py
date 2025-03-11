#!/usr/bin/env python
"""
Script to run all instruction retrieval experiments.
"""

import os
import argparse
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tasks import run_math_experiment, run_medqa_experiment, run_casehold_experiment
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

def run_experiments(args):
    """Run the specified experiments."""
    
    results = {}
    
    # Run Math experiment if requested
    if args.math or args.all:
        logger.info("Starting Math experiment")
        math_config = load_config(args.math_config)
        math_results = run_math_experiment(math_config)
        results['math'] = math_results
        logger.info("Math experiment completed")
    
    # Run MedQA experiment if requested
    if args.medqa or args.all:
        logger.info("Starting MedQA experiment")
        medqa_config = load_config(args.medqa_config)
        medqa_results = run_medqa_experiment(medqa_config)
        results['medqa'] = medqa_results
        logger.info("MedQA experiment completed")
    
    # Run CaseHOLD experiment if requested
    if args.casehold or args.all:
        logger.info("Starting CaseHOLD experiment")
        casehold_config = load_config(args.casehold_config)
        casehold_results = run_casehold_experiment(casehold_config)
        results['casehold'] = casehold_results
        logger.info("CaseHOLD experiment completed")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run instruction retrieval experiments")
    
    # Experiment selection
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--math", action="store_true", help="Run Math experiment")
    parser.add_argument("--medqa", action="store_true", help="Run MedQA experiment")
    parser.add_argument("--casehold", action="store_true", help="Run CaseHOLD experiment")
    
    # Configuration files
    parser.add_argument("--math-config", default="configs/math_config.json", 
                        help="Path to Math experiment config")
    parser.add_argument("--medqa-config", default="configs/medqa_config.json", 
                        help="Path to MedQA experiment config")
    parser.add_argument("--casehold-config", default="configs/casehold_config.json", 
                        help="Path to CaseHOLD experiment config")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Verify that at least one experiment is selected
    if not (args.all or args.math or args.medqa or args.casehold):
        parser.error("No experiment selected. Use --all or specify at least one of --math, --medqa, --casehold")
    
    # Run the selected experiments
    results = run_experiments(args)
    
    print("\nAll experiments completed successfully") 