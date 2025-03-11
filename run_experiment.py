#!/usr/bin/env python3
import os
import sys
import argparse
from src.experiments.run_experiment import main as run_experiment

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run an instruction retrieval experiment")
    parser.add_argument("--config", "-c", required=True, help="Path to experiment configuration file")
    
    # Add any additional arguments here
    
    args = parser.parse_args()
    
    # Run the experiment with the config path
    run_experiment(args.config) 