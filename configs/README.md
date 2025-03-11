# Configuration Files

This directory contains configuration files for the instruction retrieval experiments.

## Structure

- `math/`: Configuration files for the Math MVP experiment
  - `config.json`: Main configuration for the math experiment
  
- `medqa/`: Configuration files for the MedQA experiment
  - `config.json`: Main configuration for the MedQA experiment
  
- `casehold/`: Configuration files for the CaseHOLD experiment
  - `config.json`: Main configuration for the CaseHOLD experiment

## Usage

These configuration files are used by the experiment runners to set up and run the experiments. They specify:

- Dataset paths and parameters
- Model configurations
- Experiment variants to run
- Paths to templates and instructions
- Output directories for results

To run an experiment, pass the appropriate config file to the experiment runner:

```bash
python scripts/run_all_experiments.py --math --math-config configs/math/config.json
``` 