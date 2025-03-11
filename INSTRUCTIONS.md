# Instruction Retrieval Experiment Instructions

This document provides instructions for running the instruction retrieval experiments across all domains.

## Prerequisites

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your API keys if needed:
   ```bash
   cp .env.template .env
   # Edit .env to add your API keys
   ```

## Running Experiments

### Step 1: Run Sanity Checks

Before running the full experiments, it's recommended to run sanity checks to ensure each task is set up correctly. Each task has its own dedicated sanity check script:

```bash
# Run Math sanity check
python scripts/math_sanity_check.py

# Run MedQA sanity check
python scripts/medqa_sanity_check.py

# Run CaseHOLD sanity check
python scripts/casehold_sanity_check.py
```

Each sanity check script:
1. Runs a small number of examples from each prompt variant
2. Saves the inference results
3. Runs evaluation on those results
4. Reports whether the full pipeline is working end-to-end

You can customize the sanity checks with options:

```bash
# Run with a custom number of examples (default is 2)
python scripts/math_sanity_check.py --num-examples 3

# Run with a custom config file
python scripts/math_sanity_check.py --config configs/math/custom_config.json
```

The sanity check results will be saved in a `sanity_check` subdirectory of each task's results directory.

### Step 2: Run the Full Experiments

After the sanity checks pass, run the full experiments:

```bash
# Run all experiments
python scripts/run_all_experiments.py --all

# Run specific experiments
python scripts/run_all_experiments.py --math
python scripts/run_all_experiments.py --medqa
python scripts/run_all_experiments.py --casehold

# Run multiple experiments
python scripts/run_all_experiments.py --math --medqa

# Run with custom configurations
python scripts/run_all_experiments.py --math --math-config configs/math/custom_config.json
```

## Experiment Details

### Math MVP Experiment

- **Dataset**: DeepMind Analysing Mathematical Reasoning dataset (subset)
- **Topics**: Polynomials, Integration, Prime Factorization
- **Variants**:
  - baseline (zero-shot)
  - few_shot_cot (Chain-of-Thought examples)
  - instructions_* (various instruction types)
  - instructions_*_few_shot_cot (instructions + CoT)

### MedQA Experiment

- **Dataset**: MedQA (USMLE) medical questions
- **Variants**:
  - zero_shot (baseline)
  - rag (Retrieval-Augmented Generation with textbook passages)
  - instruction_retrieval (domain-specific instructions)
  - rag_instruction_retrieval (RAG + instructions)

### CaseHOLD Experiment

- **Dataset**: CaseHOLD legal holdings dataset
- **Variants**:
  - zero_shot (baseline)
  - rag (Retrieval-Augmented Generation with case law)
  - instruction_retrieval (domain-specific instructions)
  - rag_instruction_retrieval (RAG + instructions)

## Generating Instructions

To generate new instructions for any domain:

```bash
python scripts/generate_instructions.py --domain math --model llama-3.2-3b-instruct
python scripts/generate_instructions.py --domain medqa --model llama-3.2-3b-instruct
python scripts/generate_instructions.py --domain casehold --model llama-3.2-3b-instruct
```

## Prompt Format

The system now uses Markdown formatting for prompts to improve readability and structure. Prompts are broken into clearly labeled sections:

```markdown
# Instruction Retrieval Task

## Domain Instructions
[Domain-specific instructions for the task]

## Retrieved Context
[Retrieved context from textbooks or case law, if applicable]

## Chain-of-Thought Examples
[Examples with step-by-step reasoning, if applicable]

## Problem to Solve
[Actual problem the model needs to solve]

Provide a step-by-step solution to this problem.
```

This structured format helps the model understand the different components of the prompt and makes it clear what needs to be answered.

## Viewing Results

Results will be available in:
- `results/math/` - Math experiment results
- `results/medqa/` - MedQA experiment results
- `results/casehold/` - CaseHOLD experiment results

Each results directory contains:
- `evaluation/` - Evaluation metrics and summaries
- `visualizations/` - Plots and visualizations
- `inference_results/` - Raw inference results

## Project Structure

```
instruction-retrieval/
├── README.md                 # Project overview
├── requirements.txt          # Python dependencies
├── src/                      # Core Python source code
│   ├── data/                 # Data preparation and loading
│   ├── modeling/             # Model inference, prompt runners, retrieval
│   ├── evaluation/           # Evaluation metrics and analysis
│   ├── tasks/                # Task orchestration for each domain
│   └── utils/                # Utility functions
├── scripts/                  # Command-line utilities
├── data/                     # Lightweight resources (templates, instructions)
└── results/                  # Generated outputs and visualizations
```

## Troubleshooting

- If you encounter CUDA out-of-memory errors, try reducing the batch size in the configuration files
- Check logs in the results directories for detailed error messages
- For retrieval issues in MedQA or CaseHOLD, verify that the corpus paths are correct in the configuration files 