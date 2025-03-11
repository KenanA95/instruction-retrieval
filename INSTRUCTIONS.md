# Instruction Retrieval Experiment Instructions

This document provides concise instructions for running the instruction retrieval experiment.

## Prerequisites

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your OpenAI API key:
   ```bash
   cp .env.template .env
   # Edit .env to add your OpenAI API key
   ```

## Step-by-Step Guide

### Step 1: Generate Instruction Prompts

Generate instruction prompts for all variants using OpenAI's o1 model:

```bash
./scripts/generate_instructions.py \
  --model o1 \
  --dataset /shared/3/projects/instruction-retrieval/mathematics_dataset/processed/easy_100.tsv \
  --templates-dir src/data/templates \
  --output-dir src/data/instructions \
  --topics calculus__differentiate algebra__polynomial_roots numbers__list_prime_factors \
  --variants baseline concise high_school graduate llm
```

This will:
- Use OpenAI's o1 model to generate instruction prompts
- Save the generated instructions to the appropriate files
- Create both the instructions and the prompts used to generate them

### Step 2: Verify Answer Extraction

Run the answer extraction tests to ensure the extraction logic works correctly:

```bash
python -m tests.test_answer_extraction
```

### Step 3: Run Sanity Check

Run a sanity check on a few problems to verify that:
- The prompts are generated correctly
- The model can run on a handful of problems
- Answers are parsed correctly

```bash
# Run with default GPU settings (all available GPUs)
./run_sanity_check.sh

# Or specify which GPUs to use
CUDA_VISIBLE_DEVICES=0 ./run_sanity_check.sh
```

You can customize the sanity check with these options:
```bash
CUDA_VISIBLE_DEVICES=0,1 ./run_sanity_check.sh \
  --dataset /path/to/dataset.tsv \
  --topics calculus__differentiate,algebra__polynomial_roots \
  --variants baseline,concise \
  --num-problems 2 \
  --output-dir results/my_sanity_check
```

The sanity check will:
1. Verify that all required instruction files exist
2. Run answer extraction tests
3. Test the model on a small number of problems
4. Check if any variants have 0% accuracy

### Step 4: Run the Full Experiment

Once the sanity checks pass, run the full experiment:

```bash
# Run with default GPU settings (all available GPUs)
./run_pipeline.sh

# Or specify which GPUs to use
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_pipeline.sh
```

### Alternative: Run the Complete Pipeline

Alternatively, you can run the entire pipeline with a single command:

```bash
./run_pipeline.sh
```

This script will:
1. Generate instruction prompts using OpenAI's o1 model
2. Run tests to verify answer extraction
3. Run a sanity check on a few problems
4. Run the full experiment

## Experiment Details

- **Models**: 
  - Inference: Llama-3.2-3B-Instruct
  - Answer Extraction: Llama-3.1-8B-Instruct

- **Variants**:
  - baseline
  - few_shot_cot
  - instructions_baseline
  - instructions_baseline_few_shot_cot
  - instructions_concise
  - instructions_concise_few_shot_cot
  - instructions_high_school
  - instructions_high_school_few_shot_cot
  - instructions_graduate
  - instructions_graduate_few_shot_cot
  - instructions_llm
  - instructions_llm_few_shot_cot

- **Topics**:
  - calculus__differentiate
  - algebra__polynomial_roots
  - numbers__list_prime_factors

## Viewing Results

Results will be available in:
- `results/math_instruction_retrieval/evaluation/` - Evaluation metrics
- `results/math_instruction_retrieval/visualizations/` - Visualizations
- `results/math_instruction_retrieval/inference_results/` - Raw inference results

## Troubleshooting

- If answer extraction fails, run the tests:
  ```bash
  python -m tests.test_answer_extraction
  ```

- Check logs in `results/math_instruction_retrieval/logs/` for errors 