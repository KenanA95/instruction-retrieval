#!/bin/bash
set -e  # Exit on error

# Set CUDA_VISIBLE_DEVICES if not already set
# Use all available GPUs by default
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
echo "Using CUDA devices: $CUDA_VISIBLE_DEVICES"

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ] && [ ! -f .env ]; then
  echo "Error: OPENAI_API_KEY environment variable not set and no .env file found."
  echo "Please set the OPENAI_API_KEY environment variable or create a .env file."
  exit 1
fi

# Load .env file if it exists
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# Define variables
DATASET_PATH="/shared/3/projects/instruction-retrieval/mathematics_dataset/processed/easy_100.tsv"
TEMPLATES_DIR="src/data/templates"
INSTRUCTIONS_DIR="src/data/instructions"
CONFIG_PATH="configs/experiments/math_experiment.json"
VARIANTS=("baseline" "concise" "high_school" "graduate" "llm")
TOPICS=("calculus__differentiate" "algebra__polynomial_roots" "numbers__list_prime_factors")

# Check if sanity check has been run
echo "Checking if sanity check has been run..."
if [ ! -f "results/sanity_check/sanity_check_results.json" ]; then
  echo "Warning: Sanity check has not been run. It is recommended to run it first:"
  echo "  ./run_sanity_check.sh"
  read -p "Do you want to continue without running the sanity check? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Exiting. Please run the sanity check first."
    exit 1
  fi
fi

# Step 1: Generate instruction prompts
echo "Step 1: Generating instruction prompts..."
./scripts/generate_instructions.py \
  --model o1 \
  --dataset "$DATASET_PATH" \
  --templates-dir "$TEMPLATES_DIR" \
  --output-dir "$INSTRUCTIONS_DIR" \
  --topics "${TOPICS[@]}" \
  --variants "${VARIANTS[@]}"

# Step 2: Run tests to verify answer extraction
echo "Step 2: Running answer extraction tests..."
python -m tests.test_answer_extraction

# Step 3: Run the full experiment
echo "Step 3: Running the full experiment..."
./run_experiment.py --config "$CONFIG_PATH"

echo "Experiment pipeline completed successfully!"
echo "Results are available in the results/math_instruction_retrieval directory." 