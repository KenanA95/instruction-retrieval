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
VARIANTS=("baseline" "concise" "high_school" "graduate" "llm")
TOPICS=("calculus__differentiate" "algebra__polynomial_roots" "numbers__list_prime_factors")
NUM_PROBLEMS=1
OUTPUT_DIR="results/sanity_check"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset)
      DATASET_PATH="$2"
      shift 2
      ;;
    --topics)
      # Convert comma-separated list to array
      IFS=',' read -r -a TOPICS <<< "$2"
      shift 2
      ;;
    --variants)
      # Convert comma-separated list to array
      IFS=',' read -r -a VARIANTS <<< "$2"
      shift 2
      ;;
    --num-problems)
      NUM_PROBLEMS="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Verify that instruction files exist
echo "Step 1: Verifying instruction files..."
missing_files=0
for topic in "${TOPICS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    instruction_file="$INSTRUCTIONS_DIR/$variant/${topic}_composed.txt"
    if [ ! -f "$instruction_file" ]; then
      echo "  Missing instruction file: $instruction_file"
      missing_files=$((missing_files + 1))
    fi
  done
done

if [ $missing_files -gt 0 ]; then
  echo "Error: $missing_files instruction files are missing. Please generate them first."
  echo "You can generate them using: ./scripts/generate_instructions.py"
  exit 1
fi
echo "  All instruction files exist."

# Step 2: Run tests to verify answer extraction
echo "Step 2: Running answer extraction tests..."
python -m tests.test_answer_extraction

# Step 3: Run sanity check on a few problems
echo "Step 3: Running sanity check on a few problems..."
# Add instructions_ prefix to variants for the sanity check script
SANITY_VARIANTS=()
for variant in "${VARIANTS[@]}"; do
  SANITY_VARIANTS+=("$variant" "instructions_$variant")
done

# Run the sanity check and save the log
./scripts/run_sanity_check.py \
  --dataset "$DATASET_PATH" \
  --num-problems "$NUM_PROBLEMS" \
  --variants "${SANITY_VARIANTS[@]}" \
  --topics "${TOPICS[@]}" \
  --output-dir "$OUTPUT_DIR" 2>&1 | tee "$OUTPUT_DIR/sanity_check.log"

# Step 4: Display results
echo "Step 4: Displaying results..."
echo "Sanity check results are available in $OUTPUT_DIR/sanity_check_results.json"

# Check if any variant had 0% accuracy
accuracy_issues=0
for variant in "${SANITY_VARIANTS[@]}"; do
  # Extract accuracy from the log file
  accuracy=$(grep -A 10 "Summary:" "$OUTPUT_DIR/sanity_check.log" | grep "$variant:" | awk '{print $NF}' | tr -d '()%')
  if [ -z "$accuracy" ] || [ "$accuracy" = "0.00" ]; then
    echo "  Warning: $variant had 0% accuracy. Please check the results."
    accuracy_issues=$((accuracy_issues + 1))
  fi
done

if [ $accuracy_issues -gt 0 ]; then
  echo "Warning: $accuracy_issues variants had 0% accuracy. Please check the results before running the full experiment."
else
  echo "All variants had non-zero accuracy. Sanity check passed!"
fi

echo "Sanity check completed successfully!"
echo "You can now run the full experiment with: ./run_pipeline.sh" 