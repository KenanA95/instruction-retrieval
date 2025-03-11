# Math Task Data

This directory contains data for the math reasoning task, including problems and domain-specific instructions.

## Data Structure

- `examples/`: Contains examples from the DeepMind Analysing Mathematical Reasoning dataset (subset)
  - Each file contains multiple math problems in a structured format
  - Problems cover topics like algebra, calculus, and number theory
  
- `instructions/`: Domain-specific instructions for solving math problems
  - `baseline/`: General mathematical instructions
  - `graduate/`: Advanced mathematical instructions
  - Each instruction file focuses on a specific mathematical concept or technique

## Instruction Types

The instructions are organized by difficulty level:

### Baseline Instructions
Basic instructions that explain fundamental mathematical concepts and problem-solving techniques.

### Graduate Instructions
Advanced instructions that cover more complex mathematical reasoning and techniques typically taught at graduate level.

## File Format

### Problem Format
Problems in the examples directory follow this structure:
```json
{
  "problem": "Solve the following equation: 2x + 3 = 7",
  "solution": "2x + 3 = 7\n2x = 4\nx = 2",
  "answer": "2",
  "topic": "algebra",
  "subtopic": "linear_equations"
}
```

### Instruction Format
Instructions are text files containing detailed explanations of mathematical concepts and step-by-step methods for solving problems in specific domains.

## Using the Data

This data is used by the Math MVP experiment to evaluate how well language models can utilize domain-specific instructions to solve mathematical problems.

To run experiments with this data:
```bash
python scripts/run_all_experiments.py --math
```

To run a sanity check with a small subset of the data:
```bash
python scripts/sanity_check.py --math
``` 