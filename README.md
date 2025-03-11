# Instruction Retrieval for Small, On-the-Fly Reasoning Models

This research project investigates whether adding explicit reasoning instructions at inference time can improve the performance of small language models on reasoning tasks, enabling them to perform complex reasoning without extensive fine-tuning or memorization.

## Project Overview

Large language models often appear to reason effectively but may rely on memorized patterns rather than true generalization. This project explores a lightweight alternative: using small, general-purpose reasoning models that retrieve task-specific instructions at inference time.

Our approach aims to replace the implicit knowledge embedded in large models' weights with explicit, retrievable instructions that can be updated without retraining the model.

## Domains and Tasks

1. **Math MVP**: Proof-of-concept on math problems from the DeepMind AMR dataset
   - Topics: polynomials, integration, prime factorization
   - Different instruction variants and combinations with Chain-of-Thought

2. **MedQA**: Medical domain questions from the USMLE
   - Tests the model's ability to apply medical knowledge using instruction retrieval
   - Compares instruction retrieval with RAG using textbook passages

3. **Legal Case Rulings**: Legal reasoning using the CaseHOLD dataset
   - Tests complex legal reasoning by retrieving domain-specific instructions
   - Compares instruction retrieval with case law RAG

## Experimental Setup

We compare the following approaches:
1. Zero-shot (baseline)
2. Few-shot with Chain-of-Thought (CoT) examples
3. Instruction retrieval with different instruction types:
   - Baseline instructions
   - Concise instructions
   - High school level instructions
   - Graduate level instructions
   - LLM-optimized instructions
4. Combinations of instruction retrieval + few-shot CoT
5. RAG (for MedQA and CaseHOLD domains)

## Running the Experiment

See [INSTRUCTIONS.md](INSTRUCTIONS.md) for detailed steps to run the experiment.

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

## License

This project is for research purposes only.
