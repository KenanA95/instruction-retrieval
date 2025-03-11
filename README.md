# Instruction Retrieval for Small Reasoning Models

This research project investigates whether adding explicit reasoning instructions at inference time can improve the performance of small language models on reasoning tasks.

## Research Questions

- Can explicit reasoning instructions help small models perform better on reasoning tasks?
- How does instruction retrieval compare to few-shot Chain-of-Thought (CoT) examples?
- What combination of approaches yields the best results?

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

## Running the Experiment

See [INSTRUCTIONS.md](INSTRUCTIONS.md) for detailed steps to run the experiment.

## Project Structure

```
instruction-retrieval/
├── src/                      # Source code
│   ├── data/                 # Data processing and prompt generation
│   ├── models/               # Model inference
│   ├── evaluation/           # Evaluation metrics and analysis
│   ├── utils/                # Utility functions
│   └── experiments/          # Experiment runners
├── scripts/                  # Utility scripts
├── tests/                    # Unit tests
├── configs/                  # Configuration files
└── results/                  # Experiment results
```

## License

This project is for research purposes only.
