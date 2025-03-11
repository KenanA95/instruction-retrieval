# CaseHOLD Task Data

This directory contains data for the legal reasoning task, including case holdings and domain-specific legal instructions based on the CaseHOLD dataset.

## Data Structure

- `examples/`: Contains examples from the CaseHOLD dataset
  - Each file contains multiple legal questions in a structured format
  - Questions require selecting the correct holding for a legal case based on context

- `instructions/`: Domain-specific instructions for legal reasoning
  - `legal_analysis/`: Instructions for analyzing legal cases
  - `precedent/`: Instructions for understanding and applying legal precedent
  - `statutory/`: Instructions for statutory interpretation
  - Each instruction file focuses on a specific aspect of legal reasoning

- `retrieval_corpus/`: (If applicable) Contains relevant case law passages for retrieval-augmented generation
  - Organized by jurisdiction and legal domain
  - Used for RAG variants of the experiment

## Instruction Types

The instructions are organized by legal domain:

### Legal Analysis Instructions
Instructions that explain the process of legal case analysis, identifying key facts, issues, and holdings.

### Precedent Instructions
Guidelines for understanding how precedent works in legal reasoning and how to apply previous case holdings to new situations.

### Statutory Interpretation Instructions
Explanations of methods for interpreting statutes and regulations in legal contexts.

## File Format

### Question Format
Questions in the examples directory follow this structure:
```json
{
  "context": "The court has previously held that evidence obtained in violation of the Fourth Amendment must be excluded from trial, subject to certain exceptions.",
  "query": "What is the appropriate ruling in a case where evidence was obtained through an illegal search but would have inevitably been discovered through legal means?",
  "options": {
    "A": "The evidence must be excluded as fruit of the poisonous tree.",
    "B": "The evidence is admissible under the inevitable discovery exception.",
    "C": "The evidence is admissible only if obtained in good faith.",
    "D": "The evidence is admissible only if it is non-testimonial in nature."
  },
  "answer": "B",
  "area": "criminal_procedure",
  "subtopic": "fourth_amendment"
}
```

### Instruction Format
Instructions are text files containing detailed explanations of legal concepts, reasoning frameworks, and guidelines for interpreting and applying case law.

## Using the Data

This data is used by the CaseHOLD experiment to evaluate how well language models can utilize domain-specific legal instructions to answer questions about legal holdings and precedents.

To run experiments with this data:
```bash
python scripts/run_all_experiments.py --casehold
```

To run a sanity check with a small subset of the data:
```bash
python scripts/sanity_check.py --casehold
``` 