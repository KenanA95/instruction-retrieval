# MedQA Task Data

This directory contains data for the medical question answering task, including USMLE-style clinical questions and domain-specific medical instructions.

## Data Structure

- `examples/`: Contains examples from the MedQA dataset (USMLE)
  - Each file contains multiple medical questions in a structured format
  - Questions cover a wide range of medical topics including diagnosis, treatment, and epidemiology

- `instructions/`: Domain-specific instructions for medical reasoning
  - `clinical_reasoning/`: Instructions for clinical diagnostic reasoning
  - `treatment/`: Instructions for therapeutic decision making
  - `pathophysiology/`: Instructions explaining disease mechanisms
  - Each instruction file focuses on a specific aspect of medical knowledge or reasoning

- `retrieval_corpus/`: (If applicable) Contains medical textbook passages for retrieval-augmented generation
  - Organized by medical specialty and topic
  - Used for RAG variants of the experiment

## Instruction Types

The instructions are organized by medical domain:

### Clinical Reasoning Instructions
Instructions that explain the process of differential diagnosis, clinical assessment, and diagnostic decision making.

### Treatment Instructions
Guidelines for therapeutic decision making, treatment selection, and management of medical conditions.

### Pathophysiology Instructions
Explanations of disease mechanisms, causation, and progression to support understanding of underlying medical concepts.

## File Format

### Question Format
Questions in the examples directory follow this structure:
```json
{
  "question": "A 65-year-old man presents with chest pain radiating to the left arm. ECG shows ST elevation in leads II, III, and aVF. Which artery is most likely occluded?",
  "options": {
    "A": "Left anterior descending artery",
    "B": "Left circumflex artery",
    "C": "Right coronary artery",
    "D": "Left main coronary artery"
  },
  "answer": "C",
  "topic": "cardiology",
  "subtopic": "myocardial_infarction"
}
```

### Instruction Format
Instructions are text files containing detailed explanations of medical concepts, diagnostic approaches, and treatment guidelines.

## Using the Data

This data is used by the MedQA experiment to evaluate how well language models can utilize domain-specific medical instructions to answer clinical questions.

To run experiments with this data:
```bash
python scripts/run_all_experiments.py --medqa
```

To run a sanity check with a small subset of the data:
```bash
python scripts/sanity_check.py --medqa
``` 