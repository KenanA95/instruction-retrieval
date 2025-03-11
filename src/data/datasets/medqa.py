import os
import json
import pandas as pd
from pathlib import Path

from ...utils.io import load_json, ensure_dir

class MedQADataset:
    """
    Dataset loader for MedQA (USMLE) medical questions.
    
    This handles loading and preprocessing the MedQA dataset
    which consists of multiple-choice medical questions from
    the United States Medical Licensing Examination (USMLE).
    """
    
    def __init__(self, config=None):
        """
        Initialize the MedQA dataset.
        
        Args:
            config: Configuration dictionary with options:
                - base_dir: Base directory for data
                - split: Data split to use (train/dev/test)
                - sample_size: Optional limit on number of samples to load
        """
        self.config = config or {}
        self.base_dir = self.config.get('base_dir', '/shared/3/projects/instruction-retrieval/medqa')
        self.split = self.config.get('split', 'test')
        self.sample_size = self.config.get('sample_size', None)
        self.examples = []
        self.topics = set()
    
    def load(self):
        """
        Load MedQA examples from files.
        
        Returns:
            self for chaining
        """
        # Construct path to the data file
        data_file = os.path.join(self.base_dir, f'medqa_{self.split}.json')
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"MedQA data file not found: {data_file}")
        
        # Load the data
        data = load_json(data_file)
        
        # Process the data
        for item in data:
            example = {
                'id': item.get('id', ''),
                'question': item['question'],
                'options': item['options'],  # A, B, C, D options
                'answer': item['answer'],    # Correct option (A, B, C, D)
                'topic': item.get('topic', 'general')  # Medical topic/category
            }
            
            self.examples.append(example)
            if 'topic' in item:
                self.topics.add(item['topic'])
        
        # Limit sample size if specified
        if self.sample_size and self.sample_size < len(self.examples):
            self.examples = self.examples[:self.sample_size]
            # Recalculate topics
            self.topics = set(example['topic'] for example in self.examples if 'topic' in example)
        
        print(f"Loaded {len(self.examples)} MedQA examples with {len(self.topics)} topics")
        return self
    
    def get_topics(self):
        """
        Get the list of topics in the dataset.
        
        Returns:
            List of topic names
        """
        return sorted(list(self.topics))
    
    def get_examples_by_topic(self, topic):
        """
        Get examples for a specific topic.
        
        Args:
            topic: Topic name to filter by
            
        Returns:
            List of examples for the topic
        """
        return [ex for ex in self.examples if ex.get('topic') == topic]
    
    def __len__(self):
        """Get the number of examples."""
        return len(self.examples)

if __name__ == "__main__":
    # Example usage
    dataset = MedQADataset({
        'base_dir': '/shared/3/projects/instruction-retrieval/medqa',
        'split': 'test',
        'sample_size': 10
    })
    
    dataset.load()
    
    # Print some statistics
    print(f"Number of examples: {len(dataset)}")
    print(f"Topics: {dataset.get_topics()}")
    
    # Print a sample example
    if dataset.examples:
        print("\nSample example:")
        example = dataset.examples[0]
        print(f"Question: {example['question']}")
        print(f"Options: {example['options']}")
        print(f"Answer: {example['answer']}") 