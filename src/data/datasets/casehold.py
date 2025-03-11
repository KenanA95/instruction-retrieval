import os
import json
import pandas as pd
from pathlib import Path

from ...utils.io import load_json, ensure_dir

class CaseHOLDDataset:
    """
    Dataset loader for CaseHOLD legal holdings dataset.
    
    This handles loading and preprocessing the CaseHOLD dataset,
    which consists of legal holdings classification examples.
    """
    
    def __init__(self, config=None):
        """
        Initialize the CaseHOLD dataset.
        
        Args:
            config: Configuration dictionary with options:
                - base_dir: Base directory for data
                - split: Data split to use (train/dev/test)
                - sample_size: Optional limit on number of samples to load
        """
        self.config = config or {}
        self.base_dir = self.config.get('base_dir', '/shared/3/projects/instruction-retrieval/casehold')
        self.split = self.config.get('split', 'test')
        self.sample_size = self.config.get('sample_size', None)
        self.examples = []
        self.categories = set()
    
    def load(self):
        """
        Load CaseHOLD examples from files.
        
        Returns:
            self for chaining
        """
        # Construct path to the data file
        data_file = os.path.join(self.base_dir, f'casehold_{self.split}.json')
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"CaseHOLD data file not found: {data_file}")
        
        # Load the data
        data = load_json(data_file)
        
        # Process the data
        for item in data:
            example = {
                'id': item.get('id', ''),
                'context': item['context'],          # Legal context
                'query': item['query'],              # The query with a blank slot
                'candidates': item['candidates'],    # List of possible holdings
                'answer': item['answer'],            # Correct holding (often a letter/index)
                'category': item.get('category', 'general')  # Legal category
            }
            
            self.examples.append(example)
            if 'category' in item:
                self.categories.add(item['category'])
        
        # Limit sample size if specified
        if self.sample_size and self.sample_size < len(self.examples):
            self.examples = self.examples[:self.sample_size]
            # Recalculate categories
            self.categories = set(example['category'] for example in self.examples if 'category' in example)
        
        print(f"Loaded {len(self.examples)} CaseHOLD examples with {len(self.categories)} categories")
        return self
    
    def get_categories(self):
        """
        Get the list of legal categories in the dataset.
        
        Returns:
            List of category names
        """
        return sorted(list(self.categories))
    
    def get_examples_by_category(self, category):
        """
        Get examples for a specific legal category.
        
        Args:
            category: Category name to filter by
            
        Returns:
            List of examples for the category
        """
        return [ex for ex in self.examples if ex.get('category') == category]
    
    def __len__(self):
        """Get the number of examples."""
        return len(self.examples)

if __name__ == "__main__":
    # Example usage
    dataset = CaseHOLDDataset({
        'base_dir': '/shared/3/projects/instruction-retrieval/casehold',
        'split': 'test',
        'sample_size': 10
    })
    
    dataset.load()
    
    # Print some statistics
    print(f"Number of examples: {len(dataset)}")
    print(f"Categories: {dataset.get_categories()}")
    
    # Print a sample example
    if dataset.examples:
        print("\nSample example:")
        example = dataset.examples[0]
        print(f"Context: {example['context'][:100]}...")
        print(f"Query: {example['query']}")
        print(f"Candidates: {example['candidates']}")
        print(f"Answer: {example['answer']}") 