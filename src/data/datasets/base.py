import pandas as pd
import os
from abc import ABC, abstractmethod
from ...utils.io import save_dataframe, load_dataframe, ensure_dir

class BaseDataset(ABC):
    def __init__(self, name, config=None):
        self.name = name
        self.config = config or {}
        self.data = None
    
    @abstractmethod
    def load(self, path=None):
        """Load the dataset from a file or other source."""
        pass
    
    @abstractmethod
    def process(self):
        """Process the dataset."""
        pass
    
    def save(self, path, format='tsv'):
        """Save the processed dataset."""
        if self.data is None:
            raise ValueError("No data to save. Call load() and process() first.")
        
        return save_dataframe(self.data, path, format)
    
    def get_sample(self, n=5):
        """Get a sample of the dataset."""
        if self.data is None:
            raise ValueError("No data to sample. Call load() first.")
        
        return self.data.sample(min(n, len(self.data)))
    
    def get_topics(self):
        """Get the list of topics in the dataset."""
        if self.data is None or 'topic' not in self.data.columns:
            return []
        
        return self.data['topic'].unique().tolist()
    
    def filter_by_topic(self, topic):
        """Filter the dataset by topic."""
        if self.data is None:
            raise ValueError("No data to filter. Call load() first.")
        
        if 'topic' not in self.data.columns:
            raise ValueError("Dataset does not have a 'topic' column.")
        
        return self.data[self.data['topic'] == topic]
    
    def sample_by_topic(self, topic, n=5):
        """Sample from a specific topic."""
        topic_data = self.filter_by_topic(topic)
        return topic_data.sample(min(n, len(topic_data)))
    
    def __len__(self):
        """Get the number of examples in the dataset."""
        return 0 if self.data is None else len(self.data) 