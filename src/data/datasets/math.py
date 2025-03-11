import os
import pandas as pd
import glob
from pathlib import Path
from .base import BaseDataset
from ...utils.io import load_dataframe, save_dataframe

class MathDataset(BaseDataset):
    def __init__(self, name="math", config=None):
        super().__init__(name, config)
        # Get base_dir from config with a sensible default
        self.base_dir = self.config.get('base_dir', '/shared/3/projects/instruction-retrieval/mathematics_dataset')
        self.output_dir = self.config.get('output_dir', os.path.join(self.base_dir, 'processed'))
        self.samples_per_topic = self.config.get('samples_per_topic', 100)
        self.difficulty = self.config.get('difficulty', 'easy')
    
    def load(self, path=None):
        """Load dataset from file or raw data sources, optionally filtering by topics."""
        # Get topics from config
        self.topics = self.config.get('topics', [])
        
        # Determine file path source (priority order: config, parameter, default)
        file_path = self.config.get('file_path')
        if file_path and os.path.exists(file_path):
            print(f"Loading dataset from config file_path: {file_path}")
            load_path = file_path
        elif path and os.path.exists(path):
            print(f"Loading dataset from path parameter: {path}")
            load_path = path
        else:
            # Use default processed file
            processed_file = os.path.join(self.output_dir, f"{self.difficulty}_{self.samples_per_topic}.tsv")
            if os.path.exists(processed_file):
                print(f"Loading dataset from processed file: {processed_file}")
                load_path = processed_file
            else:
                # No existing file, need to load from raw data
                return self._load_from_raw_files()
        
        # Load the dataset from file
        self.data = load_dataframe(load_path, format='tsv' if load_path.endswith('.tsv') else None)
        
        # Filter by topics if specified
        if self.topics:
            print(f"Filtering dataset to only include topics: {self.topics}")
            self.data = self.data[self.data['topic'].isin(self.topics)]
            print(f"Filtered to {len(self.data)} examples")
        
        return self
    
    def _load_from_raw_files(self):
        """Load the dataset from raw text files."""
        # Get raw data directory
        difficulty_dir = os.path.join(self.base_dir, f"train-{self.difficulty}")
        if not os.path.exists(difficulty_dir):
            raise FileNotFoundError(f"Difficulty directory not found: {difficulty_dir}")
        
        # Get all text files in the difficulty directory
        txt_files = glob.glob(os.path.join(difficulty_dir, "*.txt"))
        
        all_data = []
        for file_path in txt_files:
            # Extract topic from filename
            topic = os.path.basename(file_path).replace(".txt", "")
            
            # Skip if not in topics list (if specified)
            if self.topics and topic not in self.topics:
                continue
                
            # Read only enough lines to get samples_per_topic examples
            with open(file_path, 'r', encoding='utf-8') as f:
                samples_collected = 0
                
                while samples_collected < self.samples_per_topic:
                    # Read question line
                    question_line = f.readline()
                    if not question_line:  # End of file
                        break
                    
                    # Read answer line
                    answer_line = f.readline()
                    if not answer_line:  # Unexpected end of file
                        break
                    
                    question = question_line.strip()
                    answer = answer_line.strip()
                    
                    # Add to data
                    all_data.append({
                        "topic": topic,
                        "question": question,
                        "answer": answer
                    })
                    
                    samples_collected += 1
        
        self.data = pd.DataFrame(all_data)
        print(f"Loaded {len(self.data)} examples from {len(self.data['topic'].unique())} topics")
        return self
    
    def process(self):
        # Additional processing if needed
        # For now, just ensure the data is loaded
        if self.data is None:
            self.load()
        
        # Save processed data
        os.makedirs(self.output_dir, exist_ok=True)
        output_file = os.path.join(self.output_dir, f"{self.difficulty}_{self.samples_per_topic}.tsv")
        save_dataframe(self.data, output_file, format='tsv')
        
        return self
    
    def get_cot_examples(self, topic, num_examples=3):
        """Get Chain-of-Thought examples for a specific topic.
        
        Handles variations in topic naming (with or without _composed suffix).
        """
        # Get CoT examples path from config or use default
        cot_examples_path = self.config.get('cot_examples_path')
        if not cot_examples_path:
            cot_examples_path = os.path.join(self.output_dir, 'cot_examples_with_solutions.tsv')
            
        if not os.path.exists(cot_examples_path):
            raise FileNotFoundError(f"CoT examples file not found: {cot_examples_path}")
        
        # Load CoT examples
        df_cot = pd.read_csv(cot_examples_path, sep='\t')
        
        # Try different variations of the topic name
        # This handles cases where topic name might be with or without _composed suffix
        possible_topics = [
            topic,  # Original topic name
            topic.replace('_composed', ''),  # Without _composed
            f"{topic}_composed" if not topic.endswith('_composed') else topic  # With _composed
        ]
        
        # Try each topic variation until we find examples
        for try_topic in possible_topics:
            topic_examples = df_cot[df_cot['topic'] == try_topic]
            if not topic_examples.empty:
                if try_topic != topic:
                    print(f"Using examples from topic variant: {try_topic}")
                break
        
        # Report if we don't have enough examples
        if len(topic_examples) < num_examples:
            print(f"Warning: Only {len(topic_examples)} CoT examples available for {topic}")
            return topic_examples
        
        # Sample the specified number of examples
        return topic_examples.sample(num_examples, random_state=42) 