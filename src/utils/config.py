import json
import os
from pathlib import Path

class Config:
    def __init__(self, config_path=None, **kwargs):
        self.config = {}
        
        # Load from file if provided
        if config_path:
            self.load_from_file(config_path)
            
        # Update with any provided kwargs
        self.config.update(kwargs)
    
    def load_from_file(self, config_path):
        """Load configuration from a JSON file."""
        with open(config_path, 'r') as f:
            self.config.update(json.load(f))
        return self
    
    def save_to_file(self, config_path):
        """Save configuration to a JSON file."""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        return self
    
    def get(self, key, default=None):
        """Get a configuration value."""
        return self.config.get(key, default)
    
    def set(self, key, value):
        """Set a configuration value."""
        self.config[key] = value
        return self
    
    def update(self, **kwargs):
        """Update multiple configuration values."""
        self.config.update(kwargs)
        return self
    
    def __getitem__(self, key):
        return self.config[key]
    
    def __setitem__(self, key, value):
        self.config[key] = value
    
    def __contains__(self, key):
        return key in self.config


def load_config(config_path):
    """Helper function to load a configuration file."""
    return Config(config_path)


def get_project_root():
    """Get the project root directory."""
    # Assuming this file is in src/utils/
    return Path(__file__).parent.parent.parent 