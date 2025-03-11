import json
import os
import pandas as pd
from pathlib import Path

def ensure_dir(directory):
    """Ensure a directory exists."""
    os.makedirs(directory, exist_ok=True)
    return directory

def save_json(data, filepath, indent=2):
    """Save data to a JSON file."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)
    return filepath

def load_json(filepath):
    """Load data from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_dataframe(df, filepath, format='csv'):
    """Save a pandas DataFrame to a file."""
    ensure_dir(os.path.dirname(filepath))
    
    if format.lower() == 'csv':
        df.to_csv(filepath, index=False)
    elif format.lower() == 'tsv':
        df.to_csv(filepath, sep='\t', index=False)
    elif format.lower() == 'json':
        df.to_json(filepath, orient='records', indent=2)
    elif format.lower() == 'parquet':
        df.to_parquet(filepath, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return filepath

def load_dataframe(filepath, format=None):
    """Load a pandas DataFrame from a file."""
    if format is None:
        # Infer format from file extension
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.csv':
            format = 'csv'
        elif ext == '.tsv':
            format = 'tsv'
        elif ext == '.json':
            format = 'json'
        elif ext == '.parquet':
            format = 'parquet'
        else:
            raise ValueError(f"Could not infer format from extension: {ext}")
    
    if format.lower() == 'csv':
        return pd.read_csv(filepath)
    elif format.lower() == 'tsv':
        return pd.read_csv(filepath, sep='\t')
    elif format.lower() == 'json':
        return pd.read_json(filepath, orient='records')
    elif format.lower() == 'parquet':
        return pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")

def list_files(directory, pattern='*'):
    """List files in a directory matching a pattern."""
    return list(Path(directory).glob(pattern)) 