#!/usr/bin/env python3
import os
import sys
import argparse
import json
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config import Config
from src.utils.io import ensure_dir, save_json, load_json
from src.utils.logging import setup_logger
from src.models.llm_adapter import LLMAdapter

def load_problems(dataset_path, topic, num_examples=5):
    """Load example problems for a specific topic."""
    try:
        # Check if the file is JSON or TSV
        if dataset_path.endswith('.json'):
            with open(dataset_path, 'r') as f:
                data = json.load(f)
        else:  # Assume TSV
            import pandas as pd
            data = pd.read_csv(dataset_path, sep='\t').to_dict('records')
        
        # Filter by topic
        topic_data = [item for item in data if item.get('topic') == topic]
        
        # Take a sample
        if len(topic_data) > num_examples:
            topic_data = topic_data[:num_examples]
        
        return topic_data
    except Exception as e:
        print(f"Error loading problems: {e}")
        return []

def format_problems(problems):
    """Format problems for the generate_prompt template."""
    formatted = []
    for i, problem in enumerate(problems, 1):
        formatted.append(f"Problem {i}: {problem['question']}")
        formatted.append(f"Answer: {problem['answer']}")
        formatted.append("")
    
    return "\n".join(formatted)

def generate_prompt(template_path, topic, problems_text):
    """Generate a prompt using the template."""
    try:
        with open(template_path, 'r') as f:
            template = f.read()
        
        # Replace placeholders
        prompt = template.replace("{topic}", topic).replace("{problems}", problems_text)
        return prompt
    except Exception as e:
        print(f"Error generating prompt: {e}")
        return ""

def run_llm_to_generate_instructions(prompt, output_path, model_name="meta-llama/Llama-3.1-8B-Instruct"):
    """Run an LLM to generate instructions based on the prompt."""
    # Save the prompt for reference
    with open(output_path + ".prompt.txt", 'w') as f:
        f.write(prompt)
    
    # Initialize the LLM
    llm = LLMAdapter(model_name=model_name, max_new_tokens=2048, temperature=0.7)
    
    # Generate the instructions
    print(f"Generating instructions with {model_name}...")
    instructions = llm.generate(prompt)
    
    # Save the generated instructions
    with open(output_path, 'w') as f:
        f.write(instructions)
    
    print(f"Instructions saved to {output_path}")
    return instructions

def main():
    parser = argparse.ArgumentParser(description="Generate instruction prompts for different variants")
    parser.add_argument("--config", "-c", default="configs/experiments/math_experiment.json", 
                        help="Path to experiment configuration file")
    parser.add_argument("--model", "-m", default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Model to use for generating instructions")
    parser.add_argument("--dataset", "-d", default="/shared/3/projects/instruction-retrieval/mathematics_dataset/processed/easy_100.tsv",
                        help="Path to the dataset file")
    parser.add_argument("--templates-dir", default="src/data/templates",
                        help="Directory containing prompt templates")
    parser.add_argument("--output-dir", "-o", default="src/data/instructions",
                        help="Output directory for generated instructions")
    parser.add_argument("--topics", "-t", nargs="+", 
                        default=["calculus__differentiate", "algebra__polynomial_roots", "numbers__list_prime_factors"],
                        help="Topics to generate prompts for")
    parser.add_argument("--variants", "-v", nargs="+", 
                        default=["baseline", "concise", "high_school", "graduate", "llm"],
                        help="Variants to generate prompts for")
    parser.add_argument("--examples", "-e", type=int, default=5,
                        help="Number of example problems to include")
    args = parser.parse_args()
    
    # Convert relative paths to absolute paths
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.output_dir)
    if not os.path.isabs(args.templates_dir):
        args.templates_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.templates_dir)
    
    # Set up logging
    logger = setup_logger("generate_prompts_with_llm")
    logger.info(f"Starting prompt generation with {args.model}")
    
    # Ensure output directory exists
    ensure_dir(args.output_dir)
    
    # Process each topic
    for topic in args.topics:
        logger.info(f"Processing topic: {topic}")
        
        # Load example problems
        problems = load_problems(args.dataset, topic, args.examples)
        if not problems:
            logger.error(f"No problems found for topic: {topic}")
            continue
        
        # Format problems
        problems_text = format_problems(problems)
        
        # Process each variant
        for variant in args.variants:
            logger.info(f"  Generating {variant} prompt for {topic}")
            
            # Ensure variant directory exists
            variant_dir = os.path.join(args.output_dir, variant)
            ensure_dir(variant_dir)
            
            # Path to template
            template_path = os.path.join(args.templates_dir, f"{variant}.txt")
            if not os.path.exists(template_path):
                logger.error(f"Template not found: {template_path}")
                continue
            
            # Generate prompt
            prompt = generate_prompt(template_path, topic, problems_text)
            if not prompt:
                logger.error(f"Failed to generate prompt for {variant}/{topic}")
                continue
            
            # Output path for generated instructions
            output_path = os.path.join(variant_dir, f"{topic}_composed.txt")
            
            # Run LLM to generate instructions
            try:
                instructions = run_llm_to_generate_instructions(prompt, output_path, args.model)
                logger.info(f"  Generated {len(instructions)} characters of instructions")
            except Exception as e:
                logger.error(f"  Error generating instructions: {e}")
    
    logger.info("Prompt generation completed")

if __name__ == "__main__":
    # Ensure the scripts directory exists
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    main() 