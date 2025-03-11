import os
import json
import pandas as pd
from pathlib import Path
from ...utils.io import load_json, save_json, ensure_dir

class PromptManager:
    def __init__(self, base_dir=None, prompt_dirs=None):
        self.base_dir = base_dir or os.path.join(os.path.dirname(__file__), "templates")
        ensure_dir(self.base_dir)
        
        # Use configured prompt directories if provided
        self.prompt_dirs = prompt_dirs or {}
        
        # Print available prompt directories for debugging
        print(f"Available prompt directories:")
        for variant, path in self.prompt_dirs.items():
            print(f"  {variant}: {path}")
            if not os.path.exists(path):
                print(f"  WARNING: Path does not exist: {path}")
        
        # Ensure prompt directories exist
        for prompt_dir in self.prompt_dirs.values():
            ensure_dir(prompt_dir)
    
    def load_instruction(self, variant, topic=None):
        """Load an instruction prompt for a variant and optionally topic."""
        # Get the base variant by carefully removing prefixes and suffixes
        base_variant = variant
        
        # Handle instruction prefix
        if base_variant.startswith("instructions_"):
            base_variant = base_variant[len("instructions_"):]
            
        # Handle few_shot_cot suffix
        if "_few_shot_cot" in base_variant:
            base_variant = base_variant.replace("_few_shot_cot", "")
        elif base_variant.endswith("_few_shot") or base_variant == "few_shot_cot":
            base_variant = base_variant.replace("few_shot_cot", "").replace("_few_shot", "")
            
        # Clean up any trailing underscores
        base_variant = base_variant.rstrip('_')
        
        # If base_variant is empty after replacements, use "baseline"
        if not base_variant:
            base_variant = "baseline"
            
        variant_dir = self.prompt_dirs.get(base_variant)
        
        if not variant_dir:
            raise ValueError(f"Unknown variant: {base_variant}")
        
        if topic:
            # Try to load topic-specific prompt with _composed.txt suffix
            prompt_path = os.path.join(variant_dir, f"{topic}_composed.txt")
            if os.path.exists(prompt_path):
                with open(prompt_path, 'r') as f:
                    return f.read().strip()
                
            # Also try without the _composed suffix (for flexibility)
            simple_prompt_path = os.path.join(variant_dir, f"{topic}.txt")
            if os.path.exists(simple_prompt_path):
                with open(simple_prompt_path, 'r') as f:
                    return f.read().strip()
        
        # Fall back to general prompt for the variant
        prompt_path = os.path.join(variant_dir, "general.txt")
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r') as f:
                return f.read().strip()
        
        raise FileNotFoundError(f"No prompt found for {base_variant}" + (f"/{topic}" if topic else ""))
    
    def format_cot_examples(self, examples):
        """Format Chain-of-Thought examples as text."""
        formatted_examples = []
        
        for _, example in examples.iterrows():
            formatted = f"Question: {example['question']}\n\n"
            formatted += f"Solution:\n{example['cot_solution']}\n\n"
            formatted += f"Answer: {example['answer']}\n\n"
            formatted += "-" * 50 + "\n"
            formatted_examples.append(formatted)
        
        return "\n".join(formatted_examples)
    
    def build_prompt(self, variant, question, answer=None, topic=None, cot_examples=None):
        """Build a complete prompt for a variant and question."""
        instruction = ""
        
        # Skip loading instructions for pure few_shot_cot
        if variant != "few_shot_cot":
            try:
                instruction = self.load_instruction(variant, topic)
                instruction += "\n\n"
            except FileNotFoundError as e:
                print(f"Warning: {str(e)}")
                # Don't try to fall back to baseline, just propagate the error
                raise
        
        # Add CoT examples if provided and this is a few_shot variant
        cot_text = ""
        if "few_shot" in variant and cot_examples is not None and not cot_examples.empty:
            cot_text = "Here are some examples:\n\n" + self.format_cot_examples(cot_examples) + "\n"
        
        # Add the question
        prompt = f"{instruction}{cot_text}Question: {question}\n\n"
        
        # For few-shot with answer, add "Answer: " at the end
        if "few_shot" in variant and answer is None:
            prompt += "Answer: "
        
        return prompt
    
    def generate_prompt_variants(self, dataset, output_dir, variants=None, num_cot_examples=3):
        """Generate all prompt variants for a dataset."""
        if variants is None:
            variants = list(self.prompt_dirs.keys())  # Use all available variants
        
        # Ensure output directory exists
        ensure_dir(output_dir)
        
        # For each variant, generate prompts
        for variant in variants:
            variant_dir = os.path.join(output_dir, variant)
            ensure_dir(variant_dir)
            
            prompts = []
            
            for _, row in dataset.data.iterrows():
                topic = row['topic']
                question = row['question']
                answer = row['answer']
                
                # Get CoT examples if needed
                cot_examples = None
                if "few_shot" in variant:
                    try:
                        cot_examples = dataset.get_cot_examples(topic, num_cot_examples)
                    except FileNotFoundError:
                        print(f"Warning: No CoT examples found for {topic}")
                
                try:
                    # Build the prompt
                    prompt = self.build_prompt(
                        variant=variant,
                        question=question,
                        answer=answer,
                        topic=topic,
                        cot_examples=cot_examples
                    )
                    
                    prompts.append({
                        "topic": topic,
                        "question": question,
                        "answer": answer,
                        "prompt": prompt
                    })
                except FileNotFoundError as e:
                    print(f"Warning: Skipping {variant}/{topic} - {str(e)}")
                    continue
            
            if prompts:  # Only save if we have prompts
                # Save prompts to file
                output_file = os.path.join(variant_dir, f"{variant}.json")
                save_json(prompts, output_file)
                print(f"Saved {len(prompts)} {variant} prompts to {output_file}")
        
        return output_dir 