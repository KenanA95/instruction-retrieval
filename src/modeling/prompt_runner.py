import os
import json
import pandas as pd
from pathlib import Path
import glob
import re

from ..utils.io import load_json, save_json, ensure_dir

class PromptRunner:
    """
    Manager for handling prompt templates and generating experiment prompts.
    """
    
    def __init__(self, prompt_dirs=None):
        """
        Initialize the prompt runner.
        
        Args:
            prompt_dirs: Dictionary mapping variant names to prompt directory paths
        """
        self.prompt_dirs = prompt_dirs or {}
        self.templates = {}
        
        # Load templates from directories
        self._load_templates()
    
    def _load_templates(self):
        """Load templates from prompt directories."""
        for variant, prompt_dir in self.prompt_dirs.items():
            if not os.path.exists(prompt_dir):
                print(f"Warning: Prompt directory not found: {prompt_dir}")
                continue
            
            template_files = glob.glob(os.path.join(prompt_dir, "*.json"))
            if not template_files:
                print(f"Warning: No template files found in {prompt_dir}")
                continue
            
            # Load templates for this variant
            variant_templates = {}
            for template_file in template_files:
                try:
                    template_name = os.path.basename(template_file).replace('.json', '')
                    template_data = load_json(template_file)
                    variant_templates[template_name] = template_data
                except Exception as e:
                    print(f"Error loading template file {template_file}: {e}")
            
            if variant_templates:
                self.templates[variant] = variant_templates
                print(f"Loaded {len(variant_templates)} templates for variant {variant}")
            else:
                print(f"Warning: No valid templates loaded for variant {variant}")
    
    def _fill_template(self, template, **kwargs):
        """
        Fill a template with provided variables.
        
        Args:
            template: Template string
            **kwargs: Variables to fill in the template
            
        Returns:
            Filled template string
        """
        filled = template
        
        # Replace template variables with provided values
        for key, value in kwargs.items():
            placeholder = f"{{{key}}}"
            filled = filled.replace(placeholder, str(value))
        
        # Check for any remaining unfilled placeholders
        remaining_placeholders = re.findall(r"\{([^}]+)\}", filled)
        if remaining_placeholders:
            print(f"Warning: Unfilled placeholders in template: {remaining_placeholders}")
        
        return filled
    
    def _format_with_markdown(self, variant, example_data):
        """
        Format prompt components with Markdown for better structure.
        
        Args:
            variant: Variant name
            example_data: Example data dictionary
            
        Returns:
            Structured prompt with Markdown formatting
        """
        sections = []
        
        # Add task header
        sections.append("# Instruction Retrieval Task")
        
        # Add domain-specific instructions if available
        if example_data.get('instructions', ''):
            sections.append("## Domain Instructions")
            sections.append(example_data['instructions'])
        
        # Add retrieved context if available
        if example_data.get('use_retrieval', False) and example_data.get('retrieved_context', ''):
            sections.append("## Retrieved Context")
            sections.append(example_data['retrieved_context'])
        
        # Add Chain-of-Thought examples if available
        if example_data.get('use_cot', False) and example_data.get('cot_examples_text', ''):
            sections.append("## Chain-of-Thought Examples")
            sections.append(example_data['cot_examples_text'])
        
        # Add the problem to solve at the end
        if example_data.get('question', ''):
            sections.append("## Problem to Solve")
            sections.append(example_data['question'])
            # Add a clear indicator that the model should answer this problem
            sections.append("\nProvide a step-by-step solution to this problem.")
        
        # Join all sections with double newlines
        return "\n\n".join(sections)
    
    def generate_prompt(self, variant, example, template_type='default'):
        """
        Generate a prompt for a specific variant and example.
        
        Args:
            variant: Variant name
            example: Example data dictionary
            template_type: Type of template to use
            
        Returns:
            Dictionary with prompt data
        """
        if variant not in self.templates:
            raise ValueError(f"No templates loaded for variant: {variant}")
        
        if template_type not in self.templates[variant]:
            # Fall back to default template if specific type not found
            template_type = next(iter(self.templates[variant].keys()))
            print(f"Template type {template_type} not found, using {template_type}")
        
        template = self.templates[variant][template_type]
        
        # Get the template string
        if isinstance(template, dict) and 'template' in template:
            template_str = template['template']
        else:
            template_str = template
        
        # Fill the template with example data
        try:
            # Fill template placeholders
            filled_prompt = self._fill_template(template_str, **example)
            
            # Optionally apply Markdown formatting
            # Check if this is a simple template or if we should restructure it
            if variant.startswith("instructions_") or "few_shot_cot" in variant or example.get('use_retrieval', False):
                # For complex prompts with instructions, CoT, or retrieval, use Markdown structure
                formatted_prompt = self._format_with_markdown(variant, example)
            else:
                # For simpler prompts, keep the original format
                formatted_prompt = filled_prompt
            
            # Create prompt data
            prompt_data = {
                'variant': variant,
                'template_type': template_type,
                'prompt': formatted_prompt,
                'question': example.get('question', ''),
                'answer': example.get('answer', ''),
                'topic': example.get('topic', '')
            }
            
            return prompt_data
            
        except Exception as e:
            print(f"Error generating prompt for variant {variant}: {e}")
            return None
    
    def generate_prompt_variants(self, dataset, output_dir, variants=None, num_cot_examples=0, use_retrieval=False):
        """
        Generate prompt variants for all examples in a dataset.
        
        Args:
            dataset: Dataset object with examples
            output_dir: Directory to save generated prompts
            variants: List of variant names to generate (defaults to all loaded variants)
            num_cot_examples: Number of Chain-of-Thought examples to include (if supported by template)
            use_retrieval: Whether to include retrieved context (if available in examples)
            
        Returns:
            Dictionary mapping variant names to lists of generated prompts
        """
        ensure_dir(output_dir)
        
        # Determine which variants to generate
        variants = variants or list(self.templates.keys())
        if not variants:
            print("No variants to generate")
            return {}
        
        # Generate prompts for each variant
        all_prompts = {}
        
        for variant in variants:
            if variant not in self.templates:
                print(f"Warning: No templates loaded for variant {variant}")
                continue
            
            variant_prompts = []
            
            for example in dataset.examples:
                example_data = dict(example)  # Create a copy we can modify
                
                # Add CoT examples if needed
                if num_cot_examples > 0 and 'cot_examples' in example:
                    cot_examples = example.get('cot_examples', [])
                    example_data['cot_examples_text'] = "\n\n".join(cot_examples[:num_cot_examples])
                    example_data['use_cot'] = True
                else:
                    example_data['cot_examples_text'] = ""
                    example_data['use_cot'] = False
                
                # Add retrieval context if needed
                if use_retrieval and 'context' in example:
                    if isinstance(example['context'], list):
                        # Join list of context items
                        example_data['retrieved_context'] = "\n\n".join([
                            str(ctx) if isinstance(ctx, str) else json.dumps(ctx)
                            for ctx in example['context']
                        ])
                    else:
                        example_data['retrieved_context'] = str(example['context'])
                    
                    example_data['use_retrieval'] = True
                else:
                    example_data['retrieved_context'] = ""
                    example_data['use_retrieval'] = False
                
                # Generate prompt for this example
                prompt_data = self.generate_prompt(variant, example_data)
                if prompt_data:
                    variant_prompts.append(prompt_data)
            
            # Save variant prompts
            if variant_prompts:
                variant_file = os.path.join(output_dir, f"{variant}.json")
                save_json(variant_prompts, variant_file)
                print(f"Generated {len(variant_prompts)} prompts for variant {variant}")
                
                all_prompts[variant] = variant_prompts
            else:
                print(f"Warning: No prompts generated for variant {variant}")
        
        return all_prompts 