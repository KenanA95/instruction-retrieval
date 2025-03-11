#!/usr/bin/env python3
import os
import sys
import argparse
import json
import re
from pathlib import Path

# Add the src directory to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.utils.config import Config
from src.utils.io import ensure_dir, load_json
from src.utils.logging import setup_logger
from src.models.llm_adapter import LLMAdapter
from src.models.answer_extraction import AnswerExtractor

def load_test_problems(dataset_path, topics=None, num_problems=3):
    """Load a few test problems from the dataset."""
    import pandas as pd
    
    # Load the dataset
    df = pd.read_csv(dataset_path, sep='\t')
    
    # Filter by topics if provided
    if topics:
        df = df[df['topic'].isin(topics)]
    
    # Sample problems from each topic
    topics = df['topic'].unique()
    test_problems = []
    
    for topic in topics:
        topic_df = df[df['topic'] == topic]
        if len(topic_df) > 0:
            # Sample problems
            samples = topic_df.sample(min(num_problems, len(topic_df)))
            test_problems.extend(samples.to_dict('records'))
    
    return test_problems

def build_prompt(variant, topic, question, answer=None):
    """Build a prompt for a specific variant."""
    # Load the instruction file
    instruction_file = os.path.join(project_root, "src/data/instructions", variant.replace("instructions_", ""), f"{topic}_composed.txt")
    
    if not os.path.exists(instruction_file):
        raise FileNotFoundError(f"Instruction file not found: {instruction_file}")
    
    with open(instruction_file, 'r') as f:
        instructions = f.read().strip()
    
    # Add CoT examples if needed
    cot_text = ""
    if "few_shot" in variant:
        # This is just a placeholder - in a real implementation, you would load CoT examples
        cot_text = "Here are some examples:\n\n"
    
    # Build the prompt
    prompt = f"{instructions}\n\n{cot_text}Question: {question}\n\n"
    
    # Add the required output format for easier parsing - only if it's not already in the instructions
    if "You must strictly follow the following output format:" not in prompt:
        prompt += "You must strictly follow the following output format:\nReasoning: [Your reasoning here]\nFinal Answer: [Your final answer here]"
    
    # For few-shot with answer, add "Answer: " at the end
    if variant.startswith("few_shot") and answer is None:
        prompt += "Answer: "
    
    return prompt

def run_inference(problems, model_name, extract_model_name, variants):
    """Run inference on test problems with different variants."""
    # Initialize models
    print(f"Initializing inference model: {model_name}")
    llm = LLMAdapter(model_name=model_name, max_new_tokens=512, temperature=0.1)
    
    # Make sure we're using the extraction model
    print(f"Initializing answer extractor with model: {extract_model_name}")
    try:
        extractor = AnswerExtractor(extract_model_name=extract_model_name)
    except Exception as e:
        print(f"Error initializing answer extractor: {e}")
        print("Falling back to pattern-based extraction only")
        extractor = AnswerExtractor(extract_model_name=None)
    
    results = []
    
    # Process each problem
    for problem in problems:
        topic = problem['topic']
        question = problem['question']
        answer = problem['answer']
        
        problem_results = {
            'topic': topic,
            'question': question,
            'answer': answer,
            'variants': {}
        }
        
        # Process each variant
        for variant in variants:
            print(f"Processing {variant} for {topic}")
            
            try:
                # Build the prompt
                prompt = build_prompt(variant, topic, question, answer)
                
                # Generate response
                response = llm.generate(prompt)
                
                # Extract answer
                print(f"Extracting answer for {variant}...")
                try:
                    extracted_answer = extractor.extract_answer(response, question)
                    print(f"Raw extracted answer: {extracted_answer}")
                except Exception as e:
                    print(f"Error in answer extraction: {e}")
                    # Fall back to basic pattern matching
                    extracted_answer = extract_with_basic_patterns(response)
                
                # Additional post-processing for common patterns
                if not extracted_answer or extracted_answer.startswith("<|") or "extract" in extracted_answer.lower():
                    # Try to extract using direct regex patterns
                    print("Using direct regex patterns on the response...")
                    extracted_answer = extract_with_basic_patterns(response)
                
                # Store results
                problem_results['variants'][variant] = {
                    'prompt': prompt,
                    'response': response,
                    'extracted_answer': extracted_answer,
                    'correct': normalize_answer(extracted_answer) == normalize_answer(answer)
                }
                
                print(f"  Final extracted answer: {extracted_answer}")
                print(f"  Correct answer: {answer}")
                print(f"  Match: {normalize_answer(extracted_answer) == normalize_answer(answer)}")
                
            except Exception as e:
                print(f"Error processing {variant} for {topic}: {e}")
                problem_results['variants'][variant] = {
                    'error': str(e)
                }
        
        results.append(problem_results)
    
    return results

def extract_with_basic_patterns(response):
    """Extract answer using basic regex patterns."""
    # Try to match various patterns in the response
    answer_patterns = [
        # Extract boxed answers
        r'\\boxed{([^}]+)}',
        # Final answer format
        r'Final Answer:?\s*([^\n]+)',
        # The answer is format
        r'[Tt]he answer is\s*([^\n\.]+)',
        # Result is format
        r'[Tt]he result is\s*([^\n\.]+)',
        # Last equation in the response
        r'=\s*([^=\n]+)(?:\n|$)',
        # Last line with numbers or variables
        r'(?:^|\n)([^a-z\n]*?[\d\*\+\-\/\^\s\(\)xy]+[^a-z\n]*)(?:\n|$)'
    ]
    
    for pattern in answer_patterns:
        matches = re.findall(pattern, response, re.MULTILINE)
        if matches:
            extracted_answer = matches[-1].strip()
            print(f"Extracted with pattern: {extracted_answer}")
            return extracted_answer
    
    # If all else fails, try to get the last line
    lines = [line for line in response.strip().split('\n') if line.strip()]
    if lines:
        return lines[-1].strip()
    
    return ""

def normalize_answer(answer):
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    
    # Convert to string and clean up whitespace
    answer = str(answer).strip().lower()
    
    # Replace multiple spaces with a single space
    answer = re.sub(r'\s+', ' ', answer)
    
    # Remove any text markers like "final answer:" or "the answer is"
    answer = re.sub(r'^final answer:?\s*', '', answer, flags=re.IGNORECASE)
    answer = re.sub(r'^the answer is\s*', '', answer, flags=re.IGNORECASE)
    
    # Remove special formatting
    answer = re.sub(r'\\boxed{\s*(.*?)\s*}', r'\1', answer)
    
    # Remove unnecessary parentheses around the entire answer
    if answer.startswith('(') and answer.endswith(')'):
        answer = answer[1:-1].strip()
    
    return answer

def main():
    parser = argparse.ArgumentParser(description="Run a sanity check on a few problems")
    parser.add_argument("--dataset", "-d", default="/shared/3/projects/instruction-retrieval/mathematics_dataset/processed/easy_100.tsv",
                        help="Path to the dataset file")
    parser.add_argument("--model", "-m", default="meta-llama/Llama-3.2-3B-Instruct",
                        help="Model to use for inference")
    parser.add_argument("--extract-model", "-e", default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Model to use for answer extraction")
    parser.add_argument("--num-problems", "-n", type=int, default=1,
                        help="Number of problems per topic to test")
    parser.add_argument("--variants", "-v", nargs="+", 
                        default=["baseline", "instructions_baseline"],
                        help="Variants to test")
    parser.add_argument("--topics", "-t", nargs="+", 
                        default=["calculus__differentiate", "algebra__polynomial_roots", "numbers__list_prime_factors"],
                        help="Topics to test")
    parser.add_argument("--output-dir", "-o", default="results/sanity_check",
                        help="Output directory for results")
    args = parser.parse_args()
    
    # Convert relative paths to absolute paths
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(project_root, args.output_dir)
    
    # Set up logging
    logger = setup_logger("sanity_check")
    logger.info(f"Starting sanity check with model {args.model}")
    
    # Ensure output directory exists
    ensure_dir(args.output_dir)
    
    # Load test problems
    logger.info(f"Loading test problems from {args.dataset}")
    problems = load_test_problems(args.dataset, args.topics, args.num_problems)
    logger.info(f"Loaded {len(problems)} test problems")
    
    # Run inference
    logger.info(f"Running inference with variants: {args.variants}")
    results = run_inference(problems, args.model, args.extract_model, args.variants)
    
    # Save results
    output_file = os.path.join(args.output_dir, "sanity_check_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {output_file}")
    
    # Print summary
    correct_counts = {variant: 0 for variant in args.variants}
    total_problems = len(problems)
    
    for problem in results:
        for variant, result in problem['variants'].items():
            if result.get('correct', False):
                correct_counts[variant] += 1
    
    logger.info("Summary:")
    for variant, count in correct_counts.items():
        accuracy = count / total_problems if total_problems > 0 else 0
        logger.info(f"  {variant}: {count}/{total_problems} correct ({accuracy:.2%})")
    
    # Run unit tests
    logger.info("Running answer extraction tests...")
    import subprocess
    test_result = subprocess.run(["python", "-m", "tests.test_answer_extraction"], capture_output=True)
    
    if test_result.returncode == 0:
        logger.info("All tests passed!")
    else:
        logger.error("Some tests failed:")
        logger.error(test_result.stderr.decode())
    
    logger.info("Sanity check completed")

if __name__ == "__main__":
    # Ensure the scripts directory exists
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    main() 