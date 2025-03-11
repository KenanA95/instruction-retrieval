import os
import json
import pandas as pd
from tqdm import tqdm
from .llm_adapter import LLMAdapter
from .answer_extraction import AnswerExtractor
from ..utils.io import save_json, load_json, ensure_dir

class InferenceRunner:
    def __init__(self, model_name, extract_model_name=None, batch_size=8):
        self.model = LLMAdapter(model_name, max_new_tokens=4096, temperature=0.1)
        self.extractor = AnswerExtractor(extract_model_name)
        self.batch_size = batch_size
    
    def run_inference(self, prompts_data, output_dir=None, experiment_name=None):
        """Run inference on a list of prompts."""
        # Extract prompts from data
        if isinstance(prompts_data, str) and os.path.exists(prompts_data):
            # Load from file
            prompts_data = load_json(prompts_data)
        
        # Extract prompts, questions, and answers
        prompts = [item["prompt"] for item in prompts_data]
        questions = [item["question"] for item in prompts_data]
        answers = [item["answer"] for item in prompts_data]
        topics = [item.get("topic", "") for item in prompts_data]
        
        print(f"Running inference on {len(prompts)} prompts...")
        
        # Generate responses
        responses = self.model.batch_generate(prompts, batch_size=self.batch_size)
        
        # Extract answers
        extracted_answers = self.extractor.batch_extract(responses, questions)
        
        # Combine results
        results = []
        for i in range(len(prompts)):
            results.append({
                "topic": topics[i],
                "question": questions[i],
                "answer": answers[i],
                "raw_response": responses[i],
                "extracted_answer": extracted_answers[i]
            })
        
        # Save results if output_dir is provided
        if output_dir:
            ensure_dir(output_dir)
            
            # Generate filename
            model_name_clean = self.model.get_clean_model_name()
            filename = f"{experiment_name or 'inference'}_{model_name_clean}.json"
            output_file = os.path.join(output_dir, filename)
            
            # Save results
            save_json(results, output_file)
            print(f"Saved results to {output_file}")
        
        return results
    
    def run_variant_inference(self, variants_dir, output_dir, experiment_name=None):
        """Run inference on all prompt variants in a directory."""
        # Find all variant files
        variant_files = []
        for root, dirs, files in os.walk(variants_dir):
            for file in files:
                if file.endswith('.json') and not file == 'all_variants.json':
                    variant_files.append(os.path.join(root, file))
        
        print(f"Found {len(variant_files)} variant files to process")
        
        # Process each variant file
        all_results = {}
        for variant_file in variant_files:
            try:
                variant_name = os.path.basename(os.path.dirname(variant_file))
                print(f"\nProcessing variant: {variant_name}")
                
                # Run inference
                results = self.run_inference(
                    prompts_data=variant_file,
                    output_dir=output_dir,
                    experiment_name=variant_name
                )
                
                # Store results
                all_results[variant_name] = results
                
                print(f"Completed processing for variant: {variant_name}")
            except Exception as e:
                print(f"Error processing {variant_file}: {e}")
        
        # Create a combined results file
        if all_results and output_dir:
            print("\nCreating combined results file...")
            self._create_combined_results(all_results, output_dir, experiment_name)
        
        return all_results
    
    def _create_combined_results(self, results_dict, output_dir, experiment_name=None):
        """Create a combined results file from multiple variant results."""
        if not results_dict:
            print("No results to combine")
            return []
            
        combined_results = []
        
        # Get a list of all topics and questions
        all_topics_questions = set()
        for variant_name, results in results_dict.items():
            if not results:  # Skip empty results
                print(f"Warning: No results for variant {variant_name}")
                continue
                
            for item in results:
                if 'topic' in item and 'question' in item:
                    all_topics_questions.add((item['topic'], item['question']))
        
        if not all_topics_questions:
            print("No valid topic-question pairs found")
            return []
            
        print(f"Found {len(all_topics_questions)} unique topic-question pairs to combine")
        
        # For each topic-question pair, collect results from all variants
        for topic, question in all_topics_questions:
            row = {'topic': topic, 'question': question}
            
            # Find the answer (should be the same across all variants)
            answer_found = False
            for variant_name, results in results_dict.items():
                if not results:
                    continue
                    
                for item in results:
                    if item.get('topic') == topic and item.get('question') == question and 'answer' in item:
                        row['answer'] = item['answer']
                        answer_found = True
                        break
                if answer_found:
                    break
            
            if not answer_found:
                print(f"Warning: No answer found for {topic}/{question}")
                row['answer'] = ""
            
            # Collect results from each variant
            for variant_name, results in results_dict.items():
                if not results:
                    continue
                    
                result_found = False
                for item in results:
                    if item.get('topic') == topic and item.get('question') == question:
                        row[f'{variant_name}_raw_response'] = item.get('raw_response', '')
                        row[f'{variant_name}_extracted_answer'] = item.get('extracted_answer', '')
                        result_found = True
                        break
                
                if not result_found:
                    # Set empty values if this variant didn't have results for this topic/question
                    row[f'{variant_name}_raw_response'] = ''
                    row[f'{variant_name}_extracted_answer'] = ''
            
            combined_results.append(row)
        
        if not combined_results:
            print("No combined results were created")
            return []
            
        # Save combined results
        model_name_clean = self.model.get_clean_model_name()
        combined_file = os.path.join(
            output_dir, 
            f"all_variants_{experiment_name or 'experiment'}_{model_name_clean}.json"
        )
        
        save_json(combined_results, combined_file)
        print(f"Saved combined results to {combined_file}")
        
        return combined_results 