import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Just in case set the cache directory again
os.environ["TRANSFORMERS_CACHE"] = "/shared/3/projects/instruction-retrieval/.cache"
os.environ["HF_HOME"] = "/shared/3/projects/instruction-retrieval/.cache"

class LlamaInference:
    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct"):
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer.padding_side = 'left'  # Set padding side to left for decoder-only models
        print("Model loaded successfully")
    
    def format_prompt(self, query):
        # Format the prompt according to Llama 3.2 Instruct format with output format instructions
        instruction = """You must strictly follow the following output format:
Reasoning: [Your reasoning here]
Final Answer: [Your final answer here]"""
        
        return f"<|begin_of_text|><|user|>\n{query}\n\n{instruction}<|end_of_text|>\n<|assistant|>"
    
    def generate_response(self, query, max_new_tokens=2048, temperature=0.7):
        formatted_prompt = self.format_prompt(query)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract just the assistant's response
        assistant_response = response.split("<|assistant|>")[-1].strip()
        return assistant_response

    def generate_responses_batch(self, queries, max_new_tokens=2048, temperature=0.7):
        # Format all queries with the required output format
        formatted_queries = [self.format_prompt(query) for query in queries]
        
        # Tokenize all queries at once with appropriate padding and truncation
        inputs = self.tokenizer(formatted_queries, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode the batch of responses
        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        
        # Extract just the assistant's responses
        assistant_responses = [response.split("<|assistant|>")[-1].strip() for response in responses]
        return assistant_responses