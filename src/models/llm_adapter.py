import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm

class LLMAdapter:
    def __init__(self, model_name, device=None, max_new_tokens=512, temperature=0.1):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        """Load the model and tokenizer."""
        print(f"Loading model {self.model_name} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Set padding token if it doesn't exist (needed for batch processing)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            print(f"Using eos_token as pad_token for {self.model_name}")
            
        # Set padding side to left for decoder-only models 
        self.tokenizer.padding_side = 'left'
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        print(f"Model loaded successfully.")
        return self
    
    def format_prompt(self, prompt):
        """Format the prompt for the specific model."""
        # Check if this is a Llama model
        if "llama" in self.model_name.lower():
            # Check if the prompt already contains our output format
            has_output_format = "You must strictly follow the following output format:" in prompt
            output_format = ""
            
            # Only add the output format if it's not already there
            if not has_output_format:
                output_format = "\n\nYou must strictly follow the following output format:\nReasoning: [Your reasoning here]\nFinal Answer: [Your final answer here]"
            
            # Format for Llama chat models
            if "instruct" in self.model_name.lower():
                # For Llama 3 Instruct models
                formatted_prompt = f"<|begin_of_text|><|user|>\n{prompt}{output_format}<|end_of_turn|>\n<|assistant|>\n"
                return formatted_prompt
            else:
                # For other Llama models
                return prompt + output_format
        
        # Default format for other models
        return prompt
    
    def format_chat_prompt(self, messages):
        """Format a list of chat messages for the model."""
        if "llama" in self.model_name.lower() and "instruct" in self.model_name.lower():
            # Format for Llama 3 Instruct models
            formatted_prompt = "<|begin_of_text|>"
            
            for message in messages:
                role = message["role"]
                content = message["content"]
                
                if role == "system":
                    # Add system message
                    formatted_prompt += f"<|system|>\n{content}<|end_of_turn|>\n"
                elif role == "user":
                    # Add user message
                    formatted_prompt += f"<|user|>\n{content}<|end_of_turn|>\n"
                elif role == "assistant":
                    # Add assistant message
                    formatted_prompt += f"<|assistant|>\n{content}<|end_of_turn|>\n"
            
            # Add final assistant token for generation
            if not formatted_prompt.endswith("<|assistant|>\n"):
                formatted_prompt += "<|assistant|>\n"
            
            return formatted_prompt
        else:
            # For other models, just concatenate the messages
            formatted_prompt = ""
            for message in messages:
                role = message["role"]
                content = message["content"]
                
                if role == "system":
                    formatted_prompt += f"System: {content}\n\n"
                elif role == "user":
                    formatted_prompt += f"User: {content}\n\n"
                elif role == "assistant":
                    formatted_prompt += f"Assistant: {content}\n\n"
            
            formatted_prompt += "Assistant: "
            return formatted_prompt
    
    def generate(self, prompt, max_new_tokens=None, temperature=None):
        """Generate a response for a single prompt."""
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        max_new_tokens = max_new_tokens or self.max_new_tokens
        temperature = temperature or self.temperature
        
        # Format the prompt for the model
        formatted_prompt = self.format_prompt(prompt)
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from the response
        if response.startswith(formatted_prompt):
            response = response[len(formatted_prompt):]
        
        return response.strip()
    
    def batch_generate(self, prompts, batch_size=8, max_new_tokens=None, temperature=None):
        """Generate responses for a batch of prompts."""
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        max_new_tokens = max_new_tokens or self.max_new_tokens
        temperature = temperature or self.temperature
        
        responses = []
        
        # Process in batches
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating responses"):
            batch_prompts = prompts[i:i+batch_size]
            
            # Format each prompt
            formatted_prompts = [self.format_prompt(prompt) for prompt in batch_prompts]
            
            # Tokenize batch
            batch_inputs = self.tokenizer(formatted_prompts, padding=True, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                batch_outputs = self.model.generate(
                    **batch_inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode outputs
            for j, output in enumerate(batch_outputs):
                response = self.tokenizer.decode(output, skip_special_tokens=True)
                
                # Remove the prompt from the response
                formatted_prompt = formatted_prompts[j]
                if response.startswith(formatted_prompt):
                    response = response[len(formatted_prompt):]
                
                responses.append(response.strip())
        
        return responses
    
    def get_clean_model_name(self):
        """Get a clean version of the model name for file naming."""
        return self.model_name.replace("/", "_").replace("-", "_").lower() 