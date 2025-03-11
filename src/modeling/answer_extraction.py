import re
from .llm_adapter import LLMAdapter

class AnswerExtractor:
    def __init__(self, extract_model_name=None):
        self.extract_model = None
        if extract_model_name:
            try:
                self.extract_model = LLMAdapter(extract_model_name, max_new_tokens=256)
                self.extract_model.load_model()
                print(f"Initialized extraction model: {extract_model_name}")
            except Exception as e:
                print(f"Error initializing extraction model: {e}")
                self.extract_model = None
    
    def extract_answer(self, response, question=None, task_type="math"):
        """Extract the answer from a model response."""
        return self.extract_math_answer(response, question)
    
    def extract_math_answer(self, response, question=None):
        """Extract a mathematical answer from a response using only expected format patterns."""
        # Clean up response
        response = self._clean_special_tokens(response)
        
        # 1. LaTeX boxed answers (highest confidence - explicit final answer marker)
        boxed_match = re.search(r'\\boxed{([^}]+)}', response)
        if boxed_match:
            return self._clean_extracted_answer(boxed_match.group(1))
        
        # 2. "Final Answer:" format (our explicit format marker)
        final_answer_match = re.search(r'[Ff]inal [Aa]nswer:?\s*([^\n\.]+)', response)
        if final_answer_match:
            extracted = final_answer_match.group(1).strip()
            if extracted and "[Your final answer here]" not in extracted:
                return self._clean_extracted_answer(extracted)
        
        # 3. Simple "Answer:" format
        answer_match = re.search(r'(?:^|\n)[Aa]nswer:?\s*([^\n\.]+)', response)
        if answer_match:
            extracted = answer_match.group(1).strip()
            if extracted and "[Your final answer here]" not in extracted:
                return self._clean_extracted_answer(extracted)
        
        # If we don't find our expected formats, use LLM extraction
        if self.extract_model:
            print(f"No standard format found, using extraction model")
            return self._extract_with_model(response)
        
        print("No answer could be extracted")
        return ""
    
    def _clean_special_tokens(self, response):
        """Remove special tokens from response."""
        response = re.sub(r'<\|[^|]+\|>', '', response)  # Remove LLM tokens
        return response
    
    def _clean_extracted_answer(self, answer_text):
        """Clean the extracted answer to standardize mathematical notation."""
        if not answer_text:
            return ""
        
        # Remove LaTeX formatting if present
        answer_text = answer_text.replace('\\boxed{', '').replace('}', '')
        answer_text = re.sub(r'\$\\boxed{(.*?)}\$', r'\1', answer_text)
        answer_text = re.sub(r'\$([^$]+)\$', r'\1', answer_text)
        
        # Fix mathematical notation
        # Convert "x^2" to "x**2"
        answer_text = re.sub(r'(\w)\^(\d)', r'\1**\2', answer_text)
        # Add missing multiplication: "2x" → "2*x"
        answer_text = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', answer_text)
        
        return answer_text.strip()
    
    def _extract_with_model(self, response, question=None):
        """Use the LLM to extract just the final answer."""
        try:
            system_prompt = """You are a mathematical answer extractor. Your job is to identify and extract ONLY the final answer from a response to a math problem.

Extract ONLY the mathematical expression that represents the final answer.
Do not include any explanation, working, or additional text.
Return only the expression in a standard mathematical format."""
            
            user_prompt = f"""From this response, extract ONLY the final answer as a mathematical expression:

{response}"""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            prompt = self.extract_model.format_chat_prompt(messages)
            extracted = self.extract_model.generate(prompt, max_new_tokens=64, temperature=0.1)
            
            # Clean the extracted answer
            extracted = extracted.strip()
            
            # If the extracted text is too long, it's probably not a clean extraction
            if len(extracted.split()) > 10:
                # Try to extract just the first line
                first_line = extracted.split('\n')[0].strip()
                return self._clean_extracted_answer(first_line)
            
            return self._clean_extracted_answer(extracted)
            
        except Exception as e:
            print(f"Error using extraction model: {e}")
            return ""
    
    def batch_extract(self, responses, questions=None):
        """Extract answers from a batch of responses."""
        extracted_answers = []
        
        for i, response in enumerate(responses):
            question = questions[i] if questions else None
            extracted = self.extract_answer(response, question)
            extracted_answers.append(extracted)
            
        return extracted_answers 