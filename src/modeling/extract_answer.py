import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_llama_model(model_name="meta-llama/Llama-3.1-70B-Instruct"):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'  # Set padding side to left for decoder-only models
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    )
    print("Model loaded successfully")
    return model, tokenizer

def extract_answer(model, tokenizer, solution_text):
    # Check if the solution follows the structured format with "Final Answer:"
    if "Final Answer:" in solution_text:
        # Extract just the final answer part without using the model
        final_answer = solution_text.split("Final Answer:")[-1].strip()
        # Clean up common formatting issues without using the model
        final_answer = final_answer.strip('*').strip()
        return final_answer
    else:
        # Only use the model for unstructured responses where we can't easily extract the answer
        prompt = f"""<|begin_of_text|><|user|>
Extract ONLY the final numerical answer from this solution. Return ONLY the number or value with no additional text, no asterisks, no explanation, and no formatting:
{solution_text}
<|end_of_text|>
<|assistant|>"""

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=False
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("<|assistant|>")[-1].strip()
        return answer

# Example usage
if __name__ == "__main__":
    model, tokenizer = load_llama_model()
    
    # Example solution text
    solution = """To solve for r, I'll use elimination.
    From the first equation: -42r + 27c = -1167
    From the second equation: 130r + 4c = 372
    
    I'll multiply the first equation by 4:
    -168r + 108c = -4668
    
    Then add this to the second equation:
    -168r + 108c = -4668
    130r + 4c = 372
    -------------------------
    -38r + 112c = -4296
    
    Solving for c: c = (-4296 + 38r)/112 = -38.36 + 0.34r
    
    Substituting back into the first equation:
    -42r + 27(-38.36 + 0.34r) = -1167
    -42r - 1035.72 + 9.18r = -1167
    -32.82r = -131.28
    r = 4
    
    Therefore, r = 4."""
    
    answer = extract_answer(model, tokenizer, solution)
    print(f"Extracted answer: {answer}")
