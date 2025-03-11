import sys
import os
import unittest

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.answer_extraction import AnswerExtractor

class TestAnswerExtraction(unittest.TestCase):
    def setUp(self):
        self.extractor = AnswerExtractor()  # No model for testing
    
    def test_extract_answer_pattern(self):
        # Test "Answer: X" pattern
        response = "I need to solve this step by step.\nFirst, I'll do this.\nThen that.\nAnswer: 42"
        self.assertEqual(self.extractor.extract_math_answer(response), "42")
        
        # Test with different spacing
        response = "Let me solve this.\nAnswer:  42  "
        self.assertEqual(self.extractor.extract_math_answer(response), "42")
        
        # Test with content after the answer
        response = "Let me solve this.\nAnswer: 42\nI hope that's correct."
        self.assertEqual(self.extractor.extract_math_answer(response), "42")
    
    def test_the_answer_is_pattern(self):
        # Test "The answer is X" pattern
        response = "Let me solve this.\nThe answer is 42."
        self.assertEqual(self.extractor.extract_math_answer(response), "42")
        
        # Test with different spacing
        response = "Let me solve this.\nThe answer is  42  "
        self.assertEqual(self.extractor.extract_math_answer(response), "42")
    
    def test_therefore_pattern(self):
        # Test "Therefore, X" pattern
        response = "Let me solve this.\nTherefore, 42 is the answer."
        self.assertEqual(self.extractor.extract_math_answer(response), "42")
        
        # Test with "Therefore, the answer is X"
        response = "Let me solve this.\nTherefore, the answer is 42."
        self.assertEqual(self.extractor.extract_math_answer(response), "42")
    
    def test_thus_pattern(self):
        # Test "Thus, X" pattern
        response = "Let me solve this.\nThus, 42 is the answer."
        self.assertEqual(self.extractor.extract_math_answer(response), "42")
        
        # Test with "Thus, the answer is X"
        response = "Let me solve this.\nThus, the answer is 42."
        self.assertEqual(self.extractor.extract_math_answer(response), "42")
    
    def test_so_pattern(self):
        # Test "So, X" pattern
        response = "Let me solve this.\nSo, 42 is the answer."
        self.assertEqual(self.extractor.extract_math_answer(response), "42")
        
        # Test with "So, the answer is X"
        response = "Let me solve this.\nSo, the answer is 42."
        self.assertEqual(self.extractor.extract_math_answer(response), "42")
    
    def test_equals_pattern(self):
        # Test "= X" pattern
        response = "Let me solve this.\nx = 42"
        self.assertEqual(self.extractor.extract_math_answer(response), "42")
        
        # Test with multiple equals
        response = "Let me solve this.\nx = 10\ny = 20\nz = 42"
        self.assertEqual(self.extractor.extract_math_answer(response), "42")
    
    def test_final_answer_pattern(self):
        # Test "final answer is X" pattern
        response = "Let me solve this.\nThe final answer is 42."
        self.assertEqual(self.extractor.extract_math_answer(response), "42")
        
        # Test without "The"
        response = "Let me solve this.\nFinal answer is 42."
        self.assertEqual(self.extractor.extract_math_answer(response), "42")
    
    def test_result_pattern(self):
        # Test "result is X" pattern
        response = "Let me solve this.\nThe result is 42."
        self.assertEqual(self.extractor.extract_math_answer(response), "42")
        
        # Test without "The"
        response = "Let me solve this.\nResult is 42."
        self.assertEqual(self.extractor.extract_math_answer(response), "42")
    
    def test_structured_format(self):
        # Test our required structured format
        response = "Reasoning: I need to add 2 and 2 together.\nFinal Answer: 4"
        self.assertEqual(self.extractor.extract_math_answer(response), "4")
        
        # Test with extra content
        response = "Some extra text\nReasoning: I need to add 2 and 2 together.\nFinal Answer: 4\nMore text"
        self.assertEqual(self.extractor.extract_math_answer(response), "4")
        
        # Test with multiline reasoning
        response = "Reasoning: I need to add 2 and 2 together.\nThis is a multi-line reasoning.\nIt has several steps.\nFinal Answer: 4"
        self.assertEqual(self.extractor.extract_math_answer(response), "4")
    
    def test_llama_format(self):
        # Test Llama 3 format
        response = "<|begin_of_text|><|user|>\nWhat is 2+2?<|end_of_turn|>\n<|assistant|>\nTo solve this problem, I need to add 2 and 2 together.\n\n2 + 2 = 4\n\nAnswer: 4"
        self.assertEqual(self.extractor.extract_math_answer(response), "4")
        
        # Test with question and answer format
        response = "<|assistant|>\nQuestion: What is 2+2?\n\nTo solve this problem, I need to add 2 and 2 together.\n\n2 + 2 = 4\n\nAnswer: 4"
        self.assertEqual(self.extractor.extract_math_answer(response), "4")
        
        # Test with our structured format
        response = "<|assistant|>\nReasoning: To solve this problem, I need to add 2 and 2 together.\n\n2 + 2 = 4\n\nFinal Answer: 4"
        self.assertEqual(self.extractor.extract_math_answer(response), "4")
        
        # Test the exact format mentioned in the issue
        response = "<|begin_of_text|><|user|>\nWhat is 2+2?<|end_of_turn|>\n<|assistant|>\nAnswer: 4"
        self.assertEqual(self.extractor.extract_math_answer(response), "4")
    
    def test_clean_llama_tokens(self):
        # Test cleaning Llama tokens
        response = "<|begin_of_text|><|user|>\nWhat is 2+2?<|end_of_turn|>\n<|assistant|>\nAnswer: 4"
        self.assertEqual(self.extractor._clean_llama_tokens(response), "\nWhat is 2+2?\n\nAnswer: 4")
        
        # Test extracting answer part when question is included
        response = "Question: What is 2+2?\n\nTo solve this, I'll add 2 and 2.\n\nAnswer: 4"
        self.assertEqual(self.extractor._clean_llama_tokens(response), "Answer: 4")
        
        # Test extracting from structured format
        response = "Reasoning: To solve this, I'll add 2 and 2.\nFinal Answer: 4"
        self.assertEqual(self.extractor._clean_llama_tokens(response), "Final Answer: 4")
    
    def test_clean_extracted_answer(self):
        # Test cleaning "the answer is X"
        self.assertEqual(self.extractor._clean_extracted_answer("the answer is 42"), "42")
        
        # Test cleaning "is X"
        self.assertEqual(self.extractor._clean_extracted_answer("is 42"), "42")
        
        # Test cleaning trailing punctuation
        self.assertEqual(self.extractor._clean_extracted_answer("42."), "42")
        self.assertEqual(self.extractor._clean_extracted_answer("42,"), "42")
        
        # Test cleaning "equal to X"
        self.assertEqual(self.extractor._clean_extracted_answer("equal to 42"), "42")
        
        # Test cleaning "equals X"
        self.assertEqual(self.extractor._clean_extracted_answer("equals 42"), "42")
        
        # Test cleaning "= X"
        self.assertEqual(self.extractor._clean_extracted_answer("= 42"), "42")
        
        # Test cleaning sentence with period
        self.assertEqual(self.extractor._clean_extracted_answer("42. This is the answer."), "42")
    
    def test_fallback_to_last_line(self):
        # Test fallback to last line when no pattern matches
        response = "Let me solve this.\nThe result is 42"
        self.assertEqual(self.extractor.extract_math_answer(response), "42")
        
        # Test with empty lines
        response = "Let me solve this.\n\n\nThe result is 42\n\n"
        self.assertEqual(self.extractor.extract_math_answer(response), "42")
    
    def test_sanity_check_failures(self):
        """Test cases that failed in the sanity check."""
        
        # Case 1: Calculus differentiation with boxed answer
        response = """## Step 1: Identify the function and the variable with respect to which we are differentiating.
The function given is -17574*l**6 - 2*l**2 - 5*l - 25, and we are differentiating with respect to l.

## Step 2: Rewrite the function in a form best suited for differentiation
The function is already in a simple polynomial form, so no further simplification is needed.

## Step 3: Apply the power rule for differentiation
To differentiate the function, we'll apply the power rule, which states that if f(x) = x^n, then f'(x) = n*x^(n-1). We'll apply this rule to each term of the function.

## Step 4: Differentiate each term separately
- The derivative of -17574*l**6 is -17574*6*l**5 = -1054440*l**5.
- The derivative of -2*l**2 is -2*2*l**1 = -4*l.
- The derivative of -5*l is -5*1 = -5.
- The derivative of -25 is 0, since the derivative of a constant is always 0.

## Step 5: Combine all differentiated terms and simplify
The derivative of the function is -1054440*l**5 - 4*l - 5.

## Step 6: Find the second derivative
To find the second derivative, we'll differentiate the first derivative with respect to l.
- The derivative of -1054440*l**5 is -1054440*5*l**4 = -5267200*l**4.
- The derivative of -4*l is -4*1 = -4.
- The derivative of -5 is 0, since the derivative of a constant is always 0.

## Step 7: Combine all differentiated terms and simplify
The second derivative is -5267200*l**4 - 4.

## Step 8: Find the third derivative
To find the third derivative, we'll differentiate the second derivative with respect to l.
- The derivative of -5267200*l**4 is -5267200*4*l**3 = -21068800*l**3.
- The derivative of -4 is 0, since the derivative of a constant is always 0.

## Step 9: Combine all differentiated terms and simplify
The third derivative is -21068800*l**3.

The final answer is: $\\boxed{-21068800*l**3}$"""
        self.assertEqual(self.extractor.extract_math_answer(response), "-21068800*l**3")
        
        # Case 2: Algebra polynomial roots with final answer line
        response = """Let me find the first derivative of -w**3 + 1215*w - 4184.

## Step 1: Identify the function and variable to differentiate
The function to differentiate is -w**3 + 1215*w - 4184, and we are differentiating with respect to w.

## Step 2: Rewrite the function in a form best suited for differentiation
The function is already in a simple form, so no further rewriting is necessary.

## Step 3: Apply the sum and difference rules by differentiating each term separately
To differentiate the function, we'll apply the power rule for the first term, the constant multiple rule for the second term, and the constant multiple rule for the third term.

## Step 4: Differentiate each term using the power rule, product rule, quotient rule, and chain rule as necessary
- The derivative of -w**3 is -3*w**2 (power rule).
- The derivative of 1215*w is 1215 (constant multiple rule).
- The derivative of -4184 is 0 (constant multiple rule).

## Step 5: Combine all differentiated terms and simplify to get the derivative
The derivative of the function is -3*w**2 + 1215.

## Step 6: Check if the result can be further simplified
No further simplification is possible.

The final answer is: $\\boxed{-3w^2 + 1215}$"""
        self.assertEqual(self.extractor.extract_math_answer(response), "-3*w**2 + 1215")
        
        # Case 3: Second derivative with respect to a variable
        response = """## Step 1: Identify the function and the variable with respect to which we are differentiating.
The function given is 280*n**3*o**2 - n**3*o - 10*n**3 + 12*n**2*o**2 - n*o**2 - 2*n - 5*o, and we are differentiating it with respect to o.

## Step 2: Rewrite the function in a form best suited for differentiation.
The function is already in a suitable form for differentiation, with all terms expanded and radicals expressed as exponents.

## Step 3: Apply the sum and difference rules by differentiating each term separately.
To differentiate the function, we'll apply the power rule for each term. The power rule states that if f(x) = x^n, then f'(x) = nx^(n-1).

## Step 4: Differentiate each term using the power rule.
- The derivative of 280*n**3*o**2 is 2*280*n**3*o = 560*n**3*o.
- The derivative of -n**3*o is -n**3.
- The derivative of -10*n**3 is 0 (constant term with respect to o).
- The derivative of 12*n**2*o**2 is 24*n**2*o.
- The derivative of -n*o**2 is -2*n*o.
- The derivative of -2*n is 0 (constant term with respect to o).
- The derivative of -5*o is -5.

## Step 5: Combine all differentiated terms and simplify to get the first derivative.
560*n**3*o - n**3 + 24*n**2*o - 2*n*o - 5

## Step 6: Differentiate the first derivative to find the second derivative.
- The derivative of 560*n**3*o is 560*n**3.
- The derivative of -n**3 is 0.
- The derivative of 24*n**2*o is 24*n**2.
- The derivative of -2*n*o is -2*n.
- The derivative of -5 is 0.

## Step 7: Combine all differentiated terms and simplify to get the second derivative.
560*n**3 + 24*n**2 - 2*n

The final answer is: $\\boxed{560*n**3 + 24*n**2 - 2*n}$"""
        self.assertEqual(self.extractor.extract_math_answer(response), "560*n**3 + 24*n**2 - 2*n")
        
        # Case 4: Expression with both variables and multiple operations
        response = """## Step 1: Identify the function and variable with respect to which we are differentiating
The function to differentiate is 3*w**2*z**2 + 67*w**2*z - w**2 + 2*w*z - 133 with respect to z.

## Step 2: Rewrite the function in a form best suited for differentiation
The function is already in a suitable form for differentiation, with all terms expanded.

## Step 3: Apply the sum and difference rules by differentiating each term separately
To differentiate the function, we'll apply the power rule for each term. The power rule states that if f(x) = x^n, then f'(x) = nx^(n-1).

## Step 4: Differentiate each term using the power rule
- The derivative of 3*w**2*z**2 with respect to z is 3*w**2 * 2*z = 6*w**2*z.
- The derivative of 67*w**2*z with respect to z is 67*w**2.
- The derivative of -w**2 with respect to z is 0 (since -w**2 is constant with respect to z).
- The derivative of 2*w*z with respect to z is 2*w.
- The derivative of -133 with respect to z is 0 (since -133 is constant with respect to z).

## Step 5: Combine all differentiated terms and simplify to get the derivative
The derivative is 6*w**2*z + 67*w**2 + 2*w.

## Step 6: Check if the result can be further simplified
No further simplification is possible.

The final answer is: $\\boxed{6*w**2*z + 67*w**2 + 2*w}$"""
        self.assertEqual(self.extractor.extract_math_answer(response), "6*w**2*z + 67*w**2 + 2*w")
        
        # Case 5: Cut-off response where the answer is at the end
        response = """## Step 1: Identify the function and the variable with respect to which we are differentiating.
The function given is 280*n**3*o**2 - n**3*o - 10*n**3 + 12*n**2*o**2 - n*o**2 - 2*n - 5*o, and we are differentiating it with respect to o.

## Step 2: Rewrite the function in a form best suited for differentiation.
The function is already in a suitable form for differentiation, with all terms expanded and radicals expressed as exponents.

## Step 3: Apply the sum and difference rules by differentiating each term separately.
To differentiate the function, we'll apply the power rule for each term. The power rule states that if f(x) = x^n, then f'(x) = nx^(n-1).

## Step 4: Differentiate each term using the power rule.
- The derivative of 280*n**3*o**2 is 2*280*n**3*o = 560*n**3*o.
- The derivative of n**3*o is 3*n**2*o.
- The derivative of -10*n**3 is -30*n**2.
- The derivative of 12*n**2*o**2 is 24*n**2*o.
- The derivative of -n*o**2 is -o.
- The derivative of -2*n is 0 (since the derivative of a constant is 0).
- The derivative of -5*o is 0 (since the derivative of a constant is 0).

## Step 5: Combine all differentiated terms and simplify to get the first derivative.
The first derivative is 560*n**3*o + 3*n**2*o - 30*n**2 + 24*n**2*o - o.

## Step 6: Simplify the first derivative.
Combine like terms: 560*n**3*o + 33*n**2*o - o.

## Step 7: Differentiate the first derivative to find the second derivative.
Apply the power rule again to each term:
- The derivative of 560*n**3*o is 1680*n**2*o."""
        # In this case, the answer isn't complete, but our extraction should find what it can
        self.assertEqual(self.extractor.extract_math_answer(response), "560*n**3 + 24*n**2 - 2*n")

if __name__ == '__main__':
    unittest.main() 