import sys
import os
import unittest
import pandas as pd

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.metrics import normalize_answer, exact_match, contains_answer, calculate_accuracy

class TestEvaluationMetrics(unittest.TestCase):
    def test_normalize_answer(self):
        # Test basic normalization
        self.assertEqual(normalize_answer("42"), "42")
        self.assertEqual(normalize_answer(" 42 "), "42")
        
        # Test operator spacing
        self.assertEqual(normalize_answer("3 + 4"), "3+4")
        self.assertEqual(normalize_answer("3 - 4"), "3-4")
        self.assertEqual(normalize_answer("3 * 4"), "3*4")
        self.assertEqual(normalize_answer("3 / 4"), "3/4")
        
        # Test multiple spaces
        self.assertEqual(normalize_answer("hello   world"), "hello world")
    
    def test_exact_match(self):
        # Test exact matches
        self.assertTrue(exact_match("42", "42"))
        self.assertTrue(exact_match(" 42 ", "42"))
        self.assertTrue(exact_match("3 + 4", "3+4"))
        
        # Test non-matches
        self.assertFalse(exact_match("42", "43"))
        self.assertFalse(exact_match("42", ""))
        self.assertFalse(exact_match("", "42"))
    
    def test_contains_answer(self):
        # Test contains
        self.assertTrue(contains_answer("The answer is 42", "42"))
        self.assertTrue(contains_answer("I think it's 3 + 4", "3+4"))
        
        # Test does not contain
        self.assertFalse(contains_answer("The answer is 42", "43"))
        self.assertFalse(contains_answer("", "42"))
    
    def test_calculate_accuracy(self):
        # Create test data
        test_data = [
            {
                "question": "What is 2+2?",
                "answer": "4",
                "raw_response": "I think the answer is 4",
                "extracted_answer": "4"
            },
            {
                "question": "What is 3+3?",
                "answer": "6",
                "raw_response": "Let me calculate: 3+3=6",
                "extracted_answer": "6"
            },
            {
                "question": "What is 4+4?",
                "answer": "8",
                "raw_response": "The answer is 9",  # Incorrect
                "extracted_answer": "9"  # Incorrect
            }
        ]
        
        # Calculate accuracy
        metrics, df = calculate_accuracy(
            test_data, 
            answer_col='answer', 
            raw_response_col='raw_response', 
            extracted_answer_col='extracted_answer'
        )
        
        # Check metrics
        self.assertAlmostEqual(metrics['exact_match'], 2/3)
        self.assertAlmostEqual(metrics['answer_in_extracted'], 2/3)
        self.assertAlmostEqual(metrics['answer_in_raw'], 2/3)
        
        # Check DataFrame columns
        self.assertTrue('exact_match' in df.columns)
        self.assertTrue('answer_in_extracted' in df.columns)
        self.assertTrue('answer_in_raw' in df.columns)

if __name__ == '__main__':
    unittest.main() 