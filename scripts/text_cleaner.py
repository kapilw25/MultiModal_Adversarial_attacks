#!/usr/bin/env python3
"""
VLM Text Cleaner Module

Simple text cleaning module for Vision-Language Model responses.
Focuses on removing unnecessary conversation formatting and extracting clean answers.

Usage:
    from scripts.text_cleaner import clean_vlm_response
    
    # Clean a VLM response
    cleaned_text = clean_vlm_response(raw_response)
"""

import re
import logging
from difflib import SequenceMatcher

# Configure logging
logger = logging.getLogger(__name__)

def clean_vlm_response(text: str) -> str:
    """
    Clean a VLM response by removing conversation formatting and unnecessary content.
    
    Args:
        text (str): Raw VLM response text
        
    Returns:
        str: Cleaned response text
    """
    if not text or not isinstance(text, str):
        return ""
    
    original_text = text
    cleaned_text = text.strip()
    
    # Step 1: Remove conversation formatting
    # Remove User: and Assistant: prefixes with all the junk in between
    cleaned_text = re.sub(r'^User:.*?Assistant:\s*', '', cleaned_text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    cleaned_text = re.sub(r'^Assistant:\s*', '', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'^User:\s*', '', cleaned_text, flags=re.IGNORECASE)
    
    # Step 2: Remove instruction formatting
    # Remove "Answer format" instructions and similar junk
    cleaned_text = re.sub(r'Answer format.*?:\s*', '', cleaned_text, flags=re.IGNORECASE | re.DOTALL)
    cleaned_text = re.sub(r'do not generate any other content.*?:\s*', '', cleaned_text, flags=re.IGNORECASE | re.DOTALL)
    
    # Step 3: Clean up whitespace
    cleaned_text = re.sub(r'\n+', ' ', cleaned_text)  # Replace newlines with spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Replace multiple spaces with single space
    cleaned_text = cleaned_text.strip()
    
    # Step 4: Standardize answer format
    # If it doesn't start with "The answer is", add it (unless it's empty or just <answer>)
    if cleaned_text and not cleaned_text.lower().startswith("the answer is"):
        if cleaned_text != "<answer>" and cleaned_text.strip():
            cleaned_text = f"The answer is {cleaned_text}"
    
    result = cleaned_text.strip()
    
    # Log cleaning operation for debugging
    if result != original_text:
        logger.debug(f"Cleaned: '{original_text[:50]}...' -> '{result[:50]}...'")
    
    return result

def fuzzy_match(text1: str, text2: str, threshold: float = 0.8) -> bool:
    """
    Check if two texts are similar using fuzzy matching.
    
    Args:
        text1 (str): First text
        text2 (str): Second text  
        threshold (float): Similarity threshold (0.0 to 1.0)
        
    Returns:
        bool: True if texts are similar enough
    """
    if not text1 or not text2:
        return text1 == text2
    
    similarity = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    return similarity >= threshold

def test_cleaner():
    """Test the cleaner with known problematic examples using fuzzy matching."""
    test_cases = [
        {
            "input": "User:\n\n\n\nHow many different coloured sectors are there in each pie chart? Answer format (do not generate any other content): The answer is <answer>.\nAssistant: 3.",
            "expected": "The answer is 3",
            "description": "SmolVLM2 conversation format"
        },
        {
            "input": "Assistant: 45.0%.",
            "expected": "The answer is 45.0%",
            "description": "Simple assistant response with percentage"
        },
        {
            "input": "The answer is Changes in political party support from June to December 2021.",
            "expected": "The answer is Changes in political party support from June to December 2021",
            "description": "Already clean response"
        },
        {
            "input": "User:\n\n\n\nWhat is the title of the graph? Answer format (do not generate any other content): The answer is <answer>.\nAssistant: <answer>.",
            "expected": "The answer is <answer>",
            "description": "Empty answer placeholder"
        },
        {
            "input": "No.",
            "expected": "The answer is No",
            "description": "Simple one-word answer"
        },
        {
            "input": "3",
            "expected": "The answer is 3",
            "description": "Just a number"
        }
    ]
    
    print("=== VLM Text Cleaner Test Results (Fuzzy Matching) ===\n")
    
    all_passed = True
    for i, test in enumerate(test_cases, 1):
        cleaned = clean_vlm_response(test['input'])
        # Use fuzzy matching for comparison
        passed = fuzzy_match(cleaned, test['expected'], threshold=0.85)
        all_passed = all_passed and passed
        
        print(f"Test {i}: {test['description']}")
        print(f"Input:    '{test['input']}'")
        print(f"Output:   '{cleaned}'")
        print(f"Expected: '{test['expected']}'")
        
        if not passed:
            similarity = SequenceMatcher(None, cleaned.lower(), test['expected'].lower()).ratio()
            print(f"Similarity: {similarity:.2f}")
        
        print(f"✅ PASS" if passed else "❌ FAIL")
        print("-" * 80)
    
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    return all_passed

# Convenience function for integration
def clean_evaluation_result(result_dict: dict, text_field: str = 'text') -> dict:
    """
    Clean the text field in an evaluation result dictionary.
    
    Args:
        result_dict (dict): Dictionary containing evaluation results
        text_field (str): Field name containing the text to clean
        
    Returns:
        dict: Dictionary with cleaned text field
    """
    if text_field in result_dict:
        result_dict[text_field] = clean_vlm_response(result_dict[text_field])
    
    return result_dict

if __name__ == "__main__":
    # Run tests when script is executed directly
    test_cleaner()
