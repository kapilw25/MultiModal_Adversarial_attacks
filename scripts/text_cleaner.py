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
    
    # DEBUG: Log every call to verify it's being used
    logger.info(f"TEXT_CLEANER_CALLED: Input length={len(text)}, starts with: '{text[:50]}...'")
    
    if len(text) > 100:
        logger.warning(f"LONG_INPUT_DETECTED: '{text}'[:100]")
    
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
    
    # Step 3: LLAVA-specific cleaning
    # Remove "ER:" artifacts (for LLAVA-1.5-7B)
    cleaned_text = re.sub(r'\bER:\s*', '', cleaned_text)
    
    # Remove Mistral instruction tokens (for LLAVA-v1.6-Mistral-7B)
    cleaned_text = re.sub(r'\[INST\].*?\[/INST\]\s*', '', cleaned_text, flags=re.DOTALL)
    cleaned_text = re.sub(r'\[INST\]\s*|\[/INST\]\s*', '', cleaned_text)
    
    # Handle unresolved placeholders and format artifacts
    cleaned_text = re.sub(r'The answer is <answer>\.?$', 'The answer is [Unable to determine]', cleaned_text)
    cleaned_text = re.sub(r'\[your response\]', '', cleaned_text)  # Remove new format placeholder if present
    
    # Detect and fix numerical overflow (>20 digits)
    cleaned_text = re.sub(r'\b\d{20,}\b', '[Invalid number]', cleaned_text)
    
    # Step 4: Clean up whitespace
    cleaned_text = re.sub(r'\n+', ' ', cleaned_text)  # Replace newlines with spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Replace multiple spaces with single space
    cleaned_text = cleaned_text.strip()
    
    # Step 5: Extract concise answers from verbose responses
    def extract_concise_answer(text: str) -> str:
        """Extract concise answers from verbose VLM responses"""
        
        # Handle "The answer is The [verbose explanation]" pattern
        if text.lower().startswith("the answer is the "):
            verbose_part = text[18:]  # Remove "The answer is The "
            
            # For "how many" questions - extract numbers from verbose text
            # "pie charts in the image have three different coloured sectors" → "3"
            if any(word in verbose_part.lower() for word in ["three", "four", "five", "six", "seven", "eight", "nine", "ten"]):
                word_to_num = {
                    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
                    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
                    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15",
                    "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20"
                }
                for word, num in word_to_num.items():
                    if word in verbose_part.lower():
                        return f"The answer is {num}"
            
            # For percentage questions - extract percentage values
            # "percentage of support for Democrats in June 2021 is 44.0%" → "44.0%"
            percent_match = re.search(r'is (\d+(?:\.\d+)?%)', verbose_part)
            if percent_match:
                return f"The answer is {percent_match.group(1)}"
            
            # For numerical values - extract numbers
            # "final internet speed value for Germany is 85" → "85"
            num_match = re.search(r'is (\d+(?:\.\d+)?)', verbose_part)
            if num_match:
                return f"The answer is {num_match.group(1)}"
                
            # For quoted titles/captions - extract the quoted content
            # "caption of this chart is \"Changes in political party support...\"" → "Changes in political party support..."
            quote_match = re.search(r'is "([^"]+)"', verbose_part)
            if quote_match:
                return f"The answer is {quote_match.group(1)}"
            
            # For simple statements - extract after "is "
            is_match = re.search(r'is (.+?)(?:\.|$)', verbose_part)
            if is_match:
                extracted = is_match.group(1).strip()
                # Don't extract if it's still too verbose (>50 chars)
                if len(extracted) <= 50:
                    return f"The answer is {extracted}"
        
        return text
    
    # Apply concise answer extraction
    if len(cleaned_text) > 50 and cleaned_text.lower().startswith("the answer is the "):
        logger.info(f"VERBOSE INPUT DETECTED: {cleaned_text[:100]}...")
        
    cleaned_text = extract_concise_answer(cleaned_text)
    
    if len(cleaned_text) > 50 and cleaned_text.lower().startswith("the answer is the "):
        logger.warning(f"VERBOSITY NOT FIXED: {cleaned_text[:100]}...")
    
    # Step 6: Standardize answer format
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
