#!/usr/bin/env python3
"""
Unit test for vikhyatk/moondream2 model.
This script tests the model's ability to analyze a chart image.
"""

import os
import sys
import time
import torch
import argparse
from PIL import Image

# Add parent directory to path to import local modules if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Hardcoded image path and question
IMAGE_PATH = "data/test_extracted/chart/20231114102825506748.png"
QUESTION = "What is shown in this chart?"

def test_moondream2():
    """Test the moondream2 model with a chart image."""
    print("\nTesting vikhyatk/moondream2 model on chart image...")
    
    # Print system info for debugging
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Check if required packages are installed
    print("\nChecking required packages...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("Required packages are installed.")
    except ImportError as e:
        print(f"Error: Missing required package - {e}")
        print("Please install required packages with: pip install transformers")
        return False
    
    # Get absolute path for the image
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    image_path = os.path.join(project_root, IMAGE_PATH)
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return False
    
    print(f"Using image: {image_path}")
    print(f"Question: '{QUESTION}'")
    
    # Load image
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"Image loaded successfully: {image.size[0]}x{image.size[1]}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return False
    
    # Load model
    print("\nLoading moondream2 model...")
    start_time = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            revision="2025-06-21",
            trust_remote_code=True,
            device_map={"": device}
        )
        
        load_time = time.time() - start_time
        print(f"Model loaded successfully in {load_time:.2f} seconds!")
        
        # Print model info if available
        try:
            param_count = sum(p.numel() for p in model.parameters()) / 1e9  # Convert to billions
            print(f"Model size: {param_count:.2f} billion parameters")
        except:
            print("Could not determine model size")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test different capabilities of the model
    try:
        # 1. Short caption
        print("\n=== Testing Short Caption ===")
        caption_start_time = time.time()
        short_caption = model.caption(image, length="short")["caption"]
        caption_time = time.time() - caption_start_time
        print(f"Short caption ({caption_time:.2f}s): {short_caption}")
        
        # 2. Normal caption
        print("\n=== Testing Normal Caption ===")
        caption_start_time = time.time()
        normal_caption = model.caption(image, length="normal")["caption"]
        caption_time = time.time() - caption_start_time
        print(f"Normal caption ({caption_time:.2f}s): {normal_caption}")
        
        # 3. Visual query (our main test)
        print(f"\n=== Testing Visual Query: '{QUESTION}' ===")
        query_start_time = time.time()
        answer = model.query(image, QUESTION)["answer"]
        query_time = time.time() - query_start_time
        print(f"Answer ({query_time:.2f}s): {answer}")
        
        # 4. Try a more specific question about the chart
        specific_question = "What countries are shown in this chart?"
        print(f"\n=== Testing Specific Question: '{specific_question}' ===")
        query_start_time = time.time()
        specific_answer = model.query(image, specific_question)["answer"]
        query_time = time.time() - query_start_time
        print(f"Answer ({query_time:.2f}s): {specific_answer}")
        
        # 5. Try another specific question about the chart
        trend_question = "What trend is shown in this chart over time?"
        print(f"\n=== Testing Trend Question: '{trend_question}' ===")
        query_start_time = time.time()
        trend_answer = model.query(image, trend_question)["answer"]
        query_time = time.time() - query_start_time
        print(f"Answer ({query_time:.2f}s): {trend_answer}")
        
        # Summary of results
        print("\n=== Summary of Results ===")
        print(f"Short caption: {short_caption}")
        print(f"Normal caption: {normal_caption}")
        print(f"Answer to '{QUESTION}': {answer}")
        print(f"Answer to '{specific_question}': {specific_answer}")
        print(f"Answer to '{trend_question}': {trend_answer}")
        
        return True
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        try:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("\nModel resources cleaned up")
        except:
            pass

def main():
    """Main function to run the test."""
    parser = argparse.ArgumentParser(description="Test moondream2 model on chart image")
    args = parser.parse_args()
    
    test_moondream2()

if __name__ == "__main__":
    main()
