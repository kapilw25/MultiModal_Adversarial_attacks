#!/usr/bin/env python3
"""
Unit test for Microsoft UDOP-large model.
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

# Hardcoded image path and question (from test_vlm.py)
IMAGE_PATH = "data/test_extracted/chart/20231114102825506748.png"
QUESTION = "What is shown in this chart? Please provide a detailed description."

def test_udop_large():
    """Test the UDOP-large model with a chart image."""
    print("\nTesting Microsoft UDOP-large model on chart image...")
    
    # Print system info for debugging
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Set device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Check if required packages are installed
    print("\nChecking required packages...")
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
        
        # Check if UDOP classes are available
        print("Importing UDOP classes...")
        from transformers import AutoProcessor, UdopForConditionalGeneration
        print("UDOP classes imported successfully")
        
        import datasets
        print(f"Datasets version: {datasets.__version__}")
        from datasets import load_dataset
        print("Required packages are installed.")
    except ImportError as e:
        print(f"Error: Missing required package - {e}")
        print("Please install required packages with: pip install datasets transformers")
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
    
    # Load processor first
    print("\nLoading UDOP processor...")
    start_time = time.time()
    try:
        # Initialize processor with apply_ocr=False since we'll use pre-extracted OCR data
        processor = AutoProcessor.from_pretrained("microsoft/udop-large", apply_ocr=False)
        processor_time = time.time() - start_time
        print(f"Processor loaded successfully in {processor_time:.2f} seconds!")
    except Exception as e:
        print(f"Error loading processor: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Load model separately
    print("\nLoading UDOP model...")
    start_time = time.time()
    try:
        # Load model with appropriate settings
        print("Creating model object...")
        model = UdopForConditionalGeneration.from_pretrained(
            "microsoft/udop-large",
            torch_dtype=torch.float32,  # UDOP uses F32
            device_map=device if torch.cuda.is_available() else None
        )
        
        load_time = time.time() - start_time
        print(f"Model loaded successfully in {load_time:.2f} seconds!")
        
        # Print model info
        param_count = sum(p.numel() for p in model.parameters()) / 1e6  # Convert to millions
        print(f"Model size: {param_count:.2f} million parameters")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Try different question formats to get a better response
    questions = [
        "What is shown in this chart? Please provide a detailed description.",
        "Describe this chart in detail.",
        "What information is presented in this chart?",
        "Analyze this chart and explain what it shows.",
        "What data is visualized in this chart?",
        "Summarize the content of this chart."
    ]
    
    # Load image and create dummy OCR data
    print("\nLoading chart image and creating OCR data...")
    try:
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Create dummy OCR data (words and boxes)
        # For chart analysis, we'll create a simple grid of words and boxes
        width, height = image.size
        
        # Try with more descriptive dummy words
        words = [
            "chart", "data", "line", "bar", "axis", 
            "value", "label", "title", "legend", "year",
            "percent", "number", "country", "growth", "trend",
            "USA", "Germany", "China", "India", "internet",
            "usage", "rate", "population", "2008", "2020"
        ]
        
        boxes = []
        # Create a 5x5 grid of boxes
        for i in range(5):
            for j in range(5):
                # Create normalized box coordinates [x1, y1, x2, y2]
                x1 = j * 0.2
                y1 = i * 0.2
                x2 = (j + 1) * 0.2
                y2 = (i + 1) * 0.2
                boxes.append([x1, y1, x2, y2])
        
        best_answer = ""
        
        # Try each question format
        for question in questions:
            print(f"\nTrying question: '{question}'")
            
            # Process inputs for the model
            print("Processing inputs...")
            encoding = processor(image, question, words, boxes=boxes, return_tensors="pt")
            
            # Move inputs to device if using GPU
            if torch.cuda.is_available():
                encoding = {k: v.to(device) for k, v in encoding.items()}
            
            # Generate answer with different parameters
            print("Generating answer...")
            start_time = time.time()
            with torch.no_grad():
                predicted_ids = model.generate(
                    **encoding,
                    max_new_tokens=256,  # Generate more tokens
                    num_beams=5,         # Use more beams for better quality
                    length_penalty=1.0,  # Encourage longer responses
                    no_repeat_ngram_size=3,  # Avoid repetition
                    early_stopping=True
                )
            
            inference_time = time.time() - start_time
            
            # Decode the answer
            answer = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            print(f"Answer (generated in {inference_time:.2f}s): '{answer}'")
            
            # Keep track of the best (longest) answer
            if len(answer) > len(best_answer):
                best_answer = answer
        
        print("\n=== Best Answer ===")
        print(best_answer)
        
        return True
    except Exception as e:
        print(f"Error during chart analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run the test."""
    parser = argparse.ArgumentParser(description="Test UDOP-large model on chart image")
    args = parser.parse_args()
    
    test_udop_large()

if __name__ == "__main__":
    main()
