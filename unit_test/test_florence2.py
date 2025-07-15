#!/usr/bin/env python3
"""
Unit test for Microsoft Florence-2 models.
This script tests the basic functionality of both Florence-2-base and Florence-2-large models
using the official implementation approach.
"""

import os
import sys
import time
import argparse
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

# Add parent directory to path to import local modules if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_florence2_model(model_size="base"):
    """Test the Florence-2 model with basic captioning tasks.
    
    Args:
        model_size (str): Either "base" (0.23B params) or "large" (0.77B params)
    """
    model_name = f"microsoft/Florence-2-{model_size}"
    print(f"\nTesting {model_name} model...")
    
    # Set device and dtype
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    print(f"Using device: {device}, dtype: {torch_dtype}")
    
    # Load model and processor
    print(f"Loading {model_size} model and processor...")
    start_time = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch_dtype, 
            trust_remote_code=True
        ).to(device)
        
        processor = AutoProcessor.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        load_time = time.time() - start_time
        print(f"Model and processor loaded successfully in {load_time:.2f} seconds!")
        
    except Exception as e:
        print(f"Error loading model or processor: {e}")
        return
    
    # Download and open test image
    print("Downloading test image...")
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
    try:
        image = Image.open(requests.get(url, stream=True).raw)
        print("Test image downloaded successfully!")
    except Exception as e:
        print(f"Error downloading test image: {e}")
        return
    
    def run_example(task_prompt, text_input=None):
        """Run inference with the given task prompt and optional text input."""
        print(f"\nRunning task: {task_prompt}")
        
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input
            
        try:
            # Prepare inputs
            inputs = processor(
                text=prompt, 
                images=image, 
                return_tensors="pt"
            ).to(device, torch_dtype)
            
            # Generate output
            start_time = time.time()
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3
            )
            inference_time = time.time() - start_time
            
            # Decode and post-process
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = processor.post_process_generation(
                generated_text, 
                task=task_prompt, 
                image_size=(image.width, image.height)
            )
            
            print(f"Result ({inference_time:.2f}s):", parsed_answer)
            return parsed_answer
        except Exception as e:
            print(f"Error during inference: {e}")
            return None
    
    # Test basic captioning
    print("\n=== Testing Basic Captioning ===")
    run_example("<CAPTION>")
    
    # Test detailed captioning
    print("\n=== Testing Detailed Captioning ===")
    run_example("<DETAILED_CAPTION>")
    
    print(f"\nFlorence-2-{model_size} model test completed!")
    
    # Free up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    """Main function to parse arguments and run tests."""
    parser = argparse.ArgumentParser(description="Test Florence-2 models")
    parser.add_argument(
        "--model", 
        type=str, 
        default="base",
        choices=["base", "large", "both"],
        help="Which model to test: 'base' (0.23B), 'large' (0.77B), or 'both'"
    )
    args = parser.parse_args()
    
    if args.model == "base" or args.model == "both":
        test_florence2_model("base")
    
    if args.model == "large" or args.model == "both":
        test_florence2_model("large")

if __name__ == "__main__":
    main()
