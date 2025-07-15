#!/usr/bin/env python3
"""
Unit test for Intel/llava-gemma-2b model.
This script tests the model's ability to analyze a chart image.
"""

import os
import sys
import time
import torch
from PIL import Image

# Add parent directory to path to import local modules if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Hardcoded image path and question
IMAGE_PATH = "data/test_extracted/chart/20231114102825506748.png"
QUESTION = "What is shown in this chart?"

def test_llava_gemma_2b():
    """Test the llava-gemma-2b model with the chart image."""
    print("\nTesting Intel/llava-gemma-2b model on chart image...")
    
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
        from transformers import (
            LlavaForConditionalGeneration,
            AutoTokenizer,
            AutoProcessor,
            CLIPImageProcessor
        )
        print("Required packages are installed.")
    except ImportError as e:
        print(f"Error: Missing required package - {e}")
        print("Please install required packages with: pip install transformers")
        return False
    
    # Get transformers version
    import transformers
    transformers_version = transformers.__version__
    print(f"Transformers version: {transformers_version}")
    
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
    
    # Define checkpoint
    checkpoint = "Intel/llava-gemma-2b"
    
    # Load model
    print(f"\nLoading model from {checkpoint}...")
    start_time = time.time()
    try:
        model = LlavaForConditionalGeneration.from_pretrained(
            checkpoint,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map={"": device} if torch.cuda.is_available() else None
        )
        
        # Load processor
        print("Loading processor...")
        processor = AutoProcessor.from_pretrained(checkpoint)
        
        load_time = time.time() - start_time
        print(f"Model and processor loaded successfully in {load_time:.2f} seconds!")
        
        # Print model info if available
        try:
            param_count = sum(p.numel() for p in model.parameters()) / 1e9  # Convert to billions
            print(f"Model size: {param_count:.2f} billion parameters")
        except:
            print("Could not determine model size")
    except Exception as e:
        print(f"Error loading model or processor: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Load image
    print("\nLoading image...")
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"Image loaded successfully: {image.size[0]}x{image.size[1]}")
    except Exception as e:
        print(f"Error loading image: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Prepare inputs
    print("\nPreparing inputs...")
    try:
        # Use gemma chat template
        prompt_text = f"<image>\n{QUESTION}"
        
        prompt = processor.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt_text}],
            tokenize=False,
            add_generation_prompt=True
        )
        
        print(f"Prompt: {prompt}")
        
        # Process inputs
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        
        # Move inputs to device if using GPU
        if torch.cuda.is_available():
            inputs = {k: v.to(device) for k, v in inputs.items()}
    except Exception as e:
        print(f"Error preparing inputs: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Generate
    print("\nGenerating response...")
    try:
        start_time = time.time()
        
        with torch.no_grad():
            generate_ids = model.generate(
                **inputs,
                max_length=300,
                do_sample=False,
                temperature=0.7,
                top_p=0.9,
            )
        
        inference_time = time.time() - start_time
        
        # Decode output
        output = processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        print(f"\nResponse (generated in {inference_time:.2f}s):")
        print(output)
        
        return True
    except Exception as e:
        print(f"Error during generation: {e}")
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

if __name__ == "__main__":
    test_llava_gemma_2b()
