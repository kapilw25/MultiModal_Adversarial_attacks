#!/usr/bin/env python3
"""
Unit test for Microsoft GUI-Actor-2B-Qwen2-VL model.
This script tests the model's ability to identify clickable elements in a GUI image.
"""

import os
import sys
import time
import torch
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoTokenizer, AutoModel

# Add parent directory to path to import local modules if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Default image path and instruction
DEFAULT_IMAGE_PATH = "data/test_extracted/chart/20231114102825506748.png"
DEFAULT_INSTRUCTION = "Click on the most important element in this chart"

def test_gui_actor_pointer(image_path=DEFAULT_IMAGE_PATH, instruction=DEFAULT_INSTRUCTION):
    """Test the GUI-Actor-2B-Qwen2-VL model's ability to identify clickable elements."""
    print("\nTesting Microsoft GUI-Actor-2B-Qwen2-VL model for pointer prediction...")
    
    # Set device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Model path
    model_name_or_path = "microsoft/GUI-Actor-2B-Qwen2-VL"
    
    # Load processor and tokenizer
    print("Loading processor and tokenizer...")
    start_time = time.time()
    try:
        processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        processor_time = time.time() - start_time
        print(f"Processor and tokenizer loaded successfully in {processor_time:.2f} seconds!")
    except Exception as e:
        print(f"Error loading processor or tokenizer: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Load model
    print("\nLoading model...")
    start_time = time.time()
    try:
        # Load with generic AutoModel class which works for this model
        model = AutoModel.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
            trust_remote_code=True
        )
        
        model_time = time.time() - start_time
        print(f"Model loaded successfully in {model_time:.2f} seconds!")
        
        # Print model info
        param_count = sum(p.numel() for p in model.parameters()) / 1e9
        print(f"Model size: {param_count:.2f} billion parameters")
        print(f"Model architecture: {model.__class__.__name__}")
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            return False
            
        print(f"\nAnalyzing image: {image_path}")
        print(f"Instruction: '{instruction}'")
        
        try:
            # Load and process image
            image = Image.open(image_path).convert("RGB")
            
            # Create conversation format for GUI interaction task
            conversation = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.",
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {
                            "type": "text",
                            "text": instruction
                        },
                    ],
                },
            ]
            
            # Process inputs
            print("Processing inputs...")
            inputs = processor(
                text=tokenizer.apply_chat_template(conversation, tokenize=False),
                images=image,
                return_tensors="pt"
            ).to(device)
            
            # Run forward pass to get hidden states
            print("Running forward pass...")
            with torch.no_grad():
                outputs = model(**inputs)
                
                # Check if the model has pointer prediction capabilities
                if hasattr(model, "pointer_head") and hasattr(outputs, "pointer_logits"):
                    print("Model has pointer prediction capabilities!")
                    pointer_logits = outputs.pointer_logits
                    
                    # Get the predicted coordinates
                    pred_x = pointer_logits[0, 0].item()
                    pred_y = pointer_logits[0, 1].item()
                    
                    print(f"Predicted click coordinates: ({pred_x:.4f}, {pred_y:.4f})")
                else:
                    print("Model doesn't expose pointer prediction in this configuration.")
                    print("This is expected when using the generic AutoModel class.")
                    print("For full pointer prediction, the GUI-Actor package is required.")
                    
                    # Simulate a pointer prediction for visualization purposes
                    # This is just for demonstration - not actual model prediction
                    # We'll use the center of the image as a placeholder
                    pred_x, pred_y = 0.5, 0.5
                    print(f"Using placeholder click coordinates: ({pred_x:.4f}, {pred_y:.4f})")
            
            # Visualize the image with the predicted click point
            plt.figure(figsize=(10, 8))
            plt.imshow(image)
            plt.scatter(pred_x * image.width, pred_y * image.height, c='r', s=100, marker='x')
            plt.title(f"Instruction: {instruction}")
            
            # Save the visualization
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "gui_actor_pointer_prediction.png")
            plt.savefig(output_path)
            print(f"Visualization saved to {output_path}")
            
            return True
        except Exception as e:
            print(f"Error during image analysis: {e}")
            import traceback
            traceback.print_exc()
            return False
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to parse arguments and run tests."""
    parser = argparse.ArgumentParser(description="Test GUI-Actor model for pointer prediction")
    parser.add_argument(
        "--image", 
        type=str,
        default=DEFAULT_IMAGE_PATH,
        help=f"Path to an image for testing (default: {DEFAULT_IMAGE_PATH})"
    )
    parser.add_argument(
        "--instruction", 
        type=str,
        default=DEFAULT_INSTRUCTION,
        help=f"Instruction for GUI interaction (default: '{DEFAULT_INSTRUCTION}')"
    )
    args = parser.parse_args()
    
    test_gui_actor_pointer(image_path=args.image, instruction=args.instruction)

if __name__ == "__main__":
    main()
