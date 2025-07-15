#!/usr/bin/env python3
"""
Simple test for Intel/llava-gemma-2b model using the original example code.
"""

import os
import sys
import argparse
import requests
import logging
import torch
from PIL import Image
from transformers import (
    LlavaForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor
)

# Add parent directory to path to import local modules if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Parse command line arguments
parser = argparse.ArgumentParser(description='Test LLaVA-Gemma model')
parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
args = parser.parse_args()

# Configure logging based on verbose flag
if args.verbose:
    logging.basicConfig(level=logging.INFO)
    # Enable verbose output from transformers
    os.environ["TRANSFORMERS_VERBOSITY"] = "info"
    print("Verbose mode enabled")
else:
    logging.basicConfig(level=logging.WARNING)

# Hardcoded image path and question
IMAGE_PATH = "data/test_extracted/chart/20231114102825506748.png"
QUESTION = "What is shown in this chart?"

print("\nTesting Intel/llava-gemma-2b model with original code...")
print(f"Transformers version: {__import__('transformers').__version__}")

# Define checkpoint
checkpoint = "Intel/llava-gemma-2b"
print(f"Using checkpoint: {checkpoint}")

# Load model and force CUDA usage
print("Loading model...")

# Check if CUDA is available, fail if not
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This script requires GPU acceleration.")

device = torch.device("cuda")
print(f"Using device: {device} - {torch.cuda.get_device_name(0)}")

# Load model directly to GPU with optimizations
model = LlavaForConditionalGeneration.from_pretrained(
    checkpoint,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("Model loaded successfully on GPU!")

print("Loading processor...")
# Load tokenizer and processor
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
processor = AutoProcessor.from_pretrained(checkpoint)
print("Processor loaded successfully!")

# Get absolute path for the image
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
image_path = os.path.join(project_root, IMAGE_PATH)

# Load local image
print(f"Loading image from: {image_path}")
image = Image.open(image_path).convert("RGB")
print(f"Image loaded: {image.size[0]}x{image.size[1]}")

# Try a completely different approach - use a URL image instead
print("Trying with a different image from URL...")
try:
    url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    print(f"Loading image from URL: {url}")
    image_from_url = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    print(f"URL image loaded: {image_from_url.size[0]}x{image_from_url.size[1]}")
except Exception as e:
    print(f"Failed to load URL image: {e}")
    image_from_url = None

# Prepare inputs using the exact approach from the official example
print("Preparing inputs using official example approach...")
prompt = processor.tokenizer.apply_chat_template(
    [{'role': 'user', 'content': f"<image>\n{QUESTION}"}],
    tokenize=False,
    add_generation_prompt=True
)

# Try with both images
for img_idx, img in enumerate([image_from_url, image]):
    if img is None:
        continue
        
    img_name = "URL image" if img_idx == 0 else "Local image"
    print(f"\nTrying with {img_name}...")
    
    try:
        # Process inputs exactly as in the official example
        inputs = processor(text=prompt, images=img, return_tensors="pt")
        
        # Move to GPU
        inputs = {k: v.to(device) for k, v in inputs.items()}
        print(f"Inputs processed successfully for {img_name}!")
        
        # Print shapes for debugging
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
        
        # Generate
        print(f"Generating response for {img_name}...")
        with torch.amp.autocast('cuda'):
            generate_ids = model.generate(**inputs, max_length=100)
        
        output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print(f"\nModel response for {img_name}:")
        print(output)
        
        # If we got here successfully, break the loop
        break
    except Exception as e:
        print(f"Error with {img_name}: {e}")

# If both approaches failed, try a more direct approach
if 'output' not in locals():
    print("\nTrying with a more direct approach...")
    
    # Try to inspect the model to understand what it expects
    print("Model config:")
    print(f"  Type: {type(model)}")
    print(f"  Vision encoder: {type(model.vision_tower) if hasattr(model, 'vision_tower') else 'N/A'}")
    
    # Try to find the expected image size from the model config
    if hasattr(model.config, "vision_config"):
        image_size = getattr(model.config.vision_config, "image_size", 224)
        print(f"  Expected image size from config: {image_size}")
    else:
        image_size = 224
        print(f"  Using default image size: {image_size}")
    
    # Resize image to the expected size
    resized_image = image.resize((image_size, image_size))
    print(f"  Resized image to: {resized_image.size[0]}x{resized_image.size[1]}")
    
    # Try with the resized image
    try:
        inputs = processor(text=prompt, images=resized_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.amp.autocast('cuda'):
            generate_ids = model.generate(**inputs, max_length=100)
        
        output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print("\nModel response with resized image:")
        print(output)
    except Exception as e:
        print(f"Error with resized image: {e}")
        
        # Last resort: try to use the model's own preprocessing
        print("\nTrying with model's own preprocessing...")
        try:
            # If the model has a vision tower, try to use it directly
            if hasattr(model, "vision_tower") and model.vision_tower is not None:
                from torchvision import transforms
                
                # Create a transform that matches CLIP preprocessing
                transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                         std=[0.26862954, 0.26130258, 0.27577711])
                ])
                
                # Process the image
                image_tensor = transform(image).unsqueeze(0).to(device)
                print(f"  Processed image tensor shape: {image_tensor.shape}")
                
                # Tokenize the text
                text_tokens = tokenizer(prompt, return_tensors="pt").to(device)
                
                # Forward pass
                with torch.no_grad():
                    with torch.amp.autocast('cuda'):
                        outputs = model(
                            input_ids=text_tokens.input_ids,
                            attention_mask=text_tokens.attention_mask,
                            pixel_values=image_tensor,
                            return_dict=True
                        )
                
                print("  Forward pass successful!")
                print("  Attempting generation...")
                
                # Generate text
                with torch.amp.autocast('cuda'):
                    generate_ids = model.generate(
                        input_ids=text_tokens.input_ids,
                        pixel_values=image_tensor,
                        max_length=100
                    )
                
                output = tokenizer.decode(generate_ids[0], skip_special_tokens=True)
                print("\nModel response with direct preprocessing:")
                print(output)
            else:
                print("  Model does not have a vision tower attribute")
        except Exception as e:
            print(f"  Error with direct preprocessing: {e}")
