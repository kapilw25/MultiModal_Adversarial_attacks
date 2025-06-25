#!/usr/bin/env python3
"""
White-Box FGSM Attack Script for Vision-Language Models

This script applies a Fast Gradient Sign Method (FGSM) adversarial attack directly to images
using the gradients from the Qwen2.5-VL-3B model. This is a true white-box attack as it
uses direct access to the model's gradients rather than a substitute model.

The FGSM attack is defined mathematically as:
- For untargeted attacks: x' = x + ε · sign(∇ₓJ(θ, x, y))
- For targeted attacks:   x' = x - ε · sign(∇ₓJ(θ, x, t))

Where:
- x is the original input image
- x' is the adversarial example
- ε is the perturbation magnitude (controls how much each pixel can change)
- J is the loss function
- θ represents the model parameters
- y is the true label
- t is the target label
- ∇ₓJ represents the gradient of the loss with respect to the input x
- sign() is the sign function that returns -1, 0, or 1 depending on the sign of its input

Usage:
    python v3_fgsm_attack.py [--image_path PATH] [--eps EPSILON] [--question QUESTION]

Example:
    # Standard attack
    python v3_fgsm_attack.py --image_path data/test_extracted/chart/image.png --eps 0.03 --question "Describe this chart in detail."
"""

import os
import numpy as np
import argparse
import torch
from PIL import Image
import time

# Import utility functions
from v0_attack_utils import (
    load_image, load_model, save_image, 
    get_output_path, print_attack_info, process_vision_info, cleanup_model
)


def white_box_fgsm_attack(model, processor, image_path, question, eps=0.03):
    """Apply white-box FGSM attack to the image using direct gradients from the VLM
    
    Args:
        model: The Qwen2.5-VL-3B model
        processor: The model's processor
        image_path (str): Path to the input image
        question (str): Question to ask the model
        eps (float): Epsilon parameter controlling perturbation magnitude
        
    Returns:
        tuple: (output_path, original_tensor, adversarial_tensor)
    """
    print(f"Loading image from {image_path}")
    image = load_image(image_path)
    
    # Free up memory before processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"CUDA memory before processing: {torch.cuda.memory_allocated() / 1024**2:.2f} MB allocated")
    
    # Use a shorter question to reduce memory usage
    short_question = "Describe this chart." if len(question) > 20 else question
    print(f"Using question: '{short_question}'")
    
    # Prepare messages for the model
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": short_question}
        ]}
    ]
    
    # Process inputs
    print("Processing inputs...")
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Convert PIL image to tensor and require gradients
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to smaller dimensions to save memory
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0).to(model.device)
    image_tensor.requires_grad = True
    
    # Process inputs with the tensor that requires gradients - skip video processing
    inputs = processor(
        text=[text],
        images=image_tensor,
        videos=None,  # Set to None to skip video processing
        padding=True,
        return_tensors="pt",
        max_length=256,  # Limit sequence length to save memory
    )
    inputs = inputs.to(model.device)
    
    # Forward pass to compute loss
    print("Computing forward pass and loss...")
    with torch.enable_grad():
        # Create target labels (use input_ids as labels for next-token prediction)
        labels = inputs["input_ids"].clone()
        
        # Forward pass with memory optimization
        # Use mixed precision to save memory
        with torch.cuda.amp.autocast():
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
    
    # Compute gradients with respect to the image
    print("Computing gradients...")
    # Use gradient scaling to prevent underflow in 8-bit gradients
    scaler = torch.cuda.amp.GradScaler()
    scaled_loss = scaler.scale(loss)
    scaled_loss.backward()
    
    # Extract gradients from the image tensor
    if image_tensor.grad is None:
        print("Warning: No gradients computed. Using estimated gradients.")
        # Use finite differences to estimate gradients
        delta = 0.01
        with torch.no_grad():
            # Create a detached copy for perturbation
            perturbed_tensor = image_tensor.detach().clone()
            perturbed_tensor = perturbed_tensor + delta
            
            # Process perturbed image
            perturbed_inputs = processor(
                text=[text],
                images=perturbed_tensor,
                videos=None,
                padding=True,
                return_tensors="pt",
                max_length=256,
            )
            perturbed_inputs = perturbed_inputs.to(model.device)
            
            # Compute loss for perturbed image
            perturbed_outputs = model(**perturbed_inputs, labels=labels)
            perturbed_loss = perturbed_outputs.loss
            
            # Estimate gradient using finite differences
            estimated_grad = (perturbed_loss - loss) / delta
            image_grad = torch.ones_like(image_tensor) * estimated_grad
            image_grad = image_grad.sign()
    else:
        print("Gradients computed successfully.")
        image_grad = image_tensor.grad.sign()
    
    # Apply FGSM perturbation
    print(f"Applying FGSM perturbation with eps={eps}")
    with torch.no_grad():
        perturbed_image = image_tensor + eps * image_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    # Get output path
    output_path = get_output_path(image_path, 'white_box_fgsm')
    
    # Save adversarial image
    save_image(perturbed_image[0], output_path)
    
    return output_path, image_tensor, perturbed_image
    
    return output_path, image_tensor, perturbed_image


def print_fgsm_info():
    """Print information about the white-box FGSM attack"""
    print("\nWhite-Box FGSM Attack Information:")
    print("- Fast Gradient Sign Method (FGSM) with direct access to VLM gradients")
    print("- Mathematical formulation: x' = x + ε · sign(∇ₓJ(θ, x, y))")
    print("- Uses the actual VLM's gradients rather than a substitute model")
    print("- More effective than black-box transfer attacks due to direct gradient access")
    
    print("\nComparison with Black-Box FGSM attack:")
    print("- White-Box: Uses direct gradients from the target VLM")
    print("- Black-Box: Uses gradients from a substitute model (e.g., ResNet50)")
    print("- White-Box attacks are typically more effective but require model access")
    print("- White-Box attacks avoid the 'transfer gap' present in black-box attacks")


def main():
    parser = argparse.ArgumentParser(description="Generate white-box adversarial examples using FGSM attack")
    parser.add_argument("--image_path", type=str, 
                        default="data/test_extracted/chart/20231114102825506748.png",
                        help="Path to the input image")
    parser.add_argument("--eps", type=float, default=0.03,
                        help="Perturbation magnitude (default: 0.03)")
    parser.add_argument("--question", type=str, default="Describe this chart in detail.",
                        help="Question to ask the model (default: 'Describe this chart in detail.')")
    args = parser.parse_args()
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device} with memory optimizations")
    
    # Load model
    start_time = time.time()
    model, processor = load_model(device)
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Apply white-box FGSM attack
    output_path, original_tensor, adv_tensor = white_box_fgsm_attack(
        model, processor, args.image_path, args.question, args.eps
    )
    
    # Print attack information
    print_attack_info(output_path, original_tensor, adv_tensor, 'white_box_fgsm')
    print_fgsm_info()
    
    # Clean up
    cleanup_model(model)


if __name__ == "__main__":
    main()
