#!/usr/bin/env python3
"""
Utility functions for white-box adversarial attacks on Vision-Language Models

This module contains common functions used across different white-box attack implementations:
- load_image: Load and preprocess an image
- load_model: Load the Qwen2.5-VL-3B model with gradient access
- save_image: Save the image to the specified path
- process_vision_info: Process vision information for the model
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


def load_image(image_path):
    """Load and preprocess an image for the model
    
    Args:
        image_path (str): Path to the input image
        
    Returns:
        PIL.Image: Image in PIL format
        
    Raises:
        ValueError: If the image cannot be loaded
    """
    try:
        img = Image.open(image_path).convert('RGB')
        return img
    except Exception as e:
        raise ValueError(f"Could not load image from {image_path}: {str(e)}")


def load_model(device='cuda'):
    """Load the Qwen2.5-VL-3B model with gradient access and memory optimizations
    
    Args:
        device (str): Device to use for computation ('cuda', 'cpu')
        
    Returns:
        tuple: (model, processor) - The model and processor for Qwen2.5-VL-3B
    """
    print("Loading Qwen2.5-VL-3B model with memory optimizations...")
    
    # Free up CUDA memory before loading model
    if device == 'cuda':
        import torch
        torch.cuda.empty_cache()
        print(f"CUDA memory before loading: {torch.cuda.memory_allocated() / 1024**2:.2f} MB allocated")
    
    # Load model with memory optimizations
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from transformers import BitsAndBytesConfig
    
    # Configure 4-bit quantization with better gradient computation support
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"  # Normalized float 4-bit quantization
    )
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        quantization_config=quantization_config,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    
    # Enable gradient checkpointing to save memory
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    # Set model to training mode to enable gradient computation
    model.train()
    
    # Load processor with recommended pixel settings
    min_pixels = 224*224*3  # Reduced from default to save memory
    max_pixels = 224*224*3  # Reduced from default to save memory
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", 
        min_pixels=min_pixels, 
        max_pixels=max_pixels
    )
    print("Model and processor loaded successfully")
    
    if device == 'cuda':
        print(f"CUDA memory after loading: {torch.cuda.memory_allocated() / 1024**2:.2f} MB allocated")
    
    return model, processor
    
    print("Model and processor loaded successfully")
    return model, processor


def process_vision_info(messages):
    """Process vision information from messages
    
    Args:
        messages (list): List of message dictionaries with content
        
    Returns:
        tuple: (image_inputs, video_inputs) - Processed image and video inputs
    """
    image_inputs = []
    video_inputs = []
    
    for message in messages:
        if message["role"] == "user":
            for content in message["content"]:
                if content["type"] == "image":
                    if "image" in content:
                        image_inputs.append(content["image"])
    
    return image_inputs, video_inputs


def save_image(image_tensor, output_path):
    """Save the image tensor to the specified path
    
    Args:
        image_tensor (torch.Tensor): Image tensor to save
        output_path (str): Path where the image will be saved
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert tensor to numpy array and save
    from torchvision.utils import save_image
    save_image(image_tensor, output_path)
    print(f"Saved adversarial image to {output_path}")


def get_output_path(input_path, attack_type):
    """Generate output path for adversarial image based on input path and attack type
    
    Args:
        input_path (str): Path to the input image
        attack_type (str): Type of attack (e.g., 'white_box_fgsm', 'white_box_pgd', etc.)
        
    Returns:
        str: Output path for the adversarial image
    """
    input_dir = os.path.dirname(input_path)
    filename = os.path.basename(input_path)
    
    # Map attack type to directory suffix
    attack_dir_map = {
        'white_box_fgsm': 'adv_white_box_fgsm',
        'white_box_pgd': 'adv_white_box_pgd',
        'white_box_cw': 'adv_white_box_cw',
    }
    
    dir_suffix = attack_dir_map.get(attack_type, f'adv_{attack_type}')
    output_dir = input_dir.replace('test_extracted', f'test_extracted_{dir_suffix}')
    output_path = os.path.join(output_dir, filename)
    
    return output_path


def print_attack_info(output_path, original_tensor, adv_tensor, attack_type):
    """Print information about the attack and instructions for evaluation
    
    Args:
        output_path (str): Path where the adversarial image was saved
        original_tensor (torch.Tensor): Original image tensor
        adv_tensor (torch.Tensor): Adversarial image tensor
        attack_type (str): Type of attack
    """
    # Convert tensors to numpy for comparison
    original_np = original_tensor.detach().cpu().numpy()
    adv_np = adv_tensor.detach().cpu().numpy()
    
    # Print perturbation statistics
    perturbation = np.abs(original_np - adv_np)
    print(f"Max perturbation: {np.max(perturbation)}")
    print(f"Mean perturbation: {np.mean(perturbation)}")
    
    # Get directory suffix based on attack type
    attack_dir_map = {
        'white_box_fgsm': 'adv_white_box_fgsm',
        'white_box_pgd': 'adv_white_box_pgd',
        'white_box_cw': 'adv_white_box_cw',
    }
    dir_suffix = attack_dir_map.get(attack_type, f'adv_{attack_type}')
    
    print("\nTo use this adversarial image in evaluation:")
    print(f"1. The image is saved at: {output_path}")
    print("2. When running eval_model.py, the script will use the original path")
    print("3. To use adversarial images, modify the img_path in eval_model.py:")
    print("   Change: img_path = 'data/test_extracted/' + data['image']")
    print(f"   To:     img_path = 'data/test_extracted_{dir_suffix}/' + data['image']")


def cleanup_model(model):
    """Clean up GPU resources
    
    Args:
        model: The model to clean up
    """
    del model
    torch.cuda.empty_cache()
    print("Model resources cleaned up")
