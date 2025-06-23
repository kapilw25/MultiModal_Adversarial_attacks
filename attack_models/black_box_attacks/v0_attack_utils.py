#!/usr/bin/env python3
"""
Utility functions for adversarial attacks on Vision-Language Models

This module contains common functions used across different attack implementations:
- load_image: Load and preprocess an image
- create_classifier: Create a PyTorch classifier for the attack
- save_image: Save the image to the specified path
"""

import os
import cv2
import numpy as np
import torch
from torchvision import transforms, models
from art.estimators.classification import PyTorchClassifier


def load_image(image_path):
    """Load and preprocess an image for the model
    
    Args:
        image_path (str): Path to the input image
        
    Returns:
        numpy.ndarray: RGB image in numpy array format
        
    Raises:
        ValueError: If the image cannot be loaded
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def create_classifier(device='cuda:0'):
    """Create a PyTorch classifier for the attack
    
    Args:
        device (str): Device to use for computation ('cuda:0', 'cpu', etc.)
        
    Returns:
        PyTorchClassifier: ART classifier wrapper around a PyTorch model
    """
    # Use a pre-trained ResNet model as a substitute model for the attack
    # This is because we don't have direct access to the VLM's vision encoder
    model = models.resnet50(pretrained=True)
    model.to(device).eval()
    
    # Define preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Create ART classifier
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0.0, 1.0),
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=(3, 224, 224),
        nb_classes=1000,
        preprocessing=(mean, std),
        device_type=device
    )
    
    return classifier


def save_image(image, output_path):
    """Save the image to the specified path
    
    Args:
        image (numpy.ndarray): RGB image to save
        output_path (str): Path where the image will be saved
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save image
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"Saved adversarial image to {output_path}")


def preprocess_image_for_attack(image):
    """Preprocess an image for attack (resize and convert to tensor)
    
    Args:
        image (numpy.ndarray): RGB image in numpy array format
        
    Returns:
        numpy.ndarray: Preprocessed image as a numpy array with shape (1, 3, 224, 224)
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Convert to tensor and add batch dimension
    img_tensor = transform(image).unsqueeze(0).numpy()
    return img_tensor


def postprocess_adversarial_image(adv_image, original_shape):
    """Convert adversarial tensor back to image format and resize to original dimensions
    
    Args:
        adv_image (numpy.ndarray): Adversarial image as numpy array with shape (1, 3, H, W)
        original_shape (tuple): Original image shape (H, W, C)
        
    Returns:
        numpy.ndarray: Processed adversarial image as uint8 numpy array
    """
    # Convert back to uint8 format
    adv_image = adv_image[0].transpose(1, 2, 0)
    adv_image = np.clip(adv_image, 0, 1) * 255
    adv_image = adv_image.astype(np.uint8)
    
    # Resize back to original dimensions
    adv_image = cv2.resize(adv_image, (original_shape[1], original_shape[0]))
    
    return adv_image


def get_output_path(input_path, attack_type):
    """Generate output path for adversarial image based on input path and attack type
    
    Args:
        input_path (str): Path to the input image
        attack_type (str): Type of attack (e.g., 'pgd', 'fgsm', 'cw_l2', etc.)
        
    Returns:
        str: Output path for the adversarial image
    """
    input_dir = os.path.dirname(input_path)
    filename = os.path.basename(input_path)
    
    # Map attack type to directory suffix
    attack_dir_map = {
        'pgd': 'adv',
        'fgsm': 'adv_fgsm',
        'cw_l2': 'adv_cw_l2',
        'cw_l0': 'adv_cw_l0',
        'cw_linf': 'adv_cw_linf'
    }
    
    dir_suffix = attack_dir_map.get(attack_type, f'adv_{attack_type}')
    output_dir = input_dir.replace('test_extracted', f'test_extracted_{dir_suffix}')
    output_path = os.path.join(output_dir, filename)
    
    return output_path


def print_attack_info(output_path, original_image, adv_image, attack_type):
    """Print information about the attack and instructions for evaluation
    
    Args:
        output_path (str): Path where the adversarial image was saved
        original_image (numpy.ndarray): Original image
        adv_image (numpy.ndarray): Adversarial image
        attack_type (str): Type of attack
    """
    # Print perturbation statistics
    perturbation = np.abs(original_image.astype(np.float32) - adv_image.astype(np.float32))
    print(f"Max perturbation: {np.max(perturbation)}")
    print(f"Mean perturbation: {np.mean(perturbation)}")
    
    # Get directory suffix based on attack type
    attack_dir_map = {
        'pgd': 'adv',
        'fgsm': 'adv_fgsm',
        'cw_l2': 'adv_cw_l2',
        'cw_l0': 'adv_cw_l0',
        'cw_linf': 'adv_cw_linf'
    }
    dir_suffix = attack_dir_map.get(attack_type, f'adv_{attack_type}')
    
    print("\nTo use this adversarial image in evaluation:")
    print(f"1. The image is saved at: {output_path}")
    print("2. When running eval_model.py, the script will use the original path")
    print("3. To use adversarial images, modify the img_path in eval_model.py:")
    print("   Change: img_path = 'data/test_extracted/' + data['image']")
    print(f"   To:     img_path = 'data/test_extracted_{dir_suffix}/' + data['image']")
