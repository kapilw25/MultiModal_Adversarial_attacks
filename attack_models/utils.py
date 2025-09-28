#!/usr/bin/env python3
"""
Common Unified Attack Utilities

This module provides shared utilities for both white-box and black-box adversarial attacks.
Contains GPU optimizations, image processing, and epsilon-based utilities.

Shared Functions:
- GPU-optimized computations (JIT-compiled)
- Image loading, preprocessing, postprocessing
- Perturbation calculation
- Classifier creation
- Directory and path utilities
"""

import os
import cv2
import numpy as np
import torch
from torchvision import transforms, models
from art.estimators.classification import PyTorchClassifier
# Removed SSIM dependency - using epsilon-based metrics only
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
import time

# GPU-optimized implementation with Numba JIT
import numba
from numba import jit

# Configure unified logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# JIT-compiled optimization functions - GPU acceleration required
@jit(nopython=True)
def fast_perturbation_calculation(image1, image2):
    """GPU-optimized perturbation calculation using Numba JIT"""
    return np.abs(image1.astype(np.float32) - image2.astype(np.float32))

@jit(nopython=True)
def fast_clip_operation(array, min_val, max_val):
    """GPU-optimized clipping operation"""
    return np.clip(array, min_val, max_val)

# PyTorch JIT optimization for tensor operations
@torch.jit.script
def optimized_tensor_perturbation(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """JIT-compiled tensor perturbation calculation"""
    return torch.abs(tensor1 - tensor2)

# Global mapping of attack types to directory suffixes
ATTACK_DIR_MAP = {
    'pgd': 'pgd', 'fgsm': 'fgsm', 'auto_pgd': 'auto_pgd', 'auto_conjugate_gradient': 'auto_conjugate_gradient',
    'basic_iterative': 'basic_iterative', 'deepfool': 'deepfool',
    'square': 'square', 'simba': 'simba', 'boundary': 'boundary', 'sign_opt': 'sign_opt'
}

def load_image(image_path):
    """Load and preprocess an image for the model"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def create_classifier(device='cuda:0', requires_grad=True, probabilistic=False):
    """Create a GPU-only PyTorch classifier for the attack"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. CPU fallback is not supported.")

    model = models.resnet50(pretrained=True)
    model.to(device)

    # Set model to training mode for white-box attacks that need gradients
    if requires_grad:
        model.train()
        # Enable gradients for all parameters
        for param in model.parameters():
            param.requires_grad = True
    else:
        model.eval()

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Add softmax for probabilistic outputs if needed
    if probabilistic:
        class ProbabilisticModel(torch.nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model
                self.softmax = torch.nn.Softmax(dim=1)

            def forward(self, x):
                logits = self.base_model(x)
                return self.softmax(logits)

        model = ProbabilisticModel(model)

    classifier = PyTorchClassifier(
        model=model, clip_values=(0.0, 1.0), loss=torch.nn.CrossEntropyLoss(),
        input_shape=(3, 224, 224), nb_classes=1000, preprocessing=(mean, std),
        device_type=device
    )
    return classifier

def save_image(image, output_path):
    """Save the image to the specified path"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Handle both tensor and numpy array inputs
    if torch.is_tensor(image):
        # Convert tensor to numpy array
        if image.dim() == 4:  # Batch dimension
            image = image.squeeze(0)
        if image.dim() == 3 and image.shape[0] in [1, 3]:  # CHW format
            image = image.permute(1, 2, 0)
        image = image.detach().cpu().numpy()

    # Ensure image is in correct format for OpenCV
    if image.dtype != np.uint8:
        # Normalize to 0-255 if needed
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)

    # Ensure image is numpy array
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Image must be numpy array or tensor, got {type(image)}")

    # Convert RGB to BGR for OpenCV
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image

    cv2.imwrite(output_path, image_bgr)
    print(f"Saved adversarial image to {output_path}")

def preprocess_image_for_attack(image, return_tensor=False):
    """Preprocess an image for attack"""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(image).unsqueeze(0).float()
    return img_tensor if return_tensor else img_tensor.numpy()

def postprocess_adversarial_image(adv_image, original_shape):
    """Convert adversarial tensor back to image format"""
    adv_image = adv_image[0].transpose(1, 2, 0)
    adv_image = np.clip(adv_image, 0, 1) * 255
    adv_image = adv_image.astype(np.uint8)
    adv_image = cv2.resize(adv_image, (original_shape[1], original_shape[0]))
    return adv_image

def get_output_path(input_path, attack_type, is_blackbox=False, epsilon=None):
    """Generate output path for adversarial image"""
    input_dir = os.path.dirname(input_path)
    filename = os.path.basename(input_path)
    dir_suffix = ATTACK_DIR_MAP.get(attack_type, attack_type)
    box_type = 'blackbox' if is_blackbox else 'whitebox'

    # Add epsilon folder structure
    if epsilon is not None:
        eps_dir = f"eps_{epsilon:.3f}".replace(".", "")  # 0.050 -> eps_050
        output_dir = input_dir.replace('clean', f'adversarial/{box_type}/{dir_suffix}/{eps_dir}')
    else:
        output_dir = input_dir.replace('clean', f'adversarial/{box_type}/{dir_suffix}')

    return os.path.join(output_dir, filename)

def calculate_epsilon(img1, img2):
    """
    Calculate L-infinity epsilon between two images in normalized [0,1] space

    This function ensures epsilon calculation matches the space where adversarial
    attacks operate (normalized [0,1]), not the final image space [0,255].
    """
    # Always normalize to [0,1] for proper epsilon calculation in attack space
    if img1.max() > 1.0 or img2.max() > 1.0:
        img1_norm = img1.astype(np.float32) / 255.0
        img2_norm = img2.astype(np.float32) / 255.0
    else:
        img1_norm = img1.astype(np.float32)
        img2_norm = img2.astype(np.float32)

    epsilon = float(np.max(np.abs(img1_norm - img2_norm)))

    # Debug output to verify calculation
    logger.debug(f"Epsilon calculation: {epsilon:.6f} (in normalized [0,1] space)")

    return epsilon

# GPU-optimized epsilon calculation for tensors
def calculate_epsilon_tensor(tensor1, tensor2):
    """GPU-optimized epsilon calculation for PyTorch tensors"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. CPU fallback is not supported.")

    with torch.cuda.amp.autocast():
        # Convert tensors to numpy for epsilon calculation
        img1 = tensor1.squeeze().cpu().numpy()
        img2 = tensor2.squeeze().cpu().numpy()

        # Transpose from CHW to HWC if needed
        if len(img1.shape) == 3 and img1.shape[0] == 3:
            img1 = np.transpose(img1, (1, 2, 0))
            img2 = np.transpose(img2, (1, 2, 0))

        # Convert to uint8 for epsilon calculation
        img1 = fast_clip_operation(img1 * 255, 0, 255).astype(np.uint8)
        img2 = fast_clip_operation(img2 * 255, 0, 255).astype(np.uint8)

        return calculate_epsilon(img1, img2)

def print_attack_info(output_path, original_image, adv_image, attack_type):
    """Print information about the attack using optimized calculations"""
    # Use optimized perturbation calculation
    perturbation = fast_perturbation_calculation(original_image, adv_image)
    print(f"Max perturbation: {float(np.max(perturbation))}")
    print(f"Mean perturbation: {float(np.mean(perturbation))}")

    # Use epsilon calculation instead of SSIM
    epsilon_val = calculate_epsilon(original_image, adv_image)
    print(f"Epsilon (L∞): {float(epsilon_val)}")

    print("\nTo use this adversarial image in evaluation:")
    print(f"1. The image is saved at: {output_path}")
    print("2. When running eval_model.py, the script will use the original path")
    print("3. To use adversarial images, modify the img_path in eval_model.py:")
    print("   Change: img_path = 'data/clean/' + data['image']")
    box_type = 'blackbox' if 'black_box' in output_path else 'whitebox'
    print(f"   To:     img_path = 'data/adversarial/{box_type}/{attack_type}/' + data['image']")


class UniversalEpsilonOptimizer:
    """Unified epsilon-aware optimizer for both white-box and black-box attacks"""

    def __init__(self, target_epsilon: float = 0.05, tolerance: float = 0.01):
        self.target_epsilon = target_epsilon
        self.tolerance = tolerance
        self.best_result = None
        self.query_count = 0

    def calculate_objective_score(self, image: np.ndarray, adv_image: np.ndarray) -> float:
        """Calculate epsilon-based objective score (lower is better)"""
        achieved_epsilon = calculate_epsilon(image, adv_image)
        score = abs(achieved_epsilon - self.target_epsilon)
        return score, achieved_epsilon

# REMOVED: bayesian_optimization_framework - No longer needed for direct epsilon control
# def bayesian_optimization_framework(...) - DEPRECATED
# This function has been removed as the new epsilon-based approach uses direct parameter control
# without optimization. All attacks now accept epsilon parameters directly for predictable results.