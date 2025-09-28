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

def create_classifier(device='cuda:0', requires_grad=True, probabilistic=False, count_queries=True):
    """Create a GPU-only PyTorch classifier for the attack with optional query counting"""
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

    # Add query counting if requested
    if count_queries:
        classifier = add_query_counting_to_classifier(classifier, query_counter)

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

    Handles mixed input ranges correctly:
    - If both images are in [0,1]: use directly
    - If both images are in [0,255]: normalize both by 255
    - If mixed ranges: normalize each individually
    """
    # Convert to float32 for consistent processing
    img1_float = img1.astype(np.float32)
    img2_float = img2.astype(np.float32)

    # Determine range for each image independently
    img1_max = img1_float.max()
    img2_max = img2_float.max()

    # Normalize each image to [0,1] based on its own range
    if img1_max > 1.0:
        img1_norm = img1_float / 255.0
        logger.debug(f"Image 1 normalized from [0,{img1_max:.1f}] to [0,1] range")
    else:
        img1_norm = img1_float
        logger.debug(f"Image 1 already in [0,{img1_max:.3f}] range")

    if img2_max > 1.0:
        img2_norm = img2_float / 255.0
        logger.debug(f"Image 2 normalized from [0,{img2_max:.1f}] to [0,1] range")
    else:
        img2_norm = img2_float
        logger.debug(f"Image 2 already in [0,{img2_max:.3f}] range")

    # Calculate L∞ epsilon in normalized [0,1] space
    epsilon = float(np.max(np.abs(img1_norm - img2_norm)))

    # Debug output to verify calculation
    logger.debug(f"Epsilon calculation: {epsilon:.6f} (in normalized [0,1] space)")
    logger.debug(f"Image ranges - Original: [0,{img1_max:.1f}], Adversarial: [0,{img2_max:.1f}]")

    return epsilon

def calculate_l2_norm(img1, img2):
    """
    Calculate L2 norm (Euclidean distance) between two images in normalized [0,1] space

    Args:
        img1: Original image
        img2: Adversarial image

    Returns:
        float: L2 norm of the perturbation
    """
    # Convert to float32 for consistent processing
    img1_float = img1.astype(np.float32)
    img2_float = img2.astype(np.float32)

    # Normalize to [0,1] space similar to calculate_epsilon
    img1_max = img1_float.max()
    img2_max = img2_float.max()

    if img1_max > 1.0:
        img1_norm = img1_float / 255.0
    else:
        img1_norm = img1_float

    if img2_max > 1.0:
        img2_norm = img2_float / 255.0
    else:
        img2_norm = img2_float

    # Calculate L2 norm of perturbation
    perturbation = img2_norm - img1_norm
    l2_norm = float(np.sqrt(np.sum(perturbation ** 2)))

    logger.debug(f"L2 norm calculation: {l2_norm:.6f} (in normalized [0,1] space)")
    return l2_norm

def calculate_l0_norm(img1, img2, threshold=1e-6):
    """
    Calculate L0 norm (number of changed pixels) between two images

    Args:
        img1: Original image
        img2: Adversarial image
        threshold: Minimum change to consider a pixel as modified

    Returns:
        int: Number of pixels that changed
    """
    # Convert to float32 for consistent processing
    img1_float = img1.astype(np.float32)
    img2_float = img2.astype(np.float32)

    # Normalize to [0,1] space similar to calculate_epsilon
    img1_max = img1_float.max()
    img2_max = img2_float.max()

    if img1_max > 1.0:
        img1_norm = img1_float / 255.0
    else:
        img1_norm = img1_float

    if img2_max > 1.0:
        img2_norm = img2_float / 255.0
    else:
        img2_norm = img2_float

    # Calculate absolute difference
    diff = np.abs(img2_norm - img1_norm)

    # Count pixels where any channel has changed more than threshold
    if len(diff.shape) == 3:  # Color image
        # Check if any channel in each pixel changed
        changed_pixels = np.any(diff > threshold, axis=2)
    else:  # Grayscale
        changed_pixels = diff > threshold

    l0_norm = int(np.sum(changed_pixels))

    logger.debug(f"L0 norm calculation: {l0_norm} pixels changed (threshold: {threshold})")
    return l0_norm


def refine_epsilon_tolerance(original_image: np.ndarray, adversarial_image: np.ndarray,
                           target_epsilon: float, tolerance: float = 0.05, max_iterations: int = 100) -> tuple:
    """
    Iteratively refine adversarial image to achieve target epsilon ±tolerance

    This function performs post-processing epsilon refinement to ensure the final
    L∞ epsilon matches the target within specified tolerance. Uses iterative scaling
    of perturbations with safeguards against numerical instability.

    Args:
        original_image: Original clean image (any format)
        adversarial_image: Initial adversarial image (any format)
        target_epsilon: Target epsilon value in [0,1] normalized space
        tolerance: Acceptable tolerance as fraction (default 0.05 = ±5%)
        max_iterations: Maximum refinement iterations (default 10)

    Returns:
        Tuple of (refined_adversarial_image, final_epsilon)

    Example:
        >>> refined_adv, final_eps = refine_epsilon_tolerance(
        ...     original, adversarial, target_epsilon=0.02, tolerance=0.05
        ... )
        >>> # final_eps will be in range [0.019, 0.021] (±5% of 0.02)
    """
    logger.info(f"🔄 Refining epsilon to target {target_epsilon:.6f} ±{tolerance*100:.1f}%")

    # Work with copies to avoid modifying originals
    current_adv = adversarial_image.copy()
    original_work = original_image.copy()

    # Calculate tolerance bounds
    tolerance_range = target_epsilon * tolerance
    target_min = target_epsilon - tolerance_range
    target_max = target_epsilon + tolerance_range

    logger.debug(f"Target range: [{target_min:.6f}, {target_max:.6f}]")

    for iteration in range(max_iterations):
        # Calculate current epsilon using the improved calculation
        current_epsilon = calculate_epsilon(original_work, current_adv)

        logger.debug(f"Iteration {iteration+1}: epsilon={current_epsilon:.6f}")

        # Check if within tolerance
        if target_min <= current_epsilon <= target_max:
            logger.info(f"✅ Epsilon tolerance achieved: {current_epsilon:.6f} (target: {target_epsilon:.6f})")
            return current_adv, current_epsilon

        # Skip refinement if current epsilon is too small (avoid division by zero)
        if current_epsilon < 1e-8:
            logger.warning(f"⚠️  Current epsilon too small ({current_epsilon:.8f}), cannot refine")
            break

        # Calculate adjustment factor based on epsilon ratio (more aggressive for faster convergence)
        if current_epsilon < target_min:
            # Need to increase perturbation
            scale_factor = target_epsilon / current_epsilon
            scale_factor = min(scale_factor, 2.0)  # More aggressive growth for faster convergence
            logger.debug(f"Increasing perturbation: scale={scale_factor:.4f}")
        else:
            # Need to decrease perturbation
            scale_factor = target_epsilon / current_epsilon
            scale_factor = max(scale_factor, 0.3)  # More aggressive reduction for faster convergence
            logger.debug(f"Decreasing perturbation: scale={scale_factor:.4f}")

        # Convert to float for precise calculations
        original_float = original_work.astype(np.float32)
        current_float = current_adv.astype(np.float32)

        # Calculate current perturbation
        perturbation = current_float - original_float

        # Scale perturbation
        scaled_perturbation = perturbation * scale_factor

        # Apply scaled perturbation
        new_adv_float = original_float + scaled_perturbation

        # Clip to valid range based on image format
        if original_work.max() <= 1.0:
            # [0,1] range
            current_adv = np.clip(new_adv_float, 0, 1).astype(original_work.dtype)
        else:
            # [0,255] range
            current_adv = np.clip(new_adv_float, 0, 255).astype(original_work.dtype)

    # Final epsilon calculation
    final_epsilon = calculate_epsilon(original_work, current_adv)

    if final_epsilon < target_min or final_epsilon > target_max:
        logger.warning(f"⚠️  Max iterations reached. Final epsilon: {final_epsilon:.6f} (target: {target_epsilon:.6f})")
    else:
        logger.info(f"✅ Final epsilon within tolerance: {final_epsilon:.6f}")

    return current_adv, final_epsilon

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

def print_attack_info(output_path, original_image, adv_image, attack_type, query_count=None):
    """Print information about the attack using optimized calculations"""
    # Use optimized perturbation calculation
    perturbation = fast_perturbation_calculation(original_image, adv_image)
    print(f"Max perturbation: {float(np.max(perturbation))}")
    print(f"Mean perturbation: {float(np.mean(perturbation))}")

    # Use epsilon calculation instead of SSIM
    epsilon_val = calculate_epsilon(original_image, adv_image)
    print(f"Epsilon (L∞): {float(epsilon_val)}")

    # Calculate L2 norm
    l2_val = calculate_l2_norm(original_image, adv_image)
    print(f"L2 norm: {float(l2_val)}")

    # Calculate L0 norm
    l0_val = calculate_l0_norm(original_image, adv_image)
    print(f"L0 norm: {int(l0_val)}")

    # Print query count if available
    if query_count is not None:
        print(f"Total queries: {int(query_count)}")

    print("\nTo use this adversarial image in evaluation:")
    print(f"1. The image is saved at: {output_path}")
    print("2. When running eval_model.py, the script will use the original path")
    print("3. To use adversarial images, modify the img_path in eval_model.py:")
    print("   Change: img_path = 'data/clean/' + data['image']")
    box_type = 'blackbox' if 'black_box' in output_path else 'whitebox'
    print(f"   To:     img_path = 'data/adversarial/{box_type}/{attack_type}/' + data['image']")


class QueryCounter:
    """Global query counter for tracking model evaluations during attacks"""
    _instance = None
    _counter = 0

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QueryCounter, cls).__new__(cls)
        return cls._instance

    def reset(self):
        """Reset the query counter"""
        self._counter = 0

    def increment(self, count=1):
        """Increment the query counter"""
        self._counter += count

    def get_count(self):
        """Get current query count"""
        return self._counter

# Global query counter instance
query_counter = QueryCounter()

def add_query_counting_to_classifier(classifier, query_counter):
    """Add query counting to an existing classifier by patching its methods"""
    # Store original methods
    original_predict = classifier.predict
    original_loss_gradient = classifier.loss_gradient

    def counting_predict(x, batch_size=128, **kwargs):
        """Predict method that counts queries"""
        # Count the number of samples being processed
        if hasattr(x, 'shape'):
            batch_count = x.shape[0] if len(x.shape) > 0 else 1
        else:
            batch_count = 1

        query_counter.increment(batch_count)
        return original_predict(x, batch_size=batch_size, **kwargs)

    def counting_loss_gradient(x, y, **kwargs):
        """Loss gradient method that counts queries"""
        # Count the number of samples being processed
        if hasattr(x, 'shape'):
            batch_count = x.shape[0] if len(x.shape) > 0 else 1
        else:
            batch_count = 1

        query_counter.increment(batch_count)
        return original_loss_gradient(x, y, **kwargs)

    # Patch the methods
    classifier.predict = counting_predict
    classifier.loss_gradient = counting_loss_gradient

    return classifier

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