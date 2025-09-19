#!/usr/bin/env python3
"""
Common Unified Attack Utilities

This module provides shared utilities for both white-box and black-box adversarial attacks.
Contains GPU optimizations, image processing, and SSIM calculation functions.

Shared Functions:
- GPU-optimized computations (JIT-compiled)  
- Image loading, preprocessing, postprocessing
- SSIM calculation
- Classifier creation
- Directory and path utilities
"""

import os
import cv2
import numpy as np
import torch
from torchvision import transforms, models
from art.estimators.classification import PyTorchClassifier
from skimage.metrics import structural_similarity as ssim
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import optuna
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
    'pgd': 'pgd', 'fgsm': 'fgsm', 'cw_l2': 'cw_l2', 'cw_l0': 'cw_l0',
    'cw_linf': 'cw_linf', 'jsma': 'jsma', 'deepfool': 'deepfool',
    'square': 'square', 'hop_skip_jump': 'hop_skip_jump', 'pixel': 'pixel',
    'simba': 'simba', 'spatial': 'spatial', 'query_efficient_bb': 'query_efficient_bb',
    'boundary': 'boundary', 'zoo': 'zoo'
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

def get_output_path(input_path, attack_type, is_blackbox=False, ssim_threshold=None):
    """Generate output path for adversarial image"""
    input_dir = os.path.dirname(input_path)
    filename = os.path.basename(input_path)
    dir_suffix = ATTACK_DIR_MAP.get(attack_type, attack_type)
    box_type = 'blackbox' if is_blackbox else 'whitebox'

    # Add SSIM folder structure
    if ssim_threshold is not None:
        ssim_dir = f"ssim_{ssim_threshold:.2f}".replace(".", "")  # 0.85 -> ssim_085
        output_dir = input_dir.replace('clean', f'adversarial/{box_type}/{dir_suffix}/{ssim_dir}')
    else:
        output_dir = input_dir.replace('clean', f'adversarial/{box_type}/{dir_suffix}')

    return os.path.join(output_dir, filename)

def calculate_ssim(img1, img2):
    """Optimized SSIM calculation with GPU acceleration when available"""
    # Convert to grayscale using optimized operations
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        img1_gray = img1
        img2_gray = img2
    
    # Use optimized min calculation
    min_size = min(img1_gray.shape[0], img1_gray.shape[1], img2_gray.shape[0], img2_gray.shape[1])
    win_size = min(7, min_size - 1)
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        img1_gray = cv2.resize(img1_gray, (64, 64))
        img2_gray = cv2.resize(img2_gray, (64, 64))
        win_size = 7
    
    # Use optimized SSIM calculation
    return ssim(img1_gray, img2_gray, data_range=255, win_size=win_size)

# GPU-only optimized SSIM for tensors
def calculate_ssim_tensor(tensor1, tensor2):
    """GPU-only optimized SSIM calculation for PyTorch tensors"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. CPU fallback is not supported.")
    
    with torch.cuda.amp.autocast():
        # Convert tensors to numpy for SSIM calculation
        img1 = tensor1.squeeze().cpu().numpy()
        img2 = tensor2.squeeze().cpu().numpy()
        
        # Transpose from CHW to HWC if needed
        if len(img1.shape) == 3 and img1.shape[0] == 3:
            img1 = np.transpose(img1, (1, 2, 0))
            img2 = np.transpose(img2, (1, 2, 0))
        
        # Convert to uint8 for SSIM
        img1 = fast_clip_operation(img1 * 255, 0, 255).astype(np.uint8)
        img2 = fast_clip_operation(img2 * 255, 0, 255).astype(np.uint8)
        
        return calculate_ssim(img1, img2)

def print_attack_info(output_path, original_image, adv_image, attack_type):
    """Print information about the attack using optimized calculations"""
    # Use optimized perturbation calculation
    perturbation = fast_perturbation_calculation(original_image, adv_image)
    print(f"Max perturbation: {float(np.max(perturbation))}")
    print(f"Mean perturbation: {float(np.mean(perturbation))}")
    
    # Use optimized SSIM calculation
    ssim_val = calculate_ssim(original_image, adv_image)
    print(f"SSIM: {float(ssim_val)}")
    
    print("\nTo use this adversarial image in evaluation:")
    print(f"1. The image is saved at: {output_path}")
    print("2. When running eval_model.py, the script will use the original path")
    print("3. To use adversarial images, modify the img_path in eval_model.py:")
    print("   Change: img_path = 'data/clean/' + data['image']")
    box_type = 'blackbox' if 'black_box' in output_path else 'whitebox'
    print(f"   To:     img_path = 'data/adversarial/{box_type}/{attack_type}/' + data['image']")


class UniversalSSIMOptimizer:
    """Unified SSIM-aware optimizer for both white-box and black-box attacks"""
    
    def __init__(self, target_ssim: float = 0.85, tolerance: float = 0.01):
        self.target_ssim = target_ssim
        self.tolerance = tolerance
        self.best_result = None
        self.query_count = 0
        
    def calculate_objective_score(self, image: np.ndarray, adv_image: np.ndarray) -> float:
        """Calculate SSIM-based objective score (lower is better)"""
        achieved_ssim = calculate_ssim(image, adv_image)
        score = abs(achieved_ssim - self.target_ssim)
        return score, achieved_ssim

def bayesian_optimization_framework(
    image_path: str,
    attack_type: str,
    attack_function: Callable,
    classifier: object,
    param_space: Dict,
    is_blackbox: bool = False,
    target_ssim: float = 0.85,
    max_trials: int = 5,
    timeout: int = 1800
) -> Tuple[np.ndarray, float, Dict]:
    """
    Universal Bayesian optimization framework for both white-box and black-box attacks
    
    Args:
        image_path: Path to input image
        attack_type: Type of attack to perform
        attack_function: Attack function to execute
        classifier: ML classifier object
        param_space: Dictionary of parameter search spaces
        is_blackbox: Whether this is a blackbox attack
        target_ssim: Target SSIM value (default: 0.85)
        max_trials: Maximum optimization trials (default: 5)
        timeout: Timeout in seconds (default: 1800 = 30 min)
    
    Returns:
        Tuple of (adversarial_image, achieved_ssim, optimal_parameters)
    """
    start_time = time.time()
    
    # Load image
    image = load_image(image_path)
    logger.info(f"📸 Loaded image: {image_path} (Shape: {image.shape})")
    
    # Initialize optimizer
    optimizer = UniversalSSIMOptimizer(target_ssim=target_ssim)
    
    # Create Optuna study
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
    
    print(f"🔍 Starting Bayesian optimization for {attack_type.upper()} attack...")
    print(f"Target SSIM: {target_ssim:.4f}, Max trials: {max_trials}")
    
    def objective(trial):
        optimizer.query_count += 1
        
        # Sample parameters from search space
        params = {}
        for param, values in param_space.items():
            if isinstance(values[0], (int, float)) and len(set(values)) > 2:
                # Check if all values are integers
                if all(isinstance(v, int) or (isinstance(v, float) and v.is_integer()) for v in values):
                    # Integer parameter
                    params[param] = trial.suggest_int(param, int(min(values)), int(max(values)))
                else:
                    # Continuous float parameter
                    params[param] = trial.suggest_float(param, min(values), max(values))
            else:
                # Categorical parameter
                params[param] = trial.suggest_categorical(param, values)
        
        try:
            # Execute attack with sampled parameters
            if is_blackbox:
                adv_image = attack_function(image, classifier, **params)
            else:
                adv_image = attack_function(image, classifier, attack_type, params)
            
            if adv_image is not None:
                # Calculate SSIM-based score
                score, achieved_ssim = optimizer.calculate_objective_score(image, adv_image)
                
                logger.info(f"Query {optimizer.query_count}: SSIM={achieved_ssim:.4f}, Score={score:.4f}")
                
                # Update best result
                if optimizer.best_result is None or score < optimizer.best_result[0]:
                    optimizer.best_result = (score, adv_image, achieved_ssim, params.copy())
                
                # Early stopping if target achieved
                if score <= optimizer.tolerance:
                    print(f"🏆 TARGET ACHIEVED! SSIM: {achieved_ssim:.4f}, Target: {target_ssim:.4f}")
                    
                    # Print perturbation info for early stopping
                    perturbation = fast_perturbation_calculation(image, adv_image)
                    print(f"Max perturbation: {float(np.max(perturbation))}")
                    print(f"Mean perturbation: {float(np.mean(perturbation))}")
                    print(f"Achieved SSIM: {achieved_ssim:.4f}")
                    
                    study.stop()
                
                return score
            else:
                return float('inf')
                
        except Exception as e:
            logger.warning(f"Attack failed with params {params}: {e}")
            return float('inf')
    
    # Run optimization
    try:
        study.optimize(objective, n_trials=max_trials, timeout=timeout)
        
        if optimizer.best_result is not None:
            score, adv_image, achieved_ssim, best_params = optimizer.best_result
            
            # Print final perturbation info
            perturbation = fast_perturbation_calculation(image, adv_image)
            print(f"Max perturbation: {float(np.max(perturbation))}")
            print(f"Mean perturbation: {float(np.mean(perturbation))}")
            print(f"Achieved SSIM: {achieved_ssim:.4f}")
            
            # Save result
            output_path = get_output_path(image_path, attack_type, is_blackbox=is_blackbox, ssim_threshold=target_ssim)
            save_image(adv_image, output_path)
            
            execution_time = time.time() - start_time
            print(f"✅ {attack_type.upper()} attack completed in {execution_time:.1f}s")
            print(f"📁 Saved adversarial image: {output_path}")
            print(f"🎯 Target SSIM: {target_ssim:.4f}")
            print(f"✨ Achieved SSIM: {achieved_ssim:.4f}")
            
            return adv_image, achieved_ssim, best_params
        else:
            raise RuntimeError("No valid adversarial example generated")
            
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise