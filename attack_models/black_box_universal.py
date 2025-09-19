#!/usr/bin/env python3
"""
Black-Box Adversarial Attacks Universal Module

Supports: Square, SimBA, Boundary, Pixel, Spatial attacks
Usage: python black_box_universal.py --attack_type square --image_path ... --ssim_threshold ...

This module merges universal.py + utils.py following ART/CleverHans standards.
Provides specialized utilities for black-box adversarial attacks with SSIM optimization.

Supported Attacks (OPTIMIZED FOR 7.6GB GPU - 5 fast attacks):
- Square Attack - 11.25s avg
- SimBA Attack - 12.75s avg
- Boundary Attack - 39.25s avg
- Pixel Attack - 42.75s avg
- Spatial Transformation Attack - 59.25s avg ⚠️  SSIM ~0.73

REMOVED SLOW ATTACKS (for GPU memory optimization):
- HopSkipJump Attack - 914.75s avg - Too computationally expensive
- ZOO Attack (Zeroth Order Optimization) - 3263.5s avg - Too computationally expensive
- GeoDA Attack - Not implemented yet
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from scipy.optimize import minimize, differential_evolution
import optuna
from typing import Tuple, Dict, Any, Optional
import logging
import time
import json
import argparse

# Import common utilities from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    load_image, create_classifier, save_image, preprocess_image_for_attack,
    postprocess_adversarial_image, get_output_path, calculate_ssim,
    fast_perturbation_calculation, fast_clip_operation, print_attack_info,
    bayesian_optimization_framework, UniversalSSIMOptimizer, logger
)
from adaptive_ssim_optimizer import get_adaptive_search_space

# ART imports for black-box attacks
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion.hop_skip_jump import HopSkipJump
from art.attacks.evasion.simba import SimBA
from art.attacks.evasion.square_attack import SquareAttack
from art.attacks.evasion.pixel_threshold import PixelAttack
from art.attacks.evasion.zoo import ZooAttack
from art.attacks.evasion.boundary import BoundaryAttack
from art.attacks.evasion.spatial_transformation import SpatialTransformation

# BLACK-BOX PARAMETER SPACES FOR BAYESIAN OPTIMIZATION (OPTIMIZED FOR 7.6GB GPU)
# NOTE: Now using adaptive optimizer for SSIM-aware parameter spaces
def get_blackbox_param_spaces(attack_type: str, ssim_target: float = 0.85,
                             trial_number: int = 1, max_trials: int = 5) -> Dict:
    """Get adaptive parameter spaces for black-box attacks"""
    return get_adaptive_search_space(attack_type, ssim_target, trial_number, max_trials)

# Legacy static spaces (kept for reference/fallback)
BLACKBOX_PARAM_SPACES_LEGACY = {
    'square': {  # Andriushchenko et al. 2020 - "Square Attack: a query-efficient black-box adversarial attack via random search"
        'eps': [0.01, 0.05, 0.1, 0.15, 0.2],  # Andriushchenko et al. 2020 - perturbation radius (paper: ε=0.05)
        'max_iter': [50, 100, 200, 500],  # Andriushchenko et al. 2020 - iteration range (paper: 10000)
        'p_init': [0.1, 0.3, 0.5, 0.8],  # Andriushchenko et al. 2020 - initial probability (paper: 0.8)
        'norm': [np.inf, 2]  # Andriushchenko et al. 2020 - L∞ and L2 norms
    },
    # REMOVED: 'hop_skip_jump' - 914.75s avg execution time - Too computationally expensive for 7.6GB GPU
    # 'hop_skip_jump': {  # Chen et al. 2019 - "HopSkipJumpAttack: A Query-Efficient Decision-Based Attack"
    #     'max_iter': [20, 50, 100],  # Chen et al. 2019 - optimization iterations (paper: 50)
    #     'max_eval': [500, 1000, 2000],  # Chen et al. 2019 - maximum evaluations (paper: 10000)
    #     'init_eval': [50, 100, 200],  # Chen et al. 2019 - initial evaluations (paper: 100)
    #     'norm': [2, np.inf]  # Chen et al. 2019 - norm constraints
    # },
    'pixel': {  # Su et al. 2019 - "One pixel attack for fooling deep neural networks"
        'th': [5, 10, 15, 20],  # Su et al. 2019 - pixel threshold range
        'es': [0, 1],  # Su et al. 2019 - evolution strategy parameter
        'max_iter': [50, 100, 200]  # Su et al. 2019 - optimization iterations
    },
    'simba': {  # Guo et al. 2019 - "Simple Black-box Adversarial Attacks"
        'epsilon': [0.05, 0.1, 0.15, 0.2],  # Guo et al. 2019 - perturbation bound (paper: ε=0.05)
        'max_iter': [500, 1000, 2000],  # Guo et al. 2019 - max iterations (paper: 10000-30000)
        'freq_dim': [16, 32, 64],  # Guo et al. 2019 - frequency dimensions (paper: 28-38)
        'order': ['random', 'diag']  # Guo et al. 2019 - attack order strategy
    },
    'spatial': {  # Engstrom et al. 2017 - "A Rotation and a Translation Suffice: Fooling CNNs with Simple Transformations"
        # NOTE: Geometric transformations are inherently incompatible with high SSIM targets (≥0.85)
        # SSIM measures structural similarity, and even small rotations/translations cause significant
        # structural changes that SSIM detects. Results typically achieve SSIM 0.6-0.8 range.
        # Consider lowering SSIM target to 0.7-0.8 for spatial attacks or use pixel-based attacks instead.
        'max_translation': [1.0, 2.0, 3.0],  # Severely restricted for SSIM 0.85 (original: [5.0, 10.0, 15.0])
        'num_translations': [5, 7, 9],  # Increased sampling for finer control - from [3, 5, 7]
        'max_rotation': [3.0, 6.0, 9.0],  # Severely restricted for SSIM 0.85 (original: [15.0, 30.0, 45.0])
        'num_rotations': [5, 7, 9]  # Increased sampling for finer control - from [3, 5, 7]
    },
    # REMOVED: 'zoo' - 3263.5s avg execution time - Too computationally expensive for 7.6GB GPU
    # 'zoo': {  # Chen et al. 2017 - "ZOO: Zeroth Order Optimization based Black-box Attacks to Deep Neural Networks"
    #     'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3],  # Optimized for SSIM 0.85 - finer control, reduced from [1e-3, 1e-2, 1e-1]
    #     'max_iter': [200, 500, 1000],  # Increased iterations for better convergence - from [50, 100, 200]
    #     'initial_const': [1e-5, 1e-4, 1e-3],  # Lower initial constants for gentler start - from [1e-4, 1e-3, 1e-2]
    #     'confidence': [0.0, 0.05, 0.1]  # Reduced confidence for less aggressive attacks - from [0.0, 0.1, 0.5]
    # },
    'boundary': {  # Brendel et al. 2017 - "Decision-Based Adversarial Attacks: Reliable Attacks Against Machine Learning"
        'max_iter': [1000, 2000, 5000],  # Brendel et al. 2017 - boundary iterations (paper: 5000)
        'delta': [0.001, 0.01, 0.1],  # Brendel et al. 2017 - step size parameter
        'epsilon': [0.001, 0.01, 0.1],  # Brendel et al. 2017 - convergence threshold
        'step_adapt': [0.9, 0.95, 0.99]  # Brendel et al. 2017 - step adaptation factor
    }
}

# Maintain backward compatibility
BLACKBOX_PARAM_SPACES = BLACKBOX_PARAM_SPACES_LEGACY

# Individual attack functions
def square_attack(image: np.ndarray, classifier: PyTorchClassifier, **params) -> np.ndarray:
    """Execute Square Attack"""
    attack = SquareAttack(
        estimator=classifier,
        norm=params.get('norm', np.inf),
        max_iter=params.get('max_iter', 100),
        eps=params.get('eps', 0.05),
        p_init=params.get('p_init', 0.8),
        nb_restarts=params.get('nb_restarts', 1),
        batch_size=1,
        verbose=False
    )

    img_tensor = preprocess_image_for_attack(image)
    adv_tensor = attack.generate(x=img_tensor)

    return postprocess_adversarial_image(adv_tensor, image.shape[:2])

def hop_skip_jump_attack(image: np.ndarray, classifier: PyTorchClassifier, **params) -> np.ndarray:
    """REMOVED: HopSkipJump Attack - 914.75s avg execution time - Too computationally expensive for 7.6GB GPU"""
    raise NotImplementedError("HopSkipJump attack removed for GPU optimization (914.75s avg execution time)")

def pixel_attack(image: np.ndarray, classifier: PyTorchClassifier, **params) -> np.ndarray:
    """Execute Pixel Attack"""
    attack = PixelAttack(
        classifier=classifier,
        th=params.get('th', 10),
        es=params.get('es', 1),
        max_iter=params.get('max_iter', 100),
        targeted=params.get('targeted', False),
        verbose=False
    )

    img_tensor = preprocess_image_for_attack(image)
    adv_tensor = attack.generate(x=img_tensor)

    return postprocess_adversarial_image(adv_tensor, image.shape[:2])

def simba_attack(image: np.ndarray, classifier: PyTorchClassifier, **params) -> np.ndarray:
    """Execute SimBA Attack"""
    # SimBA requires probabilistic classifier - create new one if needed
    prob_classifier = create_classifier(device='cuda:0', requires_grad=False, probabilistic=True)

    attack = SimBA(
        classifier=prob_classifier,
        attack=params.get('attack_method', 'dct'),
        max_iter=params.get('max_iter', 100),  # Reduced for testing
        epsilon=params.get('epsilon', 0.1),
        freq_dim=params.get('freq_dim', 32),
        stride=params.get('stride', 1),
        order=params.get('order', 'diag'),
        targeted=params.get('targeted', False),
        verbose=False
    )

    img_tensor = preprocess_image_for_attack(image)
    adv_tensor = attack.generate(x=img_tensor)

    return postprocess_adversarial_image(adv_tensor, image.shape[:2])

def spatial_attack(image: np.ndarray, classifier: PyTorchClassifier, **params) -> np.ndarray:
    """
    Execute Spatial Transformation Attack

    WARNING: Geometric transformations (rotation + translation) are fundamentally incompatible
    with high SSIM targets (≥0.85). SSIM measures structural similarity, and geometric
    transformations inherently alter image structure more than pixel-level perturbations.

    Expected SSIM range: 0.6-0.8 (even with severely restricted parameters)
    Recommendation: Use SSIM target ≤0.8 or choose pixel-based attacks for SSIM ≥0.85
    """
    attack = SpatialTransformation(
        classifier=classifier,
        max_translation=params.get('max_translation', 10.0),
        num_translations=params.get('num_translations', 5),
        max_rotation=params.get('max_rotation', 30.0),
        num_rotations=params.get('num_rotations', 5),
        verbose=False
    )

    img_tensor = preprocess_image_for_attack(image)
    adv_tensor = attack.generate(x=img_tensor)

    return postprocess_adversarial_image(adv_tensor, image.shape[:2])

def zoo_attack(image: np.ndarray, classifier: PyTorchClassifier, **params) -> np.ndarray:
    """REMOVED: ZOO Attack - 3263.5s avg execution time - Too computationally expensive for 7.6GB GPU"""
    raise NotImplementedError("ZOO attack removed for GPU optimization (3263.5s avg execution time)")

def boundary_attack(image: np.ndarray, classifier: PyTorchClassifier, **params) -> np.ndarray:
    """Execute Boundary Attack"""
    attack = BoundaryAttack(
        estimator=classifier,
        batch_size=1,
        targeted=params.get('targeted', False),  # Set to False to avoid needing target labels
        delta=params.get('delta', 0.01),
        epsilon=params.get('epsilon', 0.01),
        step_adapt=params.get('step_adapt', 0.90),
        max_iter=params.get('max_iter', 100),  # Reduced for testing
        num_trial=params.get('num_trial', 5),  # Reduced for testing
        sample_size=params.get('sample_size', 20),
        init_size=params.get('init_size', 100),
        verbose=False
    )

    img_tensor = preprocess_image_for_attack(image)
    adv_tensor = attack.generate(x=img_tensor)

    return postprocess_adversarial_image(adv_tensor, image.shape[:2])

# Attack function mapping (OPTIMIZED FOR 7.6GB GPU - 5 fast attacks)
# REMOVED SLOW ATTACKS: hop_skip_jump (914.75s), zoo (3263.5s) - Too computationally expensive
ATTACK_FUNCTIONS = {
    'square': square_attack,          # 11.25s avg - Fast score-based attack
    'simba': simba_attack,            # 12.75s avg - Simple black-box attack
    'boundary': boundary_attack,      # 39.25s avg - Decision boundary attack
    'pixel': pixel_attack,            # 42.75s avg - Few-pixel modification attack
    'spatial': spatial_attack,        # 59.25s avg - Spatial transformation (SSIM ~0.73)

    # REMOVED: Too computationally expensive for 7.6GB GPU
    # 'hop_skip_jump': hop_skip_jump_attack,  # 914.75s avg - Too slow
    # 'zoo': zoo_attack,                      # 3263.5s avg - Too slow
}

def ssim_aware_blackbox_attack(
    image_path: str,
    attack_type: str,
    target_ssim: float = 0.85,
    optimization_method: str = "bayesian",
    max_trials: int = 5,
    trial_number: int = 1
) -> Tuple[np.ndarray, float, Dict]:
    """
    Main function for SSIM-aware black-box adversarial attacks

    Args:
        image_path: Path to input image
        attack_type: Type of attack to perform
        target_ssim: Target SSIM value (default: 0.85)
        optimization_method: Optimization method (default: "bayesian")
        max_trials: Maximum optimization trials (default: 5)

    Returns:
        Tuple of (adversarial_image, achieved_ssim, optimal_parameters)
    """
    # Create classifier - GPU ONLY, no gradients needed for black-box
    classifier = create_classifier(device='cuda:0', requires_grad=False)

    if optimization_method == "bayesian":
        # Use adaptive search space for improved optimization
        if optimization_method == "adaptive":
            param_space = get_blackbox_param_spaces(attack_type, target_ssim, trial_number, max_trials)
        else:
            # Fallback to legacy spaces for backward compatibility
            if attack_type not in BLACKBOX_PARAM_SPACES:
                raise ValueError(f"Attack type '{attack_type}' not supported for blackbox")
            param_space = BLACKBOX_PARAM_SPACES[attack_type]
        attack_func = ATTACK_FUNCTIONS[attack_type]

        # Use centralized Bayesian optimization
        return bayesian_optimization_framework(
            image_path=image_path,
            attack_type=attack_type,
            attack_function=attack_func,
            classifier=classifier,
            param_space=param_space,
            is_blackbox=True,
            target_ssim=target_ssim,
            max_trials=max_trials,
            timeout=1800
        )
    else:
        raise NotImplementedError(f"Optimization method '{optimization_method}' not implemented")

def main():
    parser = argparse.ArgumentParser(description="Universal blackbox attack with automated hyperparameter optimization")

    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--attack_type", type=str, required=True,
                        choices=["square", "pixel", "simba", "spatial", "boundary"],
                        help="Type of blackbox attack to perform (NOTE: spatial attack incompatible with SSIM ≥0.85)")
    parser.add_argument("--ssim_threshold", type=float, default=0.85,
                        help="Target SSIM value (default: 0.85)")
    parser.add_argument("--optimization_method", type=str, default="bayesian",
                        choices=["bayesian", "grid_search", "random_search", "adaptive"],
                        help="Hyperparameter optimization method (default: bayesian)")
    parser.add_argument("--max_trials", type=int, default=5,
                        help="Maximum optimization trials (default: 5)")
    parser.add_argument("--trial_number", type=int, default=1,
                        help="Current trial number for adaptive optimization (default: 1)")

    args = parser.parse_args()

    # Check if CUDA is available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Attack type: {args.attack_type.upper()}")
    print(f"Target SSIM: {args.ssim_threshold}")
    print(f"Optimization method: {args.optimization_method}")
    print(f"Max trials: {args.max_trials}")

    # Warning for spatial attack with high SSIM targets
    if args.attack_type == "spatial" and args.ssim_threshold >= 0.85:
        print("\n⚠️  WARNING: Spatial (geometric) transformations are incompatible with SSIM ≥0.85")
        print("   Expected SSIM range: 0.6-0.8 due to structural changes from rotation/translation")
        print("   Consider: (1) Lower SSIM target to ≤0.8, or (2) Use pixel-based attacks instead\n")

    try:
        # Universal attack execution - single line handles ALL blackbox attacks
        adv_image, achieved_ssim, final_params = ssim_aware_blackbox_attack(
            image_path=args.image_path,
            attack_type=args.attack_type,  # This single parameter determines the attack
            target_ssim=args.ssim_threshold,
            optimization_method=args.optimization_method,
            max_trials=args.max_trials,
            trial_number=args.trial_number
        )

        print(f"Target SSIM: {args.ssim_threshold:.4f}")
        print(f"Achieved SSIM: {achieved_ssim:.4f}")
        print(f"Optimal parameters: {final_params}")
        print(f"Attack: {args.attack_type.upper()} completed successfully")

    except Exception as e:
        print(f"{args.attack_type.upper()} attack failed: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())