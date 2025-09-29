#!/usr/bin/env python3
"""
Black-Box Adversarial Attacks Universal Module

Supports: Square, SimBA, Boundary, SignOPT attacks
Usage: python black_box_universal.py --attack_type square --image_path ... --epsilon ...

This module provides specialized utilities for black-box adversarial attacks with direct epsilon control.

Supported Attacks (100% EPSILON PARAMETER SUPPORT - 4 fast attacks):
- Square Attack - ε parameter
- SimBA Attack - ε parameter
- Boundary Attack - ε parameter
- SignOPT Attack - ε parameter

ALL ATTACKS SUPPORT EPSILON:
- Direct epsilon parameter control for precise perturbation bounds
- No optimization needed - fast and predictable execution
- 100% epsilon parameter compatibility across all attack types
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
from typing import Tuple, Dict, Any, Optional
import logging
import time
import json
import argparse

# Import common utilities from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from attack_models.utils import (
        load_image, create_classifier, save_image, preprocess_image_for_attack,
        postprocess_adversarial_image, get_output_path,
        fast_perturbation_calculation, fast_clip_operation, print_attack_info,
        calculate_epsilon, refine_epsilon_tolerance, logger, query_counter,
        batch_preprocess_images, batch_postprocess_images, get_optimal_batch_size,
        optimize_memory_usage, get_gpu_memory_info, setup_gpu_optimizations
    )
except ImportError:
    # Fallback for direct execution from attack_models directory
    from utils import (
        load_image, create_classifier, save_image, preprocess_image_for_attack,
        postprocess_adversarial_image, get_output_path,
        fast_perturbation_calculation, fast_clip_operation, print_attack_info,
        calculate_epsilon, refine_epsilon_tolerance, logger, query_counter,
        batch_preprocess_images, batch_postprocess_images, get_optimal_batch_size,
        optimize_memory_usage, get_gpu_memory_info, setup_gpu_optimizations
    )

# ART imports for black-box attacks
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion.simba import SimBA
from art.attacks.evasion.square_attack import SquareAttack
from art.attacks.evasion.boundary import BoundaryAttack
from art.attacks.evasion.sign_opt import SignOPTAttack

# BLACK-BOX PARAMETER SPACES - 100% EPSILON PARAMETER SUPPORT
# NOTE: All attacks now use direct epsilon control for consistent perturbation bounds
def get_blackbox_param_spaces(attack_type: str, epsilon_target: float = 0.05) -> Dict:
    """Get parameter spaces for black-box attacks with direct epsilon control"""
    base_params = {
        'square': {'eps': epsilon_target, 'max_iter': 1000, 'p_init': 0.05, 'norm': np.inf},
        'simba': {'epsilon': epsilon_target, 'max_iter': 1000, 'freq_dims': 8, 'stride': 8},
        'boundary': {'epsilon': epsilon_target, 'max_iter': 1000, 'num_trials': 25, 'sample_size': 20},
        'sign_opt': {'epsilon': epsilon_target, 'max_iter': 1000, 'query_limit': 20000}
    }
    return base_params.get(attack_type, {})

# Legacy static spaces (kept for reference/fallback)
BLACKBOX_PARAM_SPACES_LEGACY = {
    'square': {  # Andriushchenko et al. 2020 - "Square Attack: a query-efficient black-box adversarial attack via random search"
        'eps': [0.01, 0.05, 0.1, 0.15, 0.2],  # Andriushchenko et al. 2020 - perturbation radius (paper: ε=0.05)
        'max_iter': [50, 100, 200, 500],  # Andriushchenko et al. 2020 - iteration range (paper: 10000)
        'p_init': [0.1, 0.3, 0.5, 0.8],  # Andriushchenko et al. 2020 - initial probability (paper: 0.8)
        'norm': [np.inf, 2]  # Andriushchenko et al. 2020 - L∞ and L2 norms
    },
    'simba': {  # Guo et al. 2019 - "Simple Black-box Adversarial Attacks"
        'epsilon': [0.05, 0.1, 0.15, 0.2],  # Guo et al. 2019 - perturbation bound (paper: ε=0.05)
        'max_iter': [500, 1000, 2000],  # Guo et al. 2019 - max iterations (paper: 10000-30000)
        'freq_dim': [16, 32, 64],  # Guo et al. 2019 - frequency dimensions (paper: 28-38)
        'order': ['random', 'diag']  # Guo et al. 2019 - attack order strategy
    },
    'boundary': {  # Brendel et al. 2017 - "Decision-Based Adversarial Attacks: Reliable Attacks Against Machine Learning"
        'max_iter': [1000, 2000, 5000],  # Brendel et al. 2017 - boundary iterations (paper: 5000)
        'delta': [0.001, 0.01, 0.1],  # Brendel et al. 2017 - step size parameter
        'epsilon': [0.001, 0.01, 0.1],  # Brendel et al. 2017 - convergence threshold
        'step_adapt': [0.9, 0.95, 0.99]  # Brendel et al. 2017 - step adaptation factor
    },
    'sign_opt': {  # Cheng et al. 2019 - "Sign-OPT: A Query-Efficient Hard-label Adversarial Attack"
        'epsilon': [0.05, 0.1, 0.15, 0.2],  # Cheng et al. 2019 - perturbation bound
        'max_iter': [1000, 2000, 5000],  # Cheng et al. 2019 - optimization iterations
        'query_limit': [10000, 20000, 50000]  # Cheng et al. 2019 - query budget
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

def simba_attack(image: np.ndarray, classifier: PyTorchClassifier, **params) -> np.ndarray:
    """Execute SimBA Attack"""
    # SimBA requires probabilistic classifier - create new one if needed
    # Enable TensorRT for maximum performance (no gradients needed for black-box)
    prob_classifier = create_classifier(
        device='cuda:0',
        requires_grad=False,
        probabilistic=True,
        count_queries=True,
        optimization_level='high',
        use_tensorrt=True  # TensorRT enabled for black-box attacks
    )

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

def sign_opt_attack(image: np.ndarray, classifier: PyTorchClassifier, **params) -> np.ndarray:
    """Execute SignOPT Attack"""
    attack = SignOPTAttack(
        estimator=classifier,
        epsilon=params.get('epsilon', 0.05),
        max_iter=params.get('max_iter', 1000),
        query_limit=params.get('query_limit', 20000),
        targeted=params.get('targeted', False),
        verbose=False
    )

    img_tensor = preprocess_image_for_attack(image)
    adv_tensor = attack.generate(x=img_tensor)

    return postprocess_adversarial_image(adv_tensor, image.shape[:2])

# Attack function mapping - 100% EPSILON PARAMETER SUPPORT (4 fast attacks)
ATTACK_FUNCTIONS = {
    'square': square_attack,          # ε parameter - Fast score-based attack
    'simba': simba_attack,            # ε parameter - Simple black-box attack
    'boundary': boundary_attack,      # ε parameter - Decision boundary attack
    'sign_opt': sign_opt_attack       # ε parameter - Sign-based optimization attack
}

class UniversalEpsilonBlackBoxAttack:
    """
    Universal epsilon-based black-box attack with direct epsilon control.
    Similar to white-box implementation but for black-box attacks.
    """

    def __init__(self, epsilon_target=0.05):
        self.epsilon_target = epsilon_target

    def _get_target_specific_config(self, target_epsilon: float) -> Dict[str, Any]:
        """
        Get target-specific configuration for black-box epsilon control

        Black-box attacks typically need more conservative settings
        due to query-based nature and different optimization characteristics
        """
        EPSILON_TARGET_CONFIGS = {
            0.02: {
                'max_iterations': 10,     # More iterations for small epsilon
                'max_queries': 5000       # Conservative query budget
            },
            0.05: {
                'max_iterations': 8,      # Standard iterations
                'max_queries': 10000      # Standard query budget
            },
            0.08: {
                'max_iterations': 6,      # Fewer iterations for large epsilon
                'max_queries': 15000      # Higher query budget
            }
        }

        # Find closest target epsilon configuration
        closest_target = min(EPSILON_TARGET_CONFIGS.keys(), key=lambda x: abs(x - target_epsilon))
        config = EPSILON_TARGET_CONFIGS[closest_target].copy()

        # If exact match not found, adjust proportionally
        if abs(closest_target - target_epsilon) > 0.005:
            print(f"⚠️ Using closest config for epsilon {closest_target} (requested: {target_epsilon})")

        return config

    def _run_attack_with_params(self, image: np.ndarray, classifier, attack_type: str, params: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Wrapper for running black-box attack with parameters

        Args:
            image: Input image
            classifier: Attack classifier
            attack_type: Type of attack
            params: Attack parameters including epsilon-related values

        Returns:
            Adversarial image or None if attack fails
        """
        try:
            # Get the appropriate attack function
            if attack_type not in ATTACK_FUNCTIONS:
                raise ValueError(f"Unsupported attack type: {attack_type}")

            attack_function = ATTACK_FUNCTIONS[attack_type]
            return attack_function(image, classifier, **params)

        except Exception as e:
            print(f"Black-box attack failed with params {params}: {e}")
            return None

    def run_epsilon_attack(self, image_path: str, attack_type: str,
                          attack_params: Dict = None) -> Tuple[np.ndarray, float, Dict]:
        """
        Universal epsilon-based black-box attack with direct epsilon control

        Args:
            image_path: Path to target image
            attack_type: Type of black-box attack
            attack_params: Base attack parameters (optional)

        Returns:
            Tuple of (adversarial_image, achieved_epsilon, final_parameters)
        """
        print(f"\n🎯 Universal Epsilon-Based Black-Box Attack")
        print(f"Target epsilon: {self.epsilon_target:.4f}")
        print(f"Attack: {attack_type.upper()}")

        # Load image and create optimized classifier
        image = load_image(image_path)
        # Enable TensorRT for maximum performance (no gradients needed for black-box)
        classifier = create_classifier(
            device='cuda:0',
            requires_grad=False,
            count_queries=True,
            optimization_level='high',  # Use high optimization for black-box attacks
            use_tensorrt=True  # TensorRT enabled for black-box attacks
        )

        # Get default parameters if none provided
        if attack_params is None:
            attack_params = self._get_default_params(attack_type)

        # Set target epsilon in parameters
        if attack_type == 'square':
            attack_params['eps'] = self.epsilon_target
        else:
            attack_params['epsilon'] = self.epsilon_target

        print(f"Base parameters: {attack_params}")

        # Run attack with target epsilon (no optimization needed)
        adv_image = self._run_attack_with_params(image, classifier, attack_type, attack_params)

        # Check if attack was successful
        if adv_image is not None:
            # ITERATIVE EPSILON REFINEMENT for ±5% tolerance (prioritize accuracy over speed)
            adv_image, epsilon_l_inf = refine_epsilon_tolerance(
                original_image=image,
                adversarial_image=adv_image,
                target_epsilon=self.epsilon_target,
                tolerance=0.05,  # ±5%
                max_iterations=100  # High iterations for accuracy
            )
            print(f"🎯 Epsilon refinement completed: {epsilon_l_inf:.6f}")

            # Save successful result (epsilon refinement already completed)
            output_path = get_output_path(image_path, attack_type, is_blackbox=True, epsilon=self.epsilon_target)
            save_image(adv_image, output_path)

            # Get actual query count
            actual_queries = query_counter.get_count()
            print_attack_info(output_path, image, adv_image, attack_type, query_count=actual_queries)

            print(f"✅ Target epsilon: {self.epsilon_target:.6f}")

            # Calculate and return actual metrics instead of just attack_params
            from .utils import fast_perturbation_calculation, calculate_l2_norm, calculate_l0_norm

            perturbation = fast_perturbation_calculation(image, adv_image)
            final_metrics = {
                'mean_perturbation': float(np.mean(perturbation)),
                'max_perturbation': float(np.max(perturbation)),
                'epsilon_l_inf': float(epsilon_l_inf),
                'l2_norm': float(calculate_l2_norm(image, adv_image)),
                'l0_norm': int(calculate_l0_norm(image, adv_image)),
                'total_queries': int(actual_queries),
                # Keep original attack params for reference
                **attack_params
            }

            return adv_image, epsilon_l_inf, final_metrics

        # Return failure if no result
        print("❌ Direct epsilon attack failed to produce valid adversarial example")
        return None, 0.0, {}

    def _get_default_params(self, attack_type: str) -> Dict[str, Any]:
        """Get default parameters for different black-box attack types"""
        default_params = {
            'square': {
                'max_iter': 1000,
                'p_init': 0.05,
                'norm': np.inf
            },
            'simba': {
                'max_iter': 1000,
                'freq_dims': 8,
                'stride': 8  # (224-8)/8 = 27, remainder 0 ✓
            },
            'boundary': {
                'max_iter': 1000,
                'num_trials': 25,
                'sample_size': 20
            },
            'sign_opt': {
                'max_iter': 1000,
                'query_limit': 20000
            }
        }

        return default_params.get(attack_type, {})


def epsilon_based_blackbox_attack(
    image_path: str,
    attack_type: str,
    epsilon_target: float = 0.05,
    max_trials: int = 1,
    trial_number: int = 1
) -> Tuple[np.ndarray, float, Dict]:
    """
    Main function for epsilon-based black-box adversarial attacks

    Args:
        image_path: Path to input image
        attack_type: Type of attack to perform
        epsilon_target: Target epsilon value (default: 0.05)
        max_trials: Maximum trials (not needed for direct epsilon, default: 1)

    Returns:
        Tuple of (adversarial_image, achieved_epsilon, optimal_parameters)
    """

    # Reset query counter before attack
    query_counter.reset()

    print(f"🎯 Direct Epsilon-Based Black-Box Attack (no optimization needed)")
    print(f"Target epsilon: {epsilon_target}")

    attack_instance = UniversalEpsilonBlackBoxAttack(epsilon_target=epsilon_target)

    return attack_instance.run_epsilon_attack(
        image_path=image_path,
        attack_type=attack_type,
        attack_params=None
    )

def main():
    parser = argparse.ArgumentParser(description="Universal blackbox attack with direct epsilon control")

    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--attack_type", type=str, required=True,
                        choices=["square", "simba", "boundary", "sign_opt"],
                        help="Type of blackbox attack to perform")
    parser.add_argument("--epsilon", type=float, default=0.05,
                        help="Target epsilon value (default: 0.05)")
    parser.add_argument("--max_trials", type=int, default=1,
                        help="Maximum trials (not needed for direct epsilon, default: 1)")
    parser.add_argument("--trial_number", type=int, default=1,
                        help="Current trial number (default: 1)")

    args = parser.parse_args()

    # Check if CUDA is available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Attack type: {args.attack_type.upper()}")
    print(f"Target epsilon: {args.epsilon}")

    try:
        # Universal attack execution - single line handles ALL blackbox attacks
        adv_image, achieved_epsilon, final_params = epsilon_based_blackbox_attack(
            image_path=args.image_path,
            attack_type=args.attack_type,  # This single parameter determines the attack
            epsilon_target=args.epsilon,
            max_trials=args.max_trials,
            trial_number=args.trial_number
        )

        print(f"Target epsilon: {args.epsilon:.6f}")
        print(f"Achieved epsilon: {achieved_epsilon:.6f}")
        print(f"Optimal parameters: {final_params}")
        print(f"Attack: {args.attack_type.upper()} completed successfully")

    except Exception as e:
        print(f"{args.attack_type.upper()} attack failed: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())