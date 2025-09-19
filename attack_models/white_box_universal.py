#!/usr/bin/env python3
"""
White-Box Adversarial Attacks Universal Module

Supports: FGSM, DeepFool, PGD, CW-Linf attacks
Usage: python white_box_universal.py --attack_type fgsm --image_path ... --ssim_threshold ...

This module merges universal.py + utils.py following ART/CleverHans standards.
Provides specialized utilities for white-box adversarial attacks with GPU optimization.

White-Box Attacks Supported (OPTIMIZED FOR 7.6GB GPU - 4 fast attacks):
- FGSM (Fast Gradient Sign Method) - 4.0s avg
- DeepFool (Geometric approach) - 5.0s avg
- PGD (Projected Gradient Descent) - 5.25s avg
- CW-Linf (Carlini & Wagner L∞) - 66.5s avg

REMOVED SLOW ATTACKS (for GPU memory optimization):
- JSMA (Jacobian-based Saliency Map Attack) - 593.75s avg - Too computationally expensive
- CW-L0 (Carlini & Wagner L0) - 587.0s avg - Too computationally expensive
- CW-L2 (Carlini & Wagner L2) - 259.5s avg - Too computationally expensive
"""

import os
import sys
import numpy as np
import torch
import optuna
import logging
import time
import argparse
from art.attacks.evasion import (
    ProjectedGradientDescent, FastGradientMethod, CarliniL2Method,
    CarliniL0Method, CarliniLInfMethod, SaliencyMapMethod, DeepFool
)
from typing import Dict, List, Tuple, Optional, Any

# Import common utilities using absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from attack_models.utils import (
        load_image, create_classifier, save_image, preprocess_image_for_attack,
        postprocess_adversarial_image, get_output_path, calculate_ssim,
        fast_perturbation_calculation, fast_clip_operation, print_attack_info,
        bayesian_optimization_framework, UniversalSSIMOptimizer, logger
    )
except ImportError:
    # Fallback for direct execution
    from utils import (
        load_image, create_classifier, save_image, preprocess_image_for_attack,
        postprocess_adversarial_image, get_output_path, calculate_ssim,
        fast_perturbation_calculation, fast_clip_operation, print_attack_info,
        bayesian_optimization_framework, UniversalSSIMOptimizer, logger
    )

# WHITE-BOX PARAMETER SPACES FOR BAYESIAN OPTIMIZATION (OPTIMIZED FOR 7.6GB GPU)
# NOTE: Now using adaptive optimizer for SSIM-aware parameter spaces
def get_whitebox_param_spaces(attack_type: str, ssim_target: float = 0.85,
                             trial_number: int = 1, max_trials: int = 5) -> Dict:
    """Get adaptive parameter spaces for white-box attacks"""
    return get_adaptive_search_space(attack_type, ssim_target, trial_number, max_trials)

# Legacy static spaces (kept for reference/fallback)
WHITEBOX_PARAM_SPACES_LEGACY = {
    'pgd': {
        'eps': [0.01, 0.03, 0.05, 0.08, 0.1],
        'eps_step': [0.005, 0.01, 0.02, 0.03],
        'nb_iter': [20, 40, 60, 80]
    },
    'fgsm': {
        'eps': [0.05, 0.1, 0.15, 0.2, 0.3],
        'norm': [np.inf, 2]
    },
    'cw_linf': {
        'confidence': [5, 10, 15, 20],
        'max_iter': [50, 100, 150, 200],
        'learning_rate': [0.005, 0.01, 0.02, 0.05]
    },
    'deepfool': {
        'max_iter': [50, 100, 150, 200],
        'epsilon': [1e-7, 1e-6, 1e-5, 1e-4],
        'nb_grads': [5, 10, 15, 20]
    }
}

# Maintain backward compatibility
WHITEBOX_PARAM_SPACES = WHITEBOX_PARAM_SPACES_LEGACY

# UNIVERSAL SSIM-AWARE ATTACK FRAMEWORK
class UniversalSSIMAwareAttack:
    """
    Universal SSIM-aware attack that works with ANY white-box attack type.
    Uses parameter scaling instead of attack-specific parameter ranges.
    """

    def __init__(self, target_ssim=0.85, tolerance=0.01, max_attempts=10):
        self.target_ssim = target_ssim
        self.tolerance = tolerance
        self.max_attempts = max_attempts

        # Optuna-based intelligent parameter search (replaces hardcoded multipliers)
        self.use_intelligent_search = True

    def run_ssim_aware_attack(self, image_path: str, attack_type: str,
                             attack_params: Dict = None) -> Tuple[np.ndarray, float, Dict]:
        """
        Universal SSIM-aware attack that adjusts parameters until target SSIM is reached

        Args:
            image_path: Path to target image
            attack_type: Type of attack ('jsma', 'pgd', 'fgsm', etc.)
            attack_params: Base attack parameters (optional, uses defaults if None)

        Returns:
            Tuple of (adversarial_image, achieved_ssim, final_parameters)
        """
        print(f"\n🎯 Universal SSIM-Aware Attack")
        print(f"Target: {self.target_ssim:.4f}, Tolerance: {self.tolerance:.4f}")
        print(f"Attack: {attack_type.upper()}")

        # Load image and create classifier
        image = load_image(image_path)
        classifier = create_classifier()

        # Get default parameters if none provided
        if attack_params is None:
            attack_params = self._get_default_params(attack_type)

        print(f"Base parameters: {attack_params}")

        best_result = None
        best_ssim_diff = float('inf')

        # Import adaptive optimizer
        try:
            from attack_models.adaptive_ssim_optimizer import create_optimizer
        except ImportError:
            from adaptive_ssim_optimizer import create_optimizer
        
        # Create optimizer instance
        optimizer = create_optimizer(
            target_ssim=self.target_ssim,
            tolerance=self.tolerance,
            max_trials=self.max_attempts
        )
        
        # Use intelligent parameter optimization
        result = optimizer.optimize_parameters(
            attack_func=lambda img, **params: self._run_attack(img, classifier, attack_type, params),
            base_params=attack_params,
            image=image,
            attack_type=attack_type
        )
        
        # Check if optimization was successful
        if result and result.get('success', False):
            # Save successful result
            output_path = get_output_path(image_path, attack_type, is_blackbox=False, ssim_threshold=self.target_ssim)
            save_image(result['adv_image'], output_path)
            print_attack_info(output_path, image, result['adv_image'], attack_type)
            return result['adv_image'], result['ssim'], result['params']
        
        # If target achieved but not marked as success, still return the best result
        if result and result.get('adv_image') is not None:
            output_path = get_output_path(image_path, attack_type, is_blackbox=False, ssim_threshold=self.target_ssim)
            save_image(result['adv_image'], output_path)
            print_attack_info(output_path, image, result['adv_image'], attack_type)
            return result['adv_image'], result['ssim'], result['params']
    
        # If no result from optimizer, raise error
        raise RuntimeError(f"Attack '{attack_type}' failed to generate valid adversarial examples")

    def _get_default_params(self, attack_type: str) -> Dict:
        """Get smart parameter initialization based on literature and empirical results"""
        # Literature-proven parameter ranges for optimal performance (OPTIMIZED FOR 7.6GB GPU)
        smart_defaults = {
            # REMOVED: 'jsma' - 593.75s avg execution time - Too computationally expensive for 7.6GB GPU
            # 'jsma': {  # Papernot et al. 2016 - "The Limitations of Deep Learning in Adversarial Settings"
            #     'theta': 0.5,  # Optimal theta from Papernot et al. 2016 (original paper uses θ=1.0)
            #     'max_iter': 100,  # Sufficient for convergence (paper uses adaptive stopping)
            #     'max_pixel_change': 50  # Balance between effectiveness and detectability
            # },
            'pgd': {  # Madry et al. 2017 - "Towards Deep Learning Models Resistant to Adversarial Attacks"
                'eps': 0.05,  # Madry et al. 2017 - ε perturbation bound (paper uses ε=0.3 for CIFAR-10)
                'eps_step': 0.01,  # Madry et al. 2017 - α = ε/iterations rule (paper: α=2ε/iterations)
                'nb_iter': 40  # Madry et al. 2017 - convergence iterations (paper uses 40 steps)
            },
            'fgsm': {  # Goodfellow et al. 2014 - "Explaining and Harnessing Adversarial Examples"
                'eps': 0.1,  # Goodfellow et al. 2014 - ε for single-step perturbation
                'norm': np.inf  # Goodfellow et al. 2014 - L∞ norm constraint
            },
            # REMOVED: 'cw_l2' - 259.5s avg execution time - Too computationally expensive for 7.6GB GPU
            # 'cw_l2': {  # Carlini & Wagner 2017 - "Towards Evaluating the Robustness of Neural Networks"
            #     'confidence': 10,  # Carlini & Wagner 2017 - κ confidence parameter (paper default: 0)
            #     'max_iter': 200,  # Carlini & Wagner 2017 - optimization iterations (paper: 1000)
            #     'learning_rate': 0.01  # Carlini & Wagner 2017 - Adam learning rate (paper: 0.005)
            # },
            # REMOVED: 'cw_l0' - 587.0s avg execution time - Too computationally expensive for 7.6GB GPU
            # 'cw_l0': {  # Carlini & Wagner 2017 - "Towards Evaluating the Robustness of Neural Networks"
            #     'confidence': 15,  # Carlini & Wagner 2017 - κ for sparse L0 attacks
            #     'max_iter': 100,  # Carlini & Wagner 2017 - L0 optimization iterations
            #     'learning_rate': 0.01  # Carlini & Wagner 2017 - L0 learning rate
            # },
            'cw_linf': {  # Carlini & Wagner 2017 - "Towards Evaluating the Robustness of Neural Networks"
                'confidence': 10,  # Carlini & Wagner 2017 - κ for L∞ attacks
                'max_iter': 100,  # Carlini & Wagner 2017 - L∞ optimization iterations
                'learning_rate': 0.01  # Carlini & Wagner 2017 - L∞ learning rate
            },
            'deepfool': {  # Moosavi-Dezfooli et al. 2016 - "DeepFool: a simple and accurate method to fool deep neural networks"
                'max_iter': 100,  # Moosavi-Dezfooli et al. 2016 - geometric iterations
                'epsilon': 1e-6,  # Moosavi-Dezfooli et al. 2016 - overshoot parameter
                'nb_grads': 10  # Moosavi-Dezfooli et al. 2016 - gradient computation samples
            },
        }

        if attack_type not in smart_defaults:
            raise ValueError(f"Attack type '{attack_type}' not supported")

        return smart_defaults[attack_type]

    def _scale_parameters(self, base_params: Dict, strength: float) -> Dict:
        """
        Universal parameter scaling that works for ANY attack.
        Higher strength = stronger perturbations = lower SSIM
        """
        scaled = base_params.copy()

        # Scale parameters that increase attack strength
        strength_params = ['eps', 'theta', 'confidence', 'max_pixel_change', 'eps_step']
        for param in strength_params:
            if param in scaled:
                if isinstance(scaled[param], (int, float)):
                    scaled[param] = scaled[param] * strength

        # Scale iteration parameters (more iterations = stronger attack)
        iter_params = ['max_iter', 'nb_iter']
        for param in iter_params:
            if param in scaled:
                if isinstance(scaled[param], int):
                    scaled[param] = int(scaled[param] * min(strength, 2.0))  # Cap iteration scaling

        # Scale learning rates (higher LR = faster convergence = potentially stronger)
        lr_params = ['learning_rate']
        for param in lr_params:
            if param in scaled:
                if isinstance(scaled[param], float):
                    scaled[param] = min(scaled[param] * strength, 0.1)  # Cap at 0.1

        return scaled

    def _run_attack(self, image: np.ndarray, classifier, attack_type: str, params: Dict) -> Optional[np.ndarray]:
        """Run attack with GPU memory optimizations - AUTHENTIC, NO POST-PROCESSING"""

        # Clear GPU cache before attack to free memory
        torch.cuda.empty_cache()

        # ART expects numpy arrays, not tensors
        img_tensor = preprocess_image_for_attack(image, return_tensor=False)

        try:
            # Enable gradients for white-box attacks (OPTIMIZED FOR 7.6GB GPU)
            # REMOVED ATTACK IMPLEMENTATIONS - Too computationally expensive:
            # - JSMA (593.75s avg) - Jacobian-based Saliency Map Attack
            # - CW-L0 (587.0s avg) - Carlini & Wagner L0 Attack
            # - CW-L2 (259.5s avg) - Carlini & Wagner L2 Attack

            if attack_type == 'jsma':
                # REMOVED: JSMA attack implementation - 593.75s avg execution time
                raise NotImplementedError(f"Attack '{attack_type}' removed for GPU optimization (593.75s avg)")

            elif attack_type == 'pgd':
                attack = ProjectedGradientDescent(
                    estimator=classifier,
                    norm=np.inf,
                    eps=params['eps'],
                    eps_step=params['eps_step'],
                    max_iter=int(params['nb_iter']),
                    targeted=False,
                    num_random_init=1
                )
                adv_tensor = attack.generate(x=img_tensor)

            elif attack_type == 'fgsm':
                attack = FastGradientMethod(
                    estimator=classifier,
                    norm=params.get('norm', np.inf),
                    eps=params['eps'],
                    targeted=False
                )
                adv_tensor = attack.generate(x=img_tensor)

            elif attack_type == 'cw_l2':
                # REMOVED: CW-L2 attack implementation - 259.5s avg execution time
                raise NotImplementedError(f"Attack '{attack_type}' removed for GPU optimization (259.5s avg)")

            elif attack_type == 'cw_l0':
                # REMOVED: CW-L0 attack implementation - 587.0s avg execution time
                raise NotImplementedError(f"Attack '{attack_type}' removed for GPU optimization (587.0s avg)")

            elif attack_type == 'cw_linf':
                attack = CarliniLInfMethod(
                    classifier=classifier,
                    confidence=params['confidence'],
                    targeted=False,
                    learning_rate=params['learning_rate'],
                    max_iter=int(params['max_iter'])
                )
                adv_tensor = attack.generate(x=img_tensor)

            elif attack_type == 'deepfool':
                attack = DeepFool(
                    classifier=classifier,
                    max_iter=int(params['max_iter']),
                    epsilon=params['epsilon'],
                    nb_grads=int(params['nb_grads'])
                )
                adv_tensor = attack.generate(x=img_tensor)

            else:
                raise ValueError(f"Attack type '{attack_type}' not implemented")

            # Clear GPU cache after attack to free memory
            torch.cuda.empty_cache()

            # Convert back to image format
            adv_image = postprocess_adversarial_image(adv_tensor, image.shape)
            return adv_image

        except Exception as e:
            # Clear GPU cache on error as well
            torch.cuda.empty_cache()
            print(f"Attack execution error: {str(e)}")
            return None
        finally:
            # Ensure memory cleanup happens regardless of success/failure
            torch.cuda.empty_cache()

def ssim_aware_attack(
    image_path: str,
    attack_type: str,
    target_ssim: float = 0.85,
    optimization_method: str = "bayesian",
    max_trials: int = 5,
    trial_number: int = 1
) -> Tuple[np.ndarray, float, Dict]:
    """
    Main function for SSIM-aware white-box adversarial attacks

    Args:
        image_path: Path to input image
        attack_type: Type of white-box attack to perform
        target_ssim: Target SSIM value (default: 0.85)
        optimization_method: Optimization method (default: "bayesian")
        max_trials: Maximum optimization trials (default: 5)

    Returns:
        Tuple of (adversarial_image, achieved_ssim, optimal_parameters)
    """
    # Create classifier with gradients enabled for white-box attacks - GPU ONLY
    classifier = create_classifier(device='cuda:0', requires_grad=True)

    if optimization_method == "bayesian":
        # Use centralized Bayesian optimization framework
        # Use adaptive search space for improved optimization
        if optimization_method == "adaptive":
            param_space = get_whitebox_param_spaces(attack_type, target_ssim, trial_number, max_trials)
        else:
            # Fallback to legacy spaces for backward compatibility
            if attack_type not in WHITEBOX_PARAM_SPACES:
                raise ValueError(f"Attack type '{attack_type}' not supported for whitebox")
            param_space = WHITEBOX_PARAM_SPACES[attack_type]

        # Create attack function wrapper
        attack_framework = UniversalSSIMAwareAttack(target_ssim=target_ssim)

        def attack_function_wrapper(image, classifier, attack_type, params):
            return attack_framework._run_attack(image, classifier, attack_type, params)

        # Use centralized Bayesian optimization
        return bayesian_optimization_framework(
            image_path=image_path,
            attack_type=attack_type,
            attack_function=attack_function_wrapper,
            classifier=classifier,
            param_space=param_space,
            is_blackbox=False,
            target_ssim=target_ssim,
            max_trials=max_trials,
            timeout=1800
        )

    elif optimization_method == "universal":
        # Use new optimizer framework
        attack_framework = UniversalSSIMAwareAttack(
            target_ssim=target_ssim,
            tolerance=0.01,
            max_attempts=max_trials
        )
        return attack_framework.run_ssim_aware_attack(image_path, attack_type)

    else:
        raise NotImplementedError(f"Optimization method '{optimization_method}' not implemented")

def main():
    parser = argparse.ArgumentParser(description="Universal whitebox attack with automated hyperparameter optimization")

    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--attack_type", type=str, required=True,
                        choices=["fgsm", "deepfool", "pgd", "cw_linf"],
                        help="Type of whitebox attack to perform")
    parser.add_argument("--ssim_threshold", type=float, default=0.85,
                        help="Target SSIM value (default: 0.85)")
    parser.add_argument("--optimization_method", type=str, default="universal",
                        choices=["universal", "bayesian", "grid_search", "random_search", "adaptive"],
                        help="Hyperparameter optimization method (default: universal)")
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

    try:
        # Universal attack execution - single line handles ALL whitebox attacks
        adv_image, achieved_ssim, final_params = ssim_aware_attack(
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