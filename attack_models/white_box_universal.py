#!/usr/bin/env python3
"""
White-Box Adversarial Attacks Universal Module

Supports: FGSM, PGD, AutoPGD, AutoConjugateGradient, BasicIterativeMethod attacks
Usage: python white_box_universal.py --attack_type fgsm --image_path ... --epsilon ...

This module provides specialized utilities for white-box adversarial attacks with direct epsilon control.

White-Box Attacks Supported (100% EPSILON PARAMETER SUPPORT - 5 attacks):
- FGSM (Fast Gradient Sign Method) - ε parameter
- PGD (Projected Gradient Descent) - ε parameter
- AutoPGD (Auto Projected Gradient Descent) - ε parameter
- AutoConjugateGradient - ε parameter
- BasicIterativeMethod - ε parameter

ALL ATTACKS SUPPORT EPSILON:
- Direct epsilon parameter control for precise perturbation bounds
- No optimization needed - fast and predictable execution
- 100% epsilon parameter compatibility across all attack types
"""

import os
import sys
import numpy as np
import torch
import logging
import time
import argparse
from art.attacks.evasion import (
    ProjectedGradientDescent, FastGradientMethod,
    AutoProjectedGradientDescent, AutoConjugateGradient, BasicIterativeMethod
)
from typing import Dict, List, Tuple, Optional, Any

# Import common utilities using absolute imports
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
    # Fallback for direct execution
    from utils import (
        load_image, create_classifier, save_image, preprocess_image_for_attack,
        postprocess_adversarial_image, get_output_path,
        fast_perturbation_calculation, fast_clip_operation, print_attack_info,
        calculate_epsilon, refine_epsilon_tolerance, logger, query_counter,
        batch_preprocess_images, batch_postprocess_images, get_optimal_batch_size,
        optimize_memory_usage, get_gpu_memory_info, setup_gpu_optimizations
    )

# WHITE-BOX PARAMETER SPACES - 100% EPSILON PARAMETER SUPPORT
# NOTE: All attacks now use direct epsilon control for consistent perturbation bounds
def get_whitebox_param_spaces(attack_type: str, epsilon_target: float = 0.05) -> Dict:
    """Get parameter spaces for white-box attacks with direct epsilon control"""
    base_params = {
        'fgsm': {'eps': epsilon_target, 'norm': np.inf},
        'pgd': {'eps': epsilon_target, 'eps_step': epsilon_target/10, 'nb_iter': 40},
        'auto_pgd': {'eps': epsilon_target, 'norm': np.inf, 'nb_iter': 100},
        'auto_conjugate_gradient': {'eps': epsilon_target, 'norm': np.inf, 'nb_iter': 100},
        'basic_iterative': {'eps': epsilon_target, 'eps_step': epsilon_target/7, 'nb_iter': 40}
    }
    return base_params.get(attack_type, {})

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
    'auto_pgd': {
        'eps': [0.01, 0.03, 0.05, 0.08, 0.1],
        'norm': [np.inf, 2],
        'nb_iter': [50, 100, 150, 200]
    },
    'auto_conjugate_gradient': {
        'eps': [0.01, 0.03, 0.05, 0.08, 0.1],
        'norm': [np.inf, 2],
        'nb_iter': [50, 100, 150, 200]
    },
    'basic_iterative': {
        'eps': [0.01, 0.03, 0.05, 0.08, 0.1],
        'eps_step': [0.005, 0.01, 0.02, 0.03],
        'nb_iter': [20, 40, 60, 80]
    }
}

# Maintain backward compatibility
WHITEBOX_PARAM_SPACES = WHITEBOX_PARAM_SPACES_LEGACY

# UNIVERSAL EPSILON-BASED ATTACK FRAMEWORK
class UniversalEpsilonAttack:
    """
    Universal epsilon-based attack that works with ANY white-box attack type.
    Uses direct epsilon control instead of optimization.
    """

    def __init__(self, epsilon_target=0.05):
        self.epsilon_target = epsilon_target



    def _run_attack_with_params(self, image: np.ndarray, classifier, attack_type: str, params: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Wrapper for running attack with parameters - simplified for direct epsilon control

        Args:
            image: Input image
            classifier: Attack classifier
            attack_type: Type of attack
            params: Attack parameters including epsilon

        Returns:
            Adversarial image or None if attack fails
        """
        try:
            return self._run_attack(image, classifier, attack_type, params)
        except Exception as e:
            print(f"Attack failed with params {params}: {e}")
            return None

    def run_epsilon_attack(self, image_path: str, attack_type: str,
                          attack_params: Dict = None) -> Tuple[np.ndarray, float, float, Dict]:
        """
        Universal epsilon-based attack with direct epsilon control

        Args:
            image_path: Path to target image
            attack_type: Type of attack ('fgsm', 'pgd', 'auto_pgd', 'auto_conjugate_gradient', 'basic_iterative')
            attack_params: Base attack parameters (optional, uses defaults if None)

        Returns:
            Tuple of (adversarial_image, epsilon_target, epsilon_l_inf, final_parameters)
        """
        print(f"\n🎯 Universal Epsilon-Based Attack")
        print(f"Target epsilon: {self.epsilon_target:.6f}")
        print(f"Attack: {attack_type.upper()}")

        # Load image and create classifier
        image = load_image(image_path)
        classifier = create_classifier()

        # Get default parameters if none provided
        if attack_params is None:
            attack_params = self._get_default_params(attack_type)

        # Set target epsilon directly in parameters
        attack_params['eps'] = self.epsilon_target
        print(f"Attack parameters: {attack_params}")

        # Run attack with target epsilon (no optimization needed)
        adv_image = self._run_attack(image, classifier, attack_type, attack_params)

        if adv_image is None:
            raise RuntimeError(f"Attack '{attack_type}' failed to generate adversarial examples")

        # Save result (epsilon refinement already completed)
        output_path = get_output_path(image_path, attack_type, is_blackbox=False, epsilon=self.epsilon_target)
        # Extract actual image from wrapper for saving
        image_to_save = adv_image.image if hasattr(adv_image, 'image') else adv_image
        save_image(image_to_save, output_path)

        # Get actual query count from the global counter
        actual_queries = query_counter.get_count()
        print_attack_info(output_path, image, image_to_save, attack_type, query_count=actual_queries)

        print(f"✅ Target epsilon: {self.epsilon_target:.6f}")
        # Note: Final epsilon is reported by print_attack_info as "Epsilon (L∞)"

        # Extract final epsilon from the refined result for return
        final_epsilon = adv_image._epsilon_l_inf if hasattr(adv_image, '_epsilon_l_inf') else self.epsilon_target

        return adv_image, self.epsilon_target, final_epsilon, attack_params

    def _get_default_params(self, attack_type: str) -> Dict:
        """Get smart parameter initialization based on literature and empirical results"""
        # Literature-proven parameter ranges for optimal performance - 100% EPSILON SUPPORT
        smart_defaults = {
            'fgsm': {  # Goodfellow et al. 2014 - "Explaining and Harnessing Adversarial Examples"
                'eps': 0.05,  # Goodfellow et al. 2014 - ε for single-step perturbation
                'norm': np.inf  # Goodfellow et al. 2014 - L∞ norm constraint
            },
            'pgd': {  # Madry et al. 2017 - "Towards Deep Learning Models Resistant to Adversarial Attacks"
                'eps': 0.05,  # Madry et al. 2017 - ε perturbation bound (paper uses ε=0.3 for CIFAR-10)
                'eps_step': 0.01,  # Madry et al. 2017 - α = ε/iterations rule (paper: α=2ε/iterations)
                'nb_iter': 40  # Madry et al. 2017 - convergence iterations (paper uses 40 steps)
            },
            'auto_pgd': {  # Croce & Hein 2020 - "Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks"
                'eps': 0.05,  # Epsilon constraint for L∞ perturbation
                'norm': np.inf,  # L∞ norm constraint
                'nb_iter': 100  # Auto-tuned iterations
            },
            'auto_conjugate_gradient': {  # Yamamura et al. 2022 - "Auto Conjugate Gradient"
                'eps': 0.05,  # Epsilon constraint for perturbation bound
                'norm': np.inf,  # L∞ norm constraint
                'nb_iter': 100  # Conjugate gradient iterations
            },
            'basic_iterative': {  # Kurakin et al. 2017 - "Adversarial examples in the physical world"
                'eps': 0.05,  # Epsilon constraint for L∞ perturbation
                'eps_step': 0.007,  # Step size (eps / 7 iterations)
                'nb_iter': 40  # Basic iterative iterations
            }
        }

        if attack_type not in smart_defaults:
            raise ValueError(f"Attack type '{attack_type}' not supported")

        return smart_defaults[attack_type]

    def _scale_parameters(self, base_params: Dict, strength: float) -> Dict:
        """
        Universal parameter scaling that works for ANY attack.
        Higher strength = stronger perturbations = higher epsilon
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

    def _run_batch_attack(self, batch_tensor: torch.Tensor, classifier, attack_type: str, params: Dict) -> List[Optional[np.ndarray]]:
        """Run attack on a batch of images for improved efficiency"""
        try:
            print(f"🔄 Running batch {attack_type} attack on {batch_tensor.shape[0]} images")

            # Convert batch tensor to numpy for ART
            if torch.is_tensor(batch_tensor):
                batch_numpy = batch_tensor.cpu().numpy()
            else:
                batch_numpy = batch_tensor

            # Run attack on entire batch
            if attack_type == 'fgsm':
                attack = FastGradientMethod(
                    estimator=classifier,
                    norm=params.get('norm', np.inf),
                    eps=params['eps'],
                    targeted=False
                )
                adv_batch = attack.generate(x=batch_numpy)

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
                adv_batch = attack.generate(x=batch_numpy)

            else:
                # For other attacks, process individually
                results = []
                for i in range(batch_numpy.shape[0]):
                    single_image = batch_numpy[i:i+1]
                    result = self._run_single_attack(single_image, classifier, attack_type, params)
                    results.append(result)
                return results

            # Convert batch results back to individual images
            results = []
            for i in range(adv_batch.shape[0]):
                single_adv = adv_batch[i]

                # Create result wrapper similar to single attack
                class BatchAdversarialResult:
                    def __init__(self, image, epsilon_l_inf):
                        self.image = image
                        self._epsilon_l_inf = epsilon_l_inf
                        self.shape = image.shape
                        self.dtype = image.dtype

                    def __array__(self):
                        return self.image

                # Calculate epsilon for this image (approximate)
                epsilon_val = params['eps']  # Use target epsilon as approximation for batch
                result = BatchAdversarialResult(single_adv, epsilon_val)
                results.append(result)

            return results

        except Exception as e:
            print(f"❌ Batch attack failed: {e}")
            # Fallback to individual processing
            results = []
            for i in range(batch_tensor.shape[0]):
                single_tensor = batch_tensor[i:i+1] if batch_tensor.dim() == 4 else batch_tensor
                try:
                    # Convert back to image format for individual processing
                    single_image = postprocess_adversarial_image(single_tensor, (224, 224))
                    result = self._run_attack(single_image, classifier, attack_type, params)
                    results.append(result)
                except:
                    results.append(None)
            return results

    def _run_single_attack(self, image_numpy: np.ndarray, classifier, attack_type: str, params: Dict) -> Optional[np.ndarray]:
        """Run attack on a single image (helper for batch processing)"""
        try:
            if attack_type == 'fgsm':
                attack = FastGradientMethod(
                    estimator=classifier,
                    norm=params.get('norm', np.inf),
                    eps=params['eps'],
                    targeted=False
                )
                return attack.generate(x=image_numpy)

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
                return attack.generate(x=image_numpy)

            # Add other attack types as needed
            else:
                return self._run_attack_fallback(image_numpy, classifier, attack_type, params)

        except Exception as e:
            print(f"❌ Single attack failed: {e}")
            return None

    def _run_attack_fallback(self, image_numpy: np.ndarray, classifier, attack_type: str, params: Dict) -> Optional[np.ndarray]:
        """Fallback to original attack method for unsupported batch operations"""
        # Convert numpy back to image format
        if len(image_numpy.shape) == 4:
            image_numpy = image_numpy[0]  # Remove batch dimension

        # Transpose from CHW to HWC if needed
        if image_numpy.shape[0] == 3:
            image_numpy = np.transpose(image_numpy, (1, 2, 0))

        # Convert to uint8 image format
        image = (image_numpy * 255).astype(np.uint8) if image_numpy.max() <= 1.0 else image_numpy.astype(np.uint8)

        # Use original _run_attack method
        return self._run_attack(image, classifier, attack_type, params)

    def _run_attack(self, image: np.ndarray, classifier, attack_type: str, params: Dict) -> Optional[np.ndarray]:
        """Run attack with GPU memory optimizations and epsilon tolerance control"""

        # Clear GPU cache before attack to free memory
        torch.cuda.empty_cache()

        # ART expects numpy arrays, not tensors
        img_tensor = preprocess_image_for_attack(image, return_tensor=False)

        try:
            # All attacks support epsilon parameter - 100% EPSILON COMPATIBILITY
            if attack_type == 'fgsm':
                attack = FastGradientMethod(
                    estimator=classifier,
                    norm=params.get('norm', np.inf),
                    eps=params['eps'],
                    targeted=False
                )
                adv_tensor = attack.generate(x=img_tensor)

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

            elif attack_type == 'auto_pgd':
                attack = AutoProjectedGradientDescent(
                    estimator=classifier,
                    norm=params.get('norm', np.inf),
                    eps=params['eps'],
                    max_iter=int(params.get('nb_iter', 100)),
                    targeted=False
                )
                adv_tensor = attack.generate(x=img_tensor)

            elif attack_type == 'auto_conjugate_gradient':
                attack = AutoConjugateGradient(
                    estimator=classifier,
                    norm=params.get('norm', np.inf),
                    eps=params['eps'],
                    max_iter=int(params.get('nb_iter', 100)),
                    targeted=False
                )
                adv_tensor = attack.generate(x=img_tensor)

            elif attack_type == 'basic_iterative':
                attack = BasicIterativeMethod(
                    estimator=classifier,
                    eps=params['eps'],
                    eps_step=params['eps_step'],
                    max_iter=int(params['nb_iter']),
                    targeted=False
                )
                adv_tensor = attack.generate(x=img_tensor)

            else:
                raise ValueError(f"Attack type '{attack_type}' not implemented")

            # Clear GPU cache after attack to free memory
            torch.cuda.empty_cache()

            # Calculate epsilon on raw tensors before uint8 conversion (precision loss)
            # Note: preprocess_image_for_attack already imported at top of file
            original_tensor = preprocess_image_for_attack(image, return_tensor=False)

            # Calculate epsilon in the same space where attack operates ([0,1] normalized)
            epsilon_tensor = float(np.max(np.abs(adv_tensor - original_tensor)))

            # Convert back to image format for saving
            adv_image = postprocess_adversarial_image(adv_tensor, image.shape)

            # ITERATIVE EPSILON REFINEMENT for ±5% tolerance (prioritize accuracy over speed)
            adv_image, final_epsilon = refine_epsilon_tolerance(
                original_image=image,
                adversarial_image=adv_image,
                target_epsilon=self.epsilon_target,
                tolerance=0.05,  # ±5%
                max_iterations=100  # High iterations for accuracy
            )

            # Store epsilon result for later retrieval using a wrapper class
            class AdversarialResult:
                def __init__(self, image, epsilon_l_inf):
                    self.image = image
                    self._epsilon_l_inf = epsilon_l_inf
                    # Make it behave like an array for compatibility
                    self.shape = image.shape
                    self.dtype = image.dtype

                def __array__(self):
                    return self.image

                def __getattr__(self, name):
                    return getattr(self.image, name)

            return AdversarialResult(adv_image, final_epsilon)

        except Exception as e:
            # Clear GPU cache on error as well
            torch.cuda.empty_cache()
            print(f"Attack execution error: {str(e)}")
            return None
        finally:
            # Ensure memory cleanup happens regardless of success/failure
            torch.cuda.empty_cache()

def epsilon_based_attack(
    image_path: str,
    attack_type: str,
    epsilon_target: float = 0.05,
    max_trials: int = 1
) -> Tuple[np.ndarray, float, float, Dict]:
    """
    Main function for epsilon-based white-box adversarial attacks

    Args:
        image_path: Path to input image
        attack_type: Type of white-box attack to perform
        epsilon_target: Target epsilon value (default: 0.05)
        max_trials: Maximum trials for attack execution (default: 1)

    Returns:
        Tuple of (adversarial_image, epsilon_target, epsilon_l_inf, optimal_parameters)
    """
    # Reset query counter before attack
    query_counter.reset()

    # Force dynamic batch sizing to execute even for single images
    optimal_batch = get_optimal_batch_size()
    print(f"🔧 [ATTACK] Batch size analysis completed for attack planning")

    # Create classifier with gradients enabled for white-box attacks - GPU ONLY
    # Note: TensorRT disabled for white-box due to requires_grad=True incompatibility
    classifier = create_classifier(device='cuda:0', requires_grad=True, count_queries=True, use_tensorrt=False)

    print(f"🎯 Direct Epsilon-Based Attack (no optimization needed)")
    print(f"Target epsilon: {epsilon_target}")

    # Load image
    image = load_image(image_path)

    # Get default parameters and set target epsilon
    attack_framework = UniversalEpsilonAttack(epsilon_target=epsilon_target)
    base_params = attack_framework._get_default_params(attack_type)
    base_params['eps'] = epsilon_target  # Set target epsilon directly

    print(f"Attack parameters: {base_params}")

    # Run attack with target epsilon
    adv_image = attack_framework._run_attack(image, classifier, attack_type, base_params)

    if adv_image is None:
        raise RuntimeError(f"Attack '{attack_type}' failed to generate adversarial examples")

    # Save output (epsilon refinement already completed)
    output_path = get_output_path(image_path, attack_type, is_blackbox=False, epsilon=epsilon_target)
    # Extract actual image from wrapper for saving
    image_to_save = adv_image.image if hasattr(adv_image, 'image') else adv_image
    save_image(image_to_save, output_path)

    # Get actual query count
    actual_queries = query_counter.get_count()
    print_attack_info(output_path, image, image_to_save, attack_type, query_count=actual_queries)

    print(f"✅ Target epsilon: {epsilon_target:.6f}")
    # Note: Final epsilon is reported by print_attack_info as "Epsilon (L∞)"

    # Extract final epsilon from the refined result
    final_epsilon = adv_image._epsilon_l_inf if hasattr(adv_image, '_epsilon_l_inf') else epsilon_target

    return adv_image, epsilon_target, final_epsilon, base_params

def batch_epsilon_attack(image_paths: List[str], attack_type: str,
                        epsilon_target: float = 0.05, optimization_level: str = 'high') -> List[Dict]:
    """
    Optimized batch processing for multiple white-box adversarial attacks

    Args:
        image_paths: List of image paths to process
        attack_type: Type of white-box attack
        epsilon_target: Target epsilon value
        optimization_level: GPU optimization level ('basic', 'high', 'extreme')

    Returns:
        List of attack results with metrics
    """
    print(f"🚀 Batch White-Box Attack: {len(image_paths)} images")
    print(f"Attack: {attack_type.upper()}, Epsilon: {epsilon_target}, Optimization: {optimization_level}")

    # Reset query counter and setup optimizations
    query_counter.reset()
    setup_gpu_optimizations()

    # Show initial memory state
    memory_info = get_gpu_memory_info()
    print(f"🔍 Initial GPU Memory: {memory_info['free_gb']:.1f}GB free")

    # Create optimized classifier
    # TensorRT disabled for white-box attacks due to requires_grad=True incompatibility
    classifier = create_classifier(
        device='cuda:0',
        requires_grad=True,
        count_queries=True,
        optimization_level=optimization_level,
        use_tensorrt=False  # TensorRT incompatible with gradients
    )

    # Determine optimal batch size
    batch_size = get_optimal_batch_size()
    results = []

    # Process images in optimized batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        print(f"\n📦 Processing batch {i//batch_size + 1}: {len(batch_paths)} images")

        try:
            # Batch preprocessing
            batch_tensors, batch_images = batch_preprocess_images(batch_paths)

            # Process each tensor batch
            batch_results = []
            for tensor_batch, image_batch in zip(batch_tensors, batch_images):

                # Create attack framework for this batch
                attack_framework = UniversalEpsilonAttack(epsilon_target=epsilon_target)
                base_params = attack_framework._get_default_params(attack_type)
                base_params['eps'] = epsilon_target

                # Run batch attack with mixed precision
                with torch.amp.autocast('cuda', enabled=(optimization_level in ['high', 'extreme'])):
                    if tensor_batch.dim() == 4 and tensor_batch.shape[0] > 1:
                        # True batch processing
                        adv_tensor_batch = attack_framework._run_batch_attack(
                            tensor_batch, classifier, attack_type, base_params
                        )
                    else:
                        # Single image in batch format
                        single_tensor = tensor_batch.squeeze(0) if tensor_batch.dim() == 4 else tensor_batch
                        single_image = image_batch[0] if isinstance(image_batch, list) else image_batch

                        adv_result = attack_framework._run_attack(
                            single_image, classifier, attack_type, base_params
                        )
                        adv_tensor_batch = [adv_result]

                # Post-process results
                if isinstance(adv_tensor_batch, list):
                    for j, adv_result in enumerate(adv_tensor_batch):
                        if adv_result is not None:
                            original_image = image_batch[j] if isinstance(image_batch, list) else image_batch
                            path_idx = i + len(batch_results) + j

                            if path_idx < len(batch_paths):
                                result = process_single_result(
                                    adv_result, original_image, batch_paths[path_idx],
                                    attack_type, epsilon_target
                                )
                                batch_results.append(result)

            results.extend(batch_results)

            # Memory optimization between batches
            if i + batch_size < len(image_paths):
                optimize_memory_usage()

        except Exception as e:
            print(f"❌ Batch {i//batch_size + 1} failed: {e}")
            # Continue with next batch
            continue

    # Final statistics
    total_queries = query_counter.get_count()
    successful_attacks = len([r for r in results if r['success']])

    print(f"\n✅ Batch Processing Complete:")
    print(f"   📊 Total images: {len(image_paths)}")
    print(f"   ✅ Successful: {successful_attacks}")
    print(f"   🔢 Total queries: {total_queries}")
    print(f"   ⚡ Avg queries per image: {total_queries/len(image_paths):.1f}")

    final_memory = get_gpu_memory_info()
    print(f"   🧠 Final GPU Memory: {final_memory['free_gb']:.1f}GB free")

    return results

def process_single_result(adv_result, original_image, image_path, attack_type, epsilon_target):
    """Process a single attack result and save it"""
    try:
        # Extract image from result wrapper
        image_to_save = adv_result.image if hasattr(adv_result, 'image') else adv_result

        # Save result
        output_path = get_output_path(image_path, attack_type, is_blackbox=False, epsilon=epsilon_target)
        save_image(image_to_save, output_path)

        # Calculate metrics
        epsilon_l_inf = calculate_epsilon(original_image, image_to_save)
        final_epsilon = adv_result._epsilon_l_inf if hasattr(adv_result, '_epsilon_l_inf') else epsilon_l_inf

        return {
            'image_path': image_path,
            'output_path': output_path,
            'success': True,
            'epsilon_target': epsilon_target,
            'epsilon_l_inf': final_epsilon,
            'adversarial_image': image_to_save
        }

    except Exception as e:
        print(f"❌ Failed to process result for {image_path}: {e}")
        return {
            'image_path': image_path,
            'success': False,
            'error': str(e)
        }

def optimized_epsilon_attack(image_path: str, attack_type: str, epsilon_target: float = 0.05,
                           optimization_level: str = 'high') -> Tuple[np.ndarray, float, float, Dict]:
    """
    Single image attack with GPU optimizations

    Args:
        image_path: Path to input image
        attack_type: Type of white-box attack
        epsilon_target: Target epsilon value
        optimization_level: GPU optimization level

    Returns:
        Tuple of (adversarial_image, epsilon_target, epsilon_l_inf, parameters)
    """
    # For single images, use batch processing with batch_size=1 for consistency
    results = batch_epsilon_attack([image_path], attack_type, epsilon_target, optimization_level)

    if results and results[0]['success']:
        result = results[0]
        return (
            result['adversarial_image'],
            result['epsilon_target'],
            result['epsilon_l_inf'],
            {'eps': epsilon_target}
        )
    else:
        raise RuntimeError(f"Optimized attack '{attack_type}' failed")

def main():
    parser = argparse.ArgumentParser(description="Universal whitebox attack with direct epsilon control")

    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--attack_type", type=str, required=True,
                        choices=["fgsm", "pgd", "auto_pgd", "auto_conjugate_gradient", "basic_iterative"],
                        help="Type of whitebox attack to perform")
    parser.add_argument("--epsilon", type=float, default=0.05,
                        help="Target epsilon value for perturbation (default: 0.05)")
    parser.add_argument("--max_trials", type=int, default=1,
                        help="Maximum trials (not needed for direct epsilon control, default: 1)")
    parser.add_argument("--trial_number", type=int, default=1,
                        help="Current trial number (default: 1)")

    args = parser.parse_args()

    # Check if CUDA is available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Attack type: {args.attack_type.upper()}")
    print(f"Target epsilon: {args.epsilon}")

    try:
        # Universal attack execution - single line handles ALL whitebox attacks
        adv_image, epsilon_target, epsilon_l_inf, final_params = epsilon_based_attack(
            image_path=args.image_path,
            attack_type=args.attack_type,  # This single parameter determines the attack
            epsilon_target=args.epsilon,
            max_trials=args.max_trials
        )

        print(f"Target epsilon: {epsilon_target:.6f}")
        print(f"Actual epsilon: {epsilon_l_inf:.6f}")
        print(f"Final parameters: {final_params}")
        print(f"Attack: {args.attack_type.upper()} completed successfully")

    except Exception as e:
        print(f"{args.attack_type.upper()} attack failed: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())