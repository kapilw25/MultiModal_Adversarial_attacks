#!/usr/bin/env python3
"""
Adaptive SSIM Optimizer for Adversarial Attacks

This module provides intelligent parameter optimization for adversarial attacks
using Optuna's Tree-structured Parzen Estimator (TPE) algorithm.

Features:
- Attack-specific parameter spaces
- Bayesian optimization with early stopping
- Fallback to hardcoded multipliers
- Image-specific sensitivity adaptation
"""

import numpy as np
from typing import Dict, Any, Optional, Callable, List
import warnings

# Import Optuna with graceful fallback
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna not available. Install with: pip install optuna")


class AdaptiveSSIMOptimizer:
    """
    Intelligent parameter optimizer for SSIM-aware adversarial attacks
    """
    
    def __init__(self, target_ssim: float = 0.85, tolerance: float = 0.01, max_trials: int = 10):
        self.target_ssim = target_ssim
        self.tolerance = tolerance
        self.max_trials = max_trials
        self.fallback_multipliers = [0.3, 0.5, 0.7, 0.8, 1.0]
        
    def optimize_parameters(self, 
                          attack_func: Callable,
                          base_params: Dict[str, Any],
                          image: np.ndarray,
                          attack_type: str) -> Dict[str, Any]:
        """
        Find optimal attack parameters using intelligent search
        
        Args:
            attack_func: Function that executes the attack
            base_params: Base attack parameters
            image: Input image as numpy array
            attack_type: Type of attack (fgsm, pgd, cw_linf, deepfool)
            
        Returns:
            Dictionary with optimized parameters and results
        """
        if OPTUNA_AVAILABLE:
            try:
                return self._optuna_optimization(attack_func, base_params, image, attack_type)
            except Exception as e:
                print(f"⚠️  Optuna optimization failed: {e}")
                print("🔄 Falling back to hardcoded search...")
        
        return self._fallback_optimization(attack_func, base_params, image, attack_type)
    
    def _optuna_optimization(self, 
                           attack_func: Callable,
                           base_params: Dict[str, Any],
                           image: np.ndarray,
                           attack_type: str) -> Dict[str, Any]:
        """Optuna-based intelligent parameter search"""
        
        best_result = None
        
        def objective(trial):
            nonlocal best_result
            
            # Attack-specific parameter suggestions
            scaled_params = self._suggest_parameters(trial, base_params, attack_type)
            
            try:
                # Execute attack
                adv_image = attack_func(image, **scaled_params)
                
                if adv_image is not None:
                    # Calculate SSIM using absolute import
                    from attack_models.utils import calculate_ssim
                    ssim_value = calculate_ssim(image, adv_image)
                    diff = abs(ssim_value - self.target_ssim)
                    
                    # Update best result
                    if best_result is None or diff < best_result['diff']:
                        best_result = {
                            'adv_image': adv_image,
                            'ssim': ssim_value,
                            'params': scaled_params,
                            'diff': diff,
                            'success': diff <= self.tolerance
                        }
                    
                    print(f"Trial {trial.number + 1}: SSIM={ssim_value:.4f} (diff: {diff:.4f})")
                    
                    # Early stopping if target achieved
                    if diff <= self.tolerance:
                        print(f"🎯 TARGET ACHIEVED! SSIM: {ssim_value:.4f}")
                        trial.study.stop()
                    
                    return diff
                
                return float('inf')
                
            except Exception as e:
                print(f"Trial {trial.number + 1} failed: {e}")
                return float('inf')
        
        # Create Optuna study with intelligent sampling
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42, n_startup_trials=2),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=2)
        )
        
        # Suppress verbose Optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # Optimize with timeout
        study.optimize(objective, n_trials=self.max_trials, timeout=120)
        
        # Print final best result in expected format
        if best_result:
            print(f"🏆 Best result: SSIM diff = {best_result['diff']:.4f}")
        
        return best_result or {'success': False, 'diff': float('inf')}
    
    def _suggest_parameters(self, trial, base_params: Dict[str, Any], attack_type: str) -> Dict[str, Any]:
        """Suggest attack-specific parameters based on attack type"""
        
        if attack_type in ['fgsm', 'pgd']:
            # Gradient-based attacks
            strength = trial.suggest_float('strength', 0.1, 1.0)
            return {k: (v * strength if isinstance(v, (int, float)) else v) 
                   for k, v in base_params.items()}
        
        elif attack_type == 'cw_linf':
            # Carlini & Wagner L∞
            confidence = trial.suggest_float('confidence', 1.0, 15.0)
            max_iter = trial.suggest_int('max_iter', 20, 80)
            lr = trial.suggest_float('learning_rate', 0.003, 0.015)
            return {
                'confidence': confidence,
                'max_iter': max_iter,
                'learning_rate': lr
            }
        
        elif attack_type == 'deepfool':
            # DeepFool geometric attack
            max_iter = trial.suggest_int('max_iter', 10, 100)
            epsilon = trial.suggest_float('epsilon', 1e-7, 1e-5, log=True)
            return {
                **base_params,
                'max_iter': max_iter,
                'epsilon': epsilon
            }
        
        else:
            # Generic scaling for other attacks
            strength = trial.suggest_float('strength', 0.2, 1.2)
            return {k: (v * strength if isinstance(v, (int, float)) else v) 
                   for k, v in base_params.items()}
    
    def _fallback_optimization(self, 
                             attack_func: Callable,
                             base_params: Dict[str, Any],
                             image: np.ndarray,
                             attack_type: str) -> Dict[str, Any]:
        """Fallback optimization using hardcoded multipliers"""
        
        best_result = {'success': False, 'diff': float('inf')}
        
        print("🔄 Using fallback hardcoded multipliers...")
        
        for i, strength in enumerate(self.fallback_multipliers):
            print(f"--- Attempt {i+1}/{len(self.fallback_multipliers)} ---")
            print(f"Strength multiplier: {strength:.1f}")
            
            # Scale parameters
            scaled_params = {k: (v * strength if isinstance(v, (int, float)) else v) 
                           for k, v in base_params.items()}
            print(f"Scaled parameters: {scaled_params}")
            
            try:
                # Execute attack
                adv_image = attack_func(image, **scaled_params)
                
                if adv_image is not None:
                    # Calculate SSIM using absolute import
                    from attack_models.utils import calculate_ssim
                    ssim_value = calculate_ssim(image, adv_image)
                    diff = abs(ssim_value - self.target_ssim)
                    
                    print(f"Achieved SSIM: {ssim_value:.4f} (diff: {diff:.4f})")
                    
                    # Update best result
                    if diff < best_result['diff']:
                        best_result = {
                            'adv_image': adv_image,
                            'ssim': ssim_value,
                            'params': scaled_params,
                            'diff': diff,
                            'success': diff <= self.tolerance
                        }
                        print(f"🎯 New best result: diff={diff:.4f}")
                    
                    # Early termination if target achieved
                    if diff <= self.tolerance:
                        print(f"🎯 TARGET ACHIEVED! SSIM: {ssim_value:.4f}")
                        break
                        
            except Exception as e:
                print(f"Attempt {i+1} failed: {e}")
                continue
        
        # Print final best result in expected format
        if best_result:
            print(f"🏆 Best result: SSIM diff = {best_result['diff']:.4f}")
        
        return best_result


def get_adaptive_search_space(attack_type: str) -> Dict[str, Any]:
    """Get attack-specific parameter search space"""
    spaces = {
        'spatial': {
            'dx_max': (1, 10),
            'dy_max': (1, 10), 
            'angle_max': (1, 30),
            'scale_factor': (0.8, 1.2)
        },
        'pixel': {
            'max_pixels': (10, 100),
            'pixel_value': (0, 255)
        }
    }
    return spaces.get(attack_type, {})


def create_optimizer(target_ssim: float = 0.85, 
                    tolerance: float = 0.01, 
                    max_trials: int = 10) -> AdaptiveSSIMOptimizer:
    """
    Factory function to create an adaptive SSIM optimizer
    
    Args:
        target_ssim: Target SSIM value (default: 0.85)
        tolerance: Acceptable difference from target (default: 0.01)
        max_trials: Maximum optimization trials (default: 10)
        
    Returns:
        AdaptiveSSIMOptimizer instance
    """
    return AdaptiveSSIMOptimizer(target_ssim, tolerance, max_trials)
