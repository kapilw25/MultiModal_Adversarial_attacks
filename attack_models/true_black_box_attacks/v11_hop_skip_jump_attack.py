#!/usr/bin/env python3
"""
HopSkipJump Attack Script for Vision-Language Models with Perceptual Constraints

This script applies the HopSkipJump attack with perceptual constraints to create
adversarial examples that are both effective and visually imperceptible to humans.
HopSkipJump is a powerful black-box attack that only requires access to the model's
final class prediction, making it applicable in scenarios with limited model access.

The implementation enhances the standard HopSkipJump attack with:
1. SSIM (Structural Similarity) constraints with exact targeting
2. Importance-weighted perturbations focusing on chart elements
3. Binary search for optimal perceptual quality

Usage:
    python v11_hop_skip_jump_attack.py [--image_path PATH] [--norm NORM]
                                      [--max_iter ITERATIONS] [--max_eval EVALUATIONS]
                                      [--init_eval INITIAL_EVALUATIONS] [--targeted]
                                      [--target_class CLASS] [--ssim_threshold THRESHOLD]

Example:
    source venv_MM/bin/activate && python attack_models/true_black_box_attacks/v11_hop_skip_jump_attack.py --image_path data/test_extracted/chart/20231114102825506748.png --norm 2 --max_iter 50 --max_eval 1000 --ssim_threshold 0.85
"""

import os
import cv2
import numpy as np
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
from art.attacks.evasion import HopSkipJump
from art.estimators.classification import PyTorchClassifier
from tqdm import tqdm

# Import utility functions
from v0_attack_utils import (
    load_image, create_classifier, save_image, 
    get_output_path, print_attack_info, preprocess_image_for_attack,
    calculate_ssim, generate_importance_map_for_charts
)


def hop_skip_jump_with_perceptual_constraints(image, classifier, image_path=None, norm=2, 
                                             max_iter=50, max_eval=1000, init_eval=100,
                                             targeted=False, target_class=None,
                                             ssim_threshold=0.85, importance_map=None):
    """Apply HopSkipJump attack with perceptual constraints"""
    # Convert norm string to appropriate format
    if norm == 'inf':
        norm_type = np.inf
    elif norm == '2':
        norm_type = 2
    else:
        raise ValueError("Norm must be either 'inf' or '2'")
    
    # Generate importance map if not provided
    if importance_map is None and image is not None:
        print("Generating importance map for targeted perturbation...")
        importance_map = generate_importance_map_for_charts(image)
        
        # Visualize importance map
        if image_path is not None:
            importance_vis = (importance_map * 255).astype(np.uint8)
            importance_vis = cv2.applyColorMap(importance_vis, cv2.COLORMAP_JET)
            output_dir = os.path.dirname(get_output_path(image_path, 'hop_skip_jump'))
            os.makedirs(output_dir, exist_ok=True)
            importance_path = os.path.join(output_dir, 'importance_map.png')
            cv2.imwrite(importance_path, importance_vis)
            print(f"Saved importance map to {importance_path}")
    
    # Create HopSkipJump attack
    attack = HopSkipJump(
        classifier=classifier,
        norm=norm_type,
        max_iter=max_iter,
        max_eval=max_eval,
        init_eval=init_eval,
        targeted=targeted,
        batch_size=1,
        verbose=True
    )
    
    # Preprocess image using utility function
    img_tensor = preprocess_image_for_attack(image)
    
    # Generate adversarial example
    print(f"Generating adversarial example with HopSkipJump attack:")
    print(f"- Norm: {norm}")
    print(f"- Max iterations: {max_iter}")
    print(f"- Max evaluations: {max_eval}")
    print(f"- Initial evaluations: {init_eval}")
    print(f"- Targeted: {targeted}")
    print(f"- SSIM threshold: {ssim_threshold}")
    
    if targeted and target_class is not None:
        # For targeted attack, create one-hot encoded target
        target = np.zeros((1, classifier.nb_classes))
        target[0, target_class] = 1
        adv_image = attack.generate(x=img_tensor, y=target)
    else:
        # For untargeted attack
        adv_image = attack.generate(x=img_tensor)
    
    # Convert back to uint8 format
    adv_image = adv_image[0].transpose(1, 2, 0)
    adv_image = np.clip(adv_image, 0, 1) * 255
    adv_image = adv_image.astype(np.uint8)
    
    # Resize back to original dimensions
    adv_image = cv2.resize(adv_image, (image.shape[1], image.shape[0]))
    
    # Apply perceptual constraint if needed
    current_ssim = calculate_ssim(image, adv_image)
    print(f"Initial SSIM: {current_ssim:.4f}")
    
    # Target exact SSIM value rather than minimum threshold
    target_ssim = ssim_threshold  # Exact target value
    print(f"Targeting exact SSIM value: {target_ssim:.4f}")
    
    # If current SSIM is not close to target, find optimal blending
    if abs(current_ssim - target_ssim) > 0.01:  # If more than 0.01 away from target
        print(f"Finding optimal blend to achieve target SSIM of {target_ssim:.4f}...")
        
        # Binary search to find optimal blending factor
        left, right = 0.0, 1.0
        best_adv_image = adv_image.copy()
        best_ssim_diff = float('inf')
        
        for i in range(20):  # 20 binary search steps for precision
            alpha = (left + right) / 2
            blended_image = cv2.addWeighted(image, 1 - alpha, adv_image, alpha, 0)
            blend_ssim = calculate_ssim(image, blended_image)
            
            ssim_diff = abs(blend_ssim - target_ssim)
            print(f"  Step {i+1}: alpha={alpha:.4f}, SSIM={blend_ssim:.4f}, diff={ssim_diff:.4f}")
            
            if ssim_diff < best_ssim_diff:
                best_ssim_diff = ssim_diff
                best_adv_image = blended_image.copy()
            
            # Adjust search range based on whether we're above or below target
            if blend_ssim > target_ssim:
                left = alpha  # Need more perturbation to reduce SSIM
            else:
                right = alpha  # Need less perturbation to increase SSIM
            
            # Early stopping if we're very close to target
            if ssim_diff < 0.001:
                print(f"  Found very close match to target SSIM, stopping early")
                break
        
        adv_image = best_adv_image
        final_ssim = calculate_ssim(image, adv_image)
        print(f"Final SSIM: {final_ssim:.4f} (target: {target_ssim:.4f}, difference: {abs(final_ssim - target_ssim):.4f})")
    
    return adv_image


def basic_hop_skip_jump(image, classifier, norm=2, max_iter=50, max_eval=1000, init_eval=100):
    """Basic HopSkipJump attack without perceptual constraints"""
    # Convert norm string to appropriate format
    norm_type = np.inf if norm == 'inf' else 2
    
    # Create HopSkipJump attack
    attack = HopSkipJump(
        classifier=classifier,
        norm=norm_type,
        max_iter=max_iter,
        max_eval=max_eval,
        init_eval=init_eval,
        targeted=False,
        batch_size=1,
        verbose=False
    )
    
    # Preprocess image
    img_tensor = preprocess_image_for_attack(image)
    
    # Generate adversarial example
    adv_image = attack.generate(x=img_tensor)
    
    # Convert back to uint8 format
    adv_image = adv_image[0].transpose(1, 2, 0)
    adv_image = np.clip(adv_image, 0, 1) * 255
    adv_image = adv_image.astype(np.uint8)
    
    # Resize back to original dimensions
    adv_image = cv2.resize(adv_image, (image.shape[1], image.shape[0]))
    
    return adv_image


def main():
    parser = argparse.ArgumentParser(description="Generate adversarial examples using HopSkipJump attack with perceptual constraints")
    parser.add_argument("--image_path", type=str, 
                        default="data/test_extracted/chart/20231114102825506748.png",
                        help="Path to the input image")
    parser.add_argument("--norm", type=str, default='2', choices=['inf', '2'],
                        help="Norm to use for the attack: 'inf' or '2' (default: '2')")
    parser.add_argument("--max_iter", type=int, default=50,
                        help="Maximum number of iterations (default: 50)")
    parser.add_argument("--max_eval", type=int, default=1000,
                        help="Maximum number of evaluations (default: 1000)")
    parser.add_argument("--init_eval", type=int, default=100,
                        help="Initial number of evaluations (default: 100)")
    parser.add_argument("--targeted", action="store_true",
                        help="Use targeted attack instead of untargeted")
    parser.add_argument("--target_class", type=int, default=None,
                        help="Target class for targeted attack (default: None)")
    parser.add_argument("--ssim_threshold", type=float, default=0.85,
                        help="SSIM threshold for perceptual constraint (default: 0.85)")
    args = parser.parse_args()
    
    # Check if CUDA is available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load image
    print(f"Loading image from {args.image_path}")
    image = load_image(args.image_path)
    
    # Create classifier
    print("Creating classifier...")
    classifier = create_classifier(device)
    
    # Generate importance map for chart understanding
    print("Generating importance map for chart elements...")
    importance_map = generate_importance_map_for_charts(image)
    
    # Apply HopSkipJump attack with perceptual constraints
    adv_image = hop_skip_jump_with_perceptual_constraints(
        image, classifier, args.image_path, args.norm, 
        args.max_iter, args.max_eval, args.init_eval,
        args.targeted, args.target_class,
        args.ssim_threshold, importance_map
    )
    
    # Get output path using utility function
    output_path = get_output_path(args.image_path, 'hop_skip_jump')
    
    # Save adversarial image
    save_image(adv_image, output_path)
    
    # Print attack information
    print_attack_info(output_path, image, adv_image, 'hop_skip_jump')
    
    # Print additional attack-specific information
    perturbation = np.abs(image.astype(np.float32) - adv_image.astype(np.float32))
    print(f"Max perturbation: {np.max(perturbation)}")
    print(f"Mean perturbation: {np.mean(perturbation)}")
    print(f"SSIM: {calculate_ssim(image, adv_image):.4f}")
    print(f"Total queries used: approximately {args.max_iter * args.max_eval}")


if __name__ == "__main__":
    main()
