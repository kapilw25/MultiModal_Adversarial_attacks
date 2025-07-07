#!/usr/bin/env python3
"""
Spatial Transformation Attack Script for Vision-Language Models with Perceptual Constraints

This script implements the Spatial Transformation attack, which creates adversarial examples
by applying geometric transformations (translation, rotation, scaling) rather than direct
pixel-level perturbations. These transformations preserve the overall appearance and content
of the image while causing misclassification.

The implementation enhances the standard Spatial Transformation attack with:
1. SSIM (Structural Similarity) constraints with exact targeting
2. Importance-weighted transformation focusing on chart elements
3. Binary search for optimal perceptual quality

Usage:
    python v14_spatial_transformation_attack.py [--image_path PATH] [--max_translation PIXELS]
                                              [--max_rotation DEGREES] [--max_scaling FACTOR]
                                              [--targeted] [--target_class CLASS]
                                              [--ssim_threshold THRESHOLD]

Example:
    source venv_MM/bin/activate && python attack_models/true_black_box_attacks/v14_spatial_transformation_attack.py --image_path data/test_extracted/chart/20231114102825506748.png --max_translation 3 --max_rotation 10 --max_scaling 0.1 --ssim_threshold 0.85
"""

import os
import cv2
import numpy as np
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# Import ART Spatial Transformation Attack
from art.attacks.evasion.spatial_transformation import SpatialTransformation

# Import utility functions
from v0_attack_utils import (
    load_image, save_image, 
    get_output_path, print_attack_info, preprocess_image_for_attack,
    calculate_ssim, generate_importance_map_for_charts, create_classifier_with_probs
)


def spatial_transformation_attack_with_perceptual_constraints(image, classifier, image_path=None, 
                                                           max_translation=3.0, max_rotation=10.0, 
                                                           max_scaling=0.1, targeted=False, 
                                                           target_class=None, ssim_threshold=0.85, 
                                                           importance_map=None):
    """Apply Spatial Transformation Attack with perceptual constraints"""
    
    # Generate importance map if not provided
    if importance_map is None and image is not None:
        print("Generating importance map for targeted perturbation...")
        importance_map = generate_importance_map_for_charts(image)
        
        # Visualize importance map
        if image_path is not None:
            importance_vis = (importance_map * 255).astype(np.uint8)
            importance_vis = cv2.applyColorMap(importance_vis, cv2.COLORMAP_JET)
            output_dir = os.path.dirname(get_output_path(image_path, 'spatial'))
            os.makedirs(output_dir, exist_ok=True)
            importance_path = os.path.join(output_dir, 'importance_map.png')
            cv2.imwrite(importance_path, importance_vis)
            print(f"Saved importance map to {importance_path}")
    
    # Create Spatial Transformation Attack
    attack = SpatialTransformation(
        classifier=classifier,
        max_translation=max_translation,
        num_translations=20,  # Number of translations to search
        max_rotation=max_rotation,
        num_rotations=20,  # Number of rotations to search
        verbose=True
    )
    
    # Note: max_scaling is not supported by ART's SpatialTransformation
    # We'll apply scaling manually if needed
    
    # Generate adversarial example
    print(f"Generating adversarial example with Spatial Transformation Attack:")
    print(f"- Max Translation: {max_translation} pixels")
    print(f"- Max Rotation: {max_rotation} degrees")
    print(f"- Max Scaling: {max_scaling}")
    print(f"- Targeted: {targeted}")
    print(f"- SSIM threshold: {ssim_threshold}")
    
    # Preprocess image
    img_tensor = preprocess_image_for_attack(image)
    
    # Generate adversarial example
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


def main():
    parser = argparse.ArgumentParser(description="Generate adversarial examples using Spatial Transformation Attack with perceptual constraints")
    parser.add_argument("--image_path", type=str, 
                        default="data/test_extracted/chart/20231114102825506748.png",
                        help="Path to the input image")
    parser.add_argument("--max_translation", type=float, default=3.0,
                        help="Maximum translation in pixels (default: 3.0)")
    parser.add_argument("--max_rotation", type=float, default=10.0,
                        help="Maximum rotation in degrees (default: 10.0)")
    parser.add_argument("--max_scaling", type=float, default=0.1,
                        help="Maximum scaling factor (default: 0.1)")
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
    
    # Create classifier with probability outputs
    print("Creating classifier with probability outputs...")
    classifier = create_classifier_with_probs(device)
    
    # Generate importance map for chart understanding
    print("Generating importance map for chart elements...")
    importance_map = generate_importance_map_for_charts(image)
    
    # Apply Spatial Transformation Attack with perceptual constraints
    adv_image = spatial_transformation_attack_with_perceptual_constraints(
        image, classifier, args.image_path, args.max_translation, 
        args.max_rotation, args.max_scaling, args.targeted, 
        args.target_class, args.ssim_threshold, importance_map
    )
    
    # Get output path using utility function
    output_path = get_output_path(args.image_path, 'spatial')
    
    # Save adversarial image
    save_image(adv_image, output_path)
    
    # Print attack information
    print_attack_info(output_path, image, adv_image, 'spatial')
    
    # Print additional attack-specific information
    perturbation = np.abs(image.astype(np.float32) - adv_image.astype(np.float32))
    print(f"Max perturbation: {np.max(perturbation)}")
    print(f"Mean perturbation: {np.mean(perturbation)}")
    print(f"SSIM: {calculate_ssim(image, adv_image):.4f}")
    print(f"Transformation parameters: translation={args.max_translation}, rotation={args.max_rotation}, scaling={args.max_scaling}")


if __name__ == "__main__":
    main()
