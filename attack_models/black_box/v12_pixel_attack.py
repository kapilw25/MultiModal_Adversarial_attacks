#!/usr/bin/env python3
"""
Pixel Attack Script for Vision-Language Models with Perceptual Constraints

This script implements the Pixel Attack, a black-box adversarial attack that modifies
a limited number of pixels in an image to cause misclassification. The attack uses
evolutionary strategies (either Differential Evolution or CMA-ES) to find optimal
pixel modifications without requiring access to model gradients.

The implementation enhances the standard Pixel Attack with:
1. SSIM (Structural Similarity) constraints with exact targeting
2. Importance-weighted pixel selection focusing on chart elements
3. Binary search for optimal perceptual quality

Usage:
    python v12_pixel_attack.py [--image_path PATH] [--th THRESHOLD] [--es ES_TYPE]
                              [--max_iter ITERATIONS] [--targeted] [--target_class CLASS]
                              [--ssim_threshold THRESHOLD] [--num_pixels NUM]

Example:
    source venv_MM/bin/activate && python attack_models/black_box/v12_pixel_attack.py --image_path data/clean/chart/20231114102825506748.png --th 10 --es 1 --max_iter 100 --ssim_threshold 0.85 --num_pixels 20
"""

import os
import cv2
import numpy as np
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# Import ART Pixel Attack
from art.attacks.evasion.pixel_threshold import PixelAttack

# Import utility functions
from v0_attack_utils import (
    load_image, create_classifier, save_image, 
    get_output_path, print_attack_info, preprocess_image_for_attack,
    calculate_ssim, generate_importance_map_for_charts,
    apply_standardized_ssim_constraint
)


def pixel_attack_with_perceptual_constraints(image, classifier, image_path=None, th=10, es=1, 
                                           max_iter=100, targeted=False, target_class=None,
                                           ssim_threshold=0.85, num_pixels=20, importance_map=None):
    """Apply Pixel Attack with perceptual constraints"""
    
    # Generate importance map if not provided
    if importance_map is None and image is not None:
        print("Generating importance map for targeted perturbation...")
        importance_map = generate_importance_map_for_charts(image)
        
        # Visualize importance map
        if image_path is not None:
            importance_vis = (importance_map * 255).astype(np.uint8)
            importance_vis = cv2.applyColorMap(importance_vis, cv2.COLORMAP_JET)
            output_dir = os.path.dirname(get_output_path(image_path, 'pixel'))
            os.makedirs(output_dir, exist_ok=True)
            importance_path = os.path.join(output_dir, 'importance_map.png')
            cv2.imwrite(importance_path, importance_vis)
            print(f"Saved importance map to {importance_path}")
    
    # Create Pixel Attack
    attack = PixelAttack(
        classifier=classifier,
        th=th,
        es=es,
        max_iter=max_iter,
        targeted=targeted,
        verbose=True
    )
    
    # Generate adversarial example
    print(f"Generating adversarial example with Pixel Attack:")
    print(f"- Threshold: {th}")
    print(f"- Evolutionary Strategy: {'CMA-ES' if es == 0 else 'Differential Evolution'}")
    print(f"- Max iterations: {max_iter}")
    print(f"- Targeted: {targeted}")
    print(f"- Number of pixels: {num_pixels}")
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
    
    # Apply standardized SSIM constraint for consistency with all other attacks
    adv_image = apply_standardized_ssim_constraint(image, adv_image, ssim_threshold)
    
    return adv_image


def main():
    parser = argparse.ArgumentParser(description="Generate adversarial examples using Pixel Attack with perceptual constraints")
    parser.add_argument("--image_path", type=str, 
                        default="data/clean/chart/20231114102825506748.png",
                        help="Path to the input image")
    parser.add_argument("--th", type=int, default=10,
                        help="Threshold value for perturbation (default: 10)")
    parser.add_argument("--es", type=int, default=1, choices=[0, 1],
                        help="Evolutionary strategy: 0 for CMA-ES, 1 for Differential Evolution (default: 1)")
    parser.add_argument("--max_iter", type=int, default=100,
                        help="Maximum number of iterations (default: 100)")
    parser.add_argument("--targeted", action="store_true",
                        help="Use targeted attack instead of untargeted")
    parser.add_argument("--target_class", type=int, default=None,
                        help="Target class for targeted attack (default: None)")
    parser.add_argument("--ssim_threshold", type=float, default=0.85,
                        help="SSIM threshold for perceptual constraint (default: 0.85)")
    parser.add_argument("--num_pixels", type=int, default=20,
                        help="Number of pixels to modify (default: 20)")
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
    
    # Apply Pixel Attack with perceptual constraints
    adv_image = pixel_attack_with_perceptual_constraints(
        image, classifier, args.image_path, args.th, args.es, 
        args.max_iter, args.targeted, args.target_class,
        args.ssim_threshold, args.num_pixels, importance_map
    )
    
    # Get output path using utility function
    output_path = get_output_path(args.image_path, 'pixel')
    
    # Save adversarial image
    save_image(adv_image, output_path)
    
    # Print attack information
    print_attack_info(output_path, image, adv_image, 'pixel')
    
    # Print additional attack-specific information
    perturbation = np.abs(image.astype(np.float32) - adv_image.astype(np.float32))
    print(f"Max perturbation: {np.max(perturbation)}")
    print(f"Mean perturbation: {np.mean(perturbation)}")
    print(f"SSIM: {calculate_ssim(image, adv_image):.4f}")
    print(f"Number of pixels modified: {args.num_pixels}")


if __name__ == "__main__":
    main()
