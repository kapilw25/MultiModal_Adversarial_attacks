#!/usr/bin/env python3
"""
Targeted FGSM Attack Script for Vision-Language Models

This script applies a Fast Gradient Sign Method (FGSM) adversarial attack to images
to test the robustness of vision-language models. FGSM was introduced by Goodfellow et al.
in the paper "Explaining and Harnessing Adversarial Examples" (2014).

The FGSM attack is defined mathematically as:
- For untargeted attacks: x' = x + ε · sign(∇ₓJ(θ, x, y))
- For targeted attacks:   x' = x - ε · sign(∇ₓJ(θ, x, t))

This implementation focuses on:
1. Targeting semantically important regions of the image (text, chart elements, data points)
2. Keeping perturbations small enough to be relatively imperceptible to humans
3. Making perturbations effective enough to impact model performance

Usage:
    python v3_fgsm_attack.py [--image_path PATH] [--eps EPSILON] [--targeted] [--target_class CLASS]
                            [--targeted_regions] [--perceptual_constraint] [--ssim_threshold THRESHOLD]

Example:
    python v3_fgsm_attack.py --image_path data/test_extracted/chart/20231114102825506748.png --eps 0.03 --targeted_regions --perceptual_constraint
"""

import os
import cv2
import numpy as np
import argparse
import torch
from art.attacks.evasion import FastGradientMethod

# Import utility functions
from v0_attack_utils import (
    load_image, create_classifier, save_image, 
    get_output_path, print_attack_info, preprocess_image_for_attack,
    calculate_ssim, generate_saliency_map, create_combined_importance_map,
    apply_targeted_perturbation
)


def fgsm_attack_targeted(image, classifier, image_path, eps=0.03, targeted=False, target_class=None,
                         targeted_regions=True, perceptual_constraint=True, ssim_threshold=0.85):
    """Apply targeted FGSM attack focusing on semantically important regions"""
    # Create FGSM attack
    attack = FastGradientMethod(
        estimator=classifier,
        norm=np.inf,
        eps=eps,
        targeted=targeted,
        batch_size=1,
        minimal=False
    )
    
    # Preprocess image using utility function
    img_tensor = preprocess_image_for_attack(image)
    
    # Generate adversarial example
    print(f"Generating adversarial example with eps={eps}, targeted={targeted}")
    
    if targeted and target_class is not None:
        # For targeted attack, create one-hot encoded target
        target = np.zeros((1, classifier.nb_classes))
        target[0, target_class] = 1
        adv_image = attack.generate(x=img_tensor, y=target)
    else:
        adv_image = attack.generate(x=img_tensor)
    
    # Convert back to uint8 format
    adv_image = adv_image[0].transpose(1, 2, 0)
    adv_image = np.clip(adv_image, 0, 1) * 255
    adv_image = adv_image.astype(np.uint8)
    
    # Resize back to original dimensions
    adv_image = cv2.resize(adv_image, (image.shape[1], image.shape[0]))
    
    # Apply targeted region attack if enabled
    if targeted_regions:
        print("Generating importance map for targeted perturbation...")
        importance_mask, importance_map = create_combined_importance_map(image, classifier)
        
        # Save importance map for visualization
        importance_vis = (importance_map * 255).astype(np.uint8)
        importance_vis = cv2.applyColorMap(importance_vis, cv2.COLORMAP_JET)
        output_path = get_output_path(image_path, 'fgsm')
        importance_path = os.path.join(os.path.dirname(output_path), 'importance_map.png')
        cv2.imwrite(importance_path, importance_vis)
        print(f"Saved importance map to {importance_path}")
        
        # Apply targeted perturbation
        adv_image = apply_targeted_perturbation(image, adv_image, importance_map)
    
    # Apply perceptual constraint if enabled
    if perceptual_constraint:
        current_ssim = calculate_ssim(image, adv_image)
        print(f"Initial SSIM: {current_ssim:.4f}")
        
        # If SSIM is below threshold, blend with original image to improve perceptual quality
        if current_ssim < ssim_threshold:
            print(f"SSIM below threshold ({ssim_threshold}), applying perceptual constraint...")
            
            # Binary search to find optimal blending factor
            alpha_min, alpha_max = 0.0, 1.0
            best_adv_image = adv_image.copy()
            best_ssim = current_ssim
            
            for _ in range(10):  # 10 binary search steps
                alpha = (alpha_min + alpha_max) / 2
                blended_image = cv2.addWeighted(image, 1 - alpha, adv_image, alpha, 0)
                blend_ssim = calculate_ssim(image, blended_image)
                
                if blend_ssim >= ssim_threshold:
                    best_adv_image = blended_image
                    best_ssim = blend_ssim
                    alpha_max = alpha
                else:
                    alpha_min = alpha
            
            # Use a slightly stronger perturbation to ensure it's effective
            # but still maintains reasonable visual quality
            alpha = min(alpha_max + 0.1, 0.9)  # Add a margin to ensure perturbation is effective
            adv_image = cv2.addWeighted(image, 1 - alpha, adv_image, alpha, 0)
            final_ssim = calculate_ssim(image, adv_image)
            print(f"Final SSIM after perceptual constraint: {final_ssim:.4f}")
    
    return adv_image


def main():
    parser = argparse.ArgumentParser(description="Generate adversarial examples using targeted FGSM attack")
    parser.add_argument("--image_path", type=str, 
                        default="data/test_extracted/chart/20231114102825506748.png",
                        help="Path to the input image")
    parser.add_argument("--eps", type=float, default=0.03,
                        help="Maximum perturbation (default: 0.03)")
    parser.add_argument("--targeted", action="store_true",
                        help="Use targeted attack instead of untargeted")
    parser.add_argument("--target_class", type=int, default=None,
                        help="Target class for targeted attack (default: None)")
    parser.add_argument("--targeted_regions", action="store_true",
                        help="Apply perturbation only to important regions")
    parser.add_argument("--perceptual_constraint", action="store_true",
                        help="Apply perceptual similarity constraint")
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
    
    # Apply targeted FGSM attack
    adv_image = fgsm_attack_targeted(
        image, classifier, args.image_path, args.eps, args.targeted, args.target_class,
        args.targeted_regions, args.perceptual_constraint, args.ssim_threshold
    )
    
    # Get output path using utility function
    output_path = get_output_path(args.image_path, 'fgsm')
    
    # Save adversarial image
    save_image(adv_image, output_path)
    
    # Print attack information
    print_attack_info(output_path, image, adv_image, 'fgsm')
    
    # Print additional FGSM-specific information
    perturbation = np.abs(image.astype(np.float32) - adv_image.astype(np.float32))
    print(f"Max perturbation: {np.max(perturbation)}")
    print(f"Mean perturbation: {np.mean(perturbation)}")
    print(f"SSIM: {calculate_ssim(image, adv_image):.4f}")


if __name__ == "__main__":
    main()
