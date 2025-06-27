#!/usr/bin/env python3
"""
Targeted Carlini-Wagner L2 Attack Script for Vision-Language Models

This script applies a Carlini-Wagner L2 adversarial attack to images
to test the robustness of vision-language models. The CW attack is described in the paper
"Towards Evaluating the Robustness of Neural Networks" by Carlini and Wagner.

This implementation focuses on:
1. Targeting semantically important regions of the image (text, chart elements, data points)
2. Keeping perturbations small enough to be relatively imperceptible to humans
3. Making perturbations effective enough to impact model performance

Usage:
    python v4_cw_l2_attack.py [--image_path PATH] [--confidence CONF] [--max_iter ITERATIONS] [--learning_rate LR]
                             [--targeted_regions] [--perceptual_constraint] [--ssim_threshold THRESHOLD]

Example:
    python v4_cw_l2_attack.py --image_path data/test_extracted/chart/20231114102825506748.png --confidence 5 --max_iter 100 --targeted_regions --perceptual_constraint
"""

import os
import cv2
import numpy as np
import argparse
import torch
from art.attacks.evasion import CarliniL2Method

# Import utility functions
from v0_attack_utils import (
    load_image, create_classifier, save_image, 
    get_output_path, print_attack_info, preprocess_image_for_attack,
    calculate_ssim, generate_saliency_map, create_combined_importance_map,
    apply_targeted_perturbation
)


def cw_l2_attack_targeted(image, classifier, image_path, confidence=5.0, max_iter=100, learning_rate=0.01,
                          targeted_regions=True, perceptual_constraint=True, ssim_threshold=0.85):
    """Apply targeted CW-L2 attack focusing on semantically important regions"""
    # Create CW-L2 attack
    attack = CarliniL2Method(
        classifier=classifier,
        confidence=confidence,
        max_iter=max_iter,
        learning_rate=learning_rate,
        binary_search_steps=10,
        initial_const=0.01,
        targeted=False,
        verbose=True
    )
    
    # Preprocess image using utility function
    img_tensor = preprocess_image_for_attack(image)
    
    # Generate adversarial example
    print(f"Generating adversarial example with confidence={confidence}, max_iter={max_iter}, learning_rate={learning_rate}")
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
        output_path = get_output_path(image_path, 'cw_l2')
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
    parser = argparse.ArgumentParser(description="Generate adversarial examples using targeted CW-L2 attack")
    parser.add_argument("--image_path", type=str, 
                        default="data/test_extracted/chart/20231114102825506748.png",
                        help="Path to the input image")
    parser.add_argument("--confidence", type=float, default=5.0,
                        help="Confidence parameter for attack (higher values produce stronger attacks)")
    parser.add_argument("--max_iter", type=int, default=100,
                        help="Maximum number of iterations (default: 100)")
    parser.add_argument("--learning_rate", type=float, default=0.01,
                        help="Learning rate for optimization (default: 0.01)")
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
    
    # Apply targeted CW-L2 attack
    adv_image = cw_l2_attack_targeted(
        image, classifier, args.image_path, args.confidence, args.max_iter, args.learning_rate,
        args.targeted_regions, args.perceptual_constraint, args.ssim_threshold
    )
    
    # Get output path using utility function
    output_path = get_output_path(args.image_path, 'cw_l2')
    
    # Save adversarial image
    save_image(adv_image, output_path)
    
    # Print attack information
    print_attack_info(output_path, image, adv_image, 'cw_l2')
    
    # Print additional CW-L2-specific information
    perturbation = np.abs(image.astype(np.float32) - adv_image.astype(np.float32))
    print(f"Max perturbation: {np.max(perturbation)}")
    print(f"Mean perturbation: {np.mean(perturbation)}")
    print(f"SSIM: {calculate_ssim(image, adv_image):.4f}")
    print(f"L2 norm of perturbation: {np.linalg.norm(perturbation)}")


if __name__ == "__main__":
    main()
