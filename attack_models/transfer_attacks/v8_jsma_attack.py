#!/usr/bin/env python3
"""
Imperceptible Jacobian-based Saliency Map Attack (JSMA) Script for Vision-Language Models

This script applies a modified JSMA adversarial attack to images
to test the robustness of vision-language models. The JSMA attack is described in the paper
"The Limitations of Deep Learning in Adversarial Settings" by Papernot et al.

This implementation focuses on:
1. Creating truly imperceptible perturbations by limiting both the number and magnitude of pixel changes
2. Targeting semantically important regions of the image (text, chart elements, data points)
3. Using perceptual constraints to ensure the attack remains invisible to human observers
4. Maintaining effectiveness against vision-language models

Usage:
    python v8_jsma_attack.py [--image_path PATH] [--max_iter ITERATIONS] [--theta THETA]
                            [--targeted_regions] [--perceptual_constraint] [--ssim_threshold THRESHOLD]
                            [--max_pixel_change MAX_CHANGE]

Example:
    python v8_jsma_attack.py --image_path data/test_extracted/chart/20231114102825506748.png --max_iter 20 --theta 0.1 --max_pixel_change 10 --targeted_regions --perceptual_constraint
"""

import os
import cv2
import numpy as np
import argparse
import torch
from art.attacks.evasion import SaliencyMapMethod

# Import utility functions
from v0_attack_utils import (
    load_image, create_classifier, save_image, 
    get_output_path, print_attack_info, preprocess_image_for_attack,
    calculate_ssim, generate_saliency_map, create_combined_importance_map,
    apply_targeted_perturbation
)


def jsma_attack_imperceptible(image, classifier, image_path, theta=0.1, gamma=0.1, max_iter=20,
                             max_pixel_change=10, targeted_regions=True, perceptual_constraint=True, 
                             ssim_threshold=0.95):
    """Apply imperceptible JSMA attack focusing on semantically important regions"""
    # Preprocess image using utility function
    img_tensor = preprocess_image_for_attack(image)
    
    # Get the original prediction
    original_pred = np.argmax(classifier.predict(img_tensor)[0])
    
    # Target is any class other than the original (for untargeted attack)
    target = (original_pred + 1) % classifier.nb_classes
    
    # Create JSMA attack with reduced theta (perturbation magnitude)
    attack = SaliencyMapMethod(
        classifier=classifier,
        theta=theta,  # Reduced perturbation magnitude
        gamma=gamma,
        batch_size=1,
        verbose=True
    )
    
    # Generate adversarial example
    print(f"Generating adversarial example with target_class={target}, max_iter={max_iter}, theta={theta}")
    print(f"Original class: {original_pred}, Target class: {target}")
    
    # Create one-hot encoded target
    target_one_hot = np.zeros((1, classifier.nb_classes))
    target_one_hot[0, target] = 1
    
    adv_image = attack.generate(x=img_tensor, y=target_one_hot, max_iter=max_iter)
    
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
        output_path = get_output_path(image_path, 'jsma')
        importance_path = os.path.join(os.path.dirname(output_path), 'importance_map.png')
        cv2.imwrite(importance_path, importance_vis)
        print(f"Saved importance map to {importance_path}")
        
        # Apply targeted perturbation
        adv_image = apply_targeted_perturbation(image, adv_image, importance_map)
    
    # Apply additional constraint on maximum pixel change
    perturbation = adv_image.astype(np.float32) - image.astype(np.float32)
    # Clip perturbation to max_pixel_change
    clipped_perturbation = np.clip(perturbation, -max_pixel_change, max_pixel_change)
    adv_image = np.clip(image.astype(np.float32) + clipped_perturbation, 0, 255).astype(np.uint8)
    
    # Calculate initial SSIM
    current_ssim = calculate_ssim(image, adv_image)
    print(f"Initial SSIM after clipping: {current_ssim:.4f}")
    
    # Apply perceptual constraint if enabled
    if perceptual_constraint:
        # If SSIM is below threshold, blend with original image to improve perceptual quality
        if current_ssim < ssim_threshold:
            print(f"SSIM below threshold ({ssim_threshold}), applying perceptual constraint...")
            
            # Find optimal alpha that maintains effectiveness while ensuring imperceptibility
            # Start with a small alpha and gradually increase until we find a good balance
            best_alpha = 0.05  # Start with a very small perturbation
            best_ssim = 0.0
            best_adv_image = image.copy()
            
            # Try different alpha values to find the best balance
            for alpha_percent in range(5, 31, 5):  # Try 5%, 10%, 15%, 20%, 25%, 30%
                alpha = alpha_percent / 100.0
                blended_image = cv2.addWeighted(image, 1 - alpha, adv_image, alpha, 0)
                blend_ssim = calculate_ssim(image, blended_image)
                
                # If this alpha gives us a good SSIM, update our best
                if blend_ssim >= ssim_threshold and alpha > best_alpha:
                    best_alpha = alpha
                    best_ssim = blend_ssim
                    best_adv_image = blended_image.copy()
            
            # If we found a good alpha, use it
            if best_ssim >= ssim_threshold:
                adv_image = best_adv_image
                print(f"Found optimal alpha: {best_alpha:.4f} with SSIM: {best_ssim:.4f}")
            else:
                # If we couldn't find a good alpha, use a very small one to ensure imperceptibility
                alpha = 0.05  # Only 5% of the adversarial perturbation
                adv_image = cv2.addWeighted(image, 1 - alpha, adv_image, alpha, 0)
                final_ssim = calculate_ssim(image, adv_image)
                print(f"Using minimal alpha: {alpha:.4f} with SSIM: {final_ssim:.4f}")
    
    # Calculate final perturbation metrics
    perturbation = np.abs(image.astype(np.float32) - adv_image.astype(np.float32))
    print(f"Final perturbation - Max: {np.max(perturbation):.2f}, Mean: {np.mean(perturbation):.4f}")
    changed_pixels = np.sum(perturbation > 0.5)  # Count pixels with change > 0.5
    print(f"Number of changed pixels: {changed_pixels} ({changed_pixels/(image.shape[0]*image.shape[1]*image.shape[2])*100:.2f}%)")
    
    return adv_image


def main():
    parser = argparse.ArgumentParser(description="Generate imperceptible adversarial examples using JSMA attack")
    parser.add_argument("--image_path", type=str, 
                        default="data/test_extracted/chart/20231114102825506748.png",
                        help="Path to the input image")
    parser.add_argument("--max_iter", type=int, default=20,
                        help="Maximum number of iterations (default: 20)")
    parser.add_argument("--theta", type=float, default=0.1,
                        help="Maximum percentage of perturbed features (default: 0.1)")
    parser.add_argument("--gamma", type=float, default=0.1,
                        help="Step size (default: 0.1)")
    parser.add_argument("--max_pixel_change", type=int, default=10,
                        help="Maximum allowed change per pixel (default: 10)")
    parser.add_argument("--targeted_regions", action="store_true",
                        help="Apply perturbation only to important regions")
    parser.add_argument("--perceptual_constraint", action="store_true",
                        help="Apply perceptual similarity constraint")
    parser.add_argument("--ssim_threshold", type=float, default=0.98,
                        help="SSIM threshold for perceptual constraint (default: 0.98)")
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
    
    # Apply imperceptible JSMA attack
    adv_image = jsma_attack_imperceptible(
        image, classifier, args.image_path, args.theta, args.gamma, args.max_iter,
        args.max_pixel_change, args.targeted_regions, args.perceptual_constraint, args.ssim_threshold
    )
    
    # Get output path using utility function
    output_path = get_output_path(args.image_path, 'jsma')
    
    # Save adversarial image
    save_image(adv_image, output_path)
    
    # Print attack information
    print_attack_info(output_path, image, adv_image, 'jsma')
    
    # Print additional JSMA-specific information
    perturbation = np.abs(image.astype(np.float32) - adv_image.astype(np.float32))
    print(f"Max perturbation: {np.max(perturbation)}")
    print(f"Mean perturbation: {np.mean(perturbation)}")
    print(f"SSIM: {calculate_ssim(image, adv_image):.4f}")
    changed_pixels = np.sum(np.any(perturbation > 0.5, axis=2))
    print(f"Total modified pixels: {changed_pixels}")


if __name__ == "__main__":
    main()
