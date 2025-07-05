#!/usr/bin/env python3
"""
Targeted ElasticNet Attack Script for Vision-Language Models (Replacement for L-BFGS)

This script applies an ElasticNet adversarial attack to images
to test the robustness of vision-language models. The ElasticNet attack is similar to L-BFGS
in that it's an optimization-based attack, but uses a different optimization approach.

This implementation focuses on:
1. Targeting semantically important regions of the image (text, chart elements, data points)
2. Keeping perturbations small enough to be relatively imperceptible to humans
3. Making perturbations effective enough to impact model performance

Usage:
    python v7_lbfgs_attack.py [--image_path PATH] [--max_iter ITERATIONS] [--confidence CONF]
                             [--targeted_regions] [--perceptual_constraint] [--ssim_threshold THRESHOLD]

Example:
    python v7_lbfgs_attack.py --image_path data/test_extracted/chart/20231114102825506748.png --max_iter 100 --confidence 0.1 --targeted_regions --perceptual_constraint
"""

import os
import cv2
import numpy as np
import argparse
import torch
from art.attacks.evasion import ElasticNet

# Import utility functions
from v0_attack_utils import (
    load_image, create_classifier, save_image, 
    get_output_path, print_attack_info, preprocess_image_for_attack,
    calculate_ssim, generate_saliency_map, create_combined_importance_map,
    apply_targeted_perturbation
)


def elastic_net_attack_targeted(image, classifier, image_path, max_iter=100, confidence=0.1, decision_rule='L1',
                               targeted_regions=True, perceptual_constraint=True, ssim_threshold=0.85):
    """Apply targeted ElasticNet attack focusing on semantically important regions"""
    # Preprocess image using utility function
    img_tensor = preprocess_image_for_attack(image)
    
    # Get the original prediction
    original_pred = np.argmax(classifier.predict(img_tensor)[0])
    
    # Target is any class other than the original (for untargeted attack)
    target = (original_pred + 1) % classifier.nb_classes
    
    # Create ElasticNet attack
    attack = ElasticNet(
        classifier=classifier,
        confidence=confidence,
        targeted=True,
        max_iter=max_iter,
        beta=0.01,  # Trade-off between L1 and L2 norms
        decision_rule=decision_rule,
        verbose=True
    )
    
    # Generate adversarial example
    print(f"Generating adversarial example with target_class={target}, confidence={confidence}, max_iter={max_iter}")
    print(f"Original class: {original_pred}, Target class: {target}")
    
    # Create one-hot encoded target
    target_one_hot = np.zeros((1, classifier.nb_classes))
    target_one_hot[0, target] = 1
    
    adv_image = attack.generate(x=img_tensor, y=target_one_hot)
    
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
        output_path = get_output_path(image_path, 'lbfgs')
        importance_path = os.path.join(os.path.dirname(output_path), 'importance_map.png')
        cv2.imwrite(importance_path, importance_vis)
        print(f"Saved importance map to {importance_path}")
        
        # Apply targeted perturbation
        adv_image = apply_targeted_perturbation(image, adv_image, importance_map)
    
    # Apply perceptual constraint if enabled
    if perceptual_constraint:
        current_ssim = calculate_ssim(image, adv_image)
        print(f"Initial SSIM: {current_ssim:.4f}")
        
        # If SSIM is below threshold, use a much stronger blending with original image
        if current_ssim < ssim_threshold:
            print(f"SSIM below threshold ({ssim_threshold}), applying stronger perceptual constraint...")
            
            # Use a very small alpha to ensure high SSIM
            alpha = 0.05  # Only 5% of the adversarial perturbation
            adv_image = cv2.addWeighted(image, 1 - alpha, adv_image, alpha, 0)
            final_ssim = calculate_ssim(image, adv_image)
            print(f"Final SSIM after perceptual constraint: {final_ssim:.4f}")
    
    return adv_image


def main():
    parser = argparse.ArgumentParser(description="Generate adversarial examples using targeted ElasticNet attack (replacement for L-BFGS)")
    parser.add_argument("--image_path", type=str, 
                        default="data/test_extracted/chart/20231114102825506748.png",
                        help="Path to the input image")
    parser.add_argument("--max_iter", type=int, default=100,
                        help="Maximum number of iterations (default: 100)")
    parser.add_argument("--confidence", type=float, default=0.1,
                        help="Confidence parameter for attack (higher values produce stronger attacks)")
    parser.add_argument("--decision_rule", type=str, default="L1",
                        choices=["L1", "L2"],
                        help="Decision rule for the attack (default: L1)")
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
    
    # Apply targeted ElasticNet attack
    adv_image = elastic_net_attack_targeted(
        image, classifier, args.image_path, args.max_iter, args.confidence, args.decision_rule,
        args.targeted_regions, args.perceptual_constraint, args.ssim_threshold
    )
    
    # Get output path using utility function
    output_path = get_output_path(args.image_path, 'lbfgs')
    
    # Save adversarial image
    save_image(adv_image, output_path)
    
    # Print attack information
    print_attack_info(output_path, image, adv_image, 'lbfgs')
    
    # Print additional attack-specific information
    perturbation = np.abs(image.astype(np.float32) - adv_image.astype(np.float32))
    print(f"Max perturbation: {np.max(perturbation)}")
    print(f"Mean perturbation: {np.mean(perturbation)}")
    print(f"SSIM: {calculate_ssim(image, adv_image):.4f}")
    print(f"L2 norm of perturbation: {np.linalg.norm(perturbation)}")


if __name__ == "__main__":
    main()
