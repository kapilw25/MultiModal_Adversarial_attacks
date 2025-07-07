#!/usr/bin/env python3
"""
Geometric Decision-based Attack (GeoDA) Script for Vision-Language Models

This script applies the Geometric Decision-based Attack (GeoDA) with perceptual constraints to create
adversarial examples that are both effective and visually imperceptible to humans.
GeoDA is an efficient black-box attack that uses geometric principles and subspace optimization
to find adversarial examples with minimal perturbations.

Usage:
    python v18_geoda_attack.py [--image_path PATH] [--batch_size BATCH_SIZE]
                              [--norm NORM] [--sub_dim SUB_DIM] [--max_iter MAX_ITER]
                              [--bin_search_tol BIN_SEARCH_TOL] [--lambda_param LAMBDA_PARAM]
                              [--sigma SIGMA] [--ssim_threshold THRESHOLD]

Example:
    source venv_MM/bin/activate && python attack_models/true_black_box_attacks/v18_geoda_attack.py --image_path data/test_extracted/chart/20231114102825506748.png --norm 2 --sub_dim 10 --max_iter 1000 --ssim_threshold 0.85
"""

import os
import cv2
import numpy as np
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
from art.attacks.evasion import GeoDA
from art.estimators.classification import PyTorchClassifier
from tqdm import tqdm

# Import utility functions
from v0_attack_utils import (
    load_image, create_classifier, save_image, 
    get_output_path, print_attack_info, preprocess_image_for_attack,
    calculate_ssim, generate_importance_map_for_charts,
    apply_threshold_optimized_constraints
)


def geoda_attack_with_perceptual_constraints(image, classifier, image_path=None, batch_size=64,
                                           norm=2, sub_dim=10, max_iter=1000, bin_search_tol=0.1,
                                           lambda_param=0.6, sigma=0.0002, ssim_threshold=0.85,
                                           importance_map=None):
    """Apply Geometric Decision-based Attack (GeoDA) with perceptual constraints"""
    
    # Generate importance map if not provided
    if importance_map is None and image is not None:
        print("Generating importance map for targeted perturbation...")
        importance_map = generate_importance_map_for_charts(image)
        
        # Visualize importance map
        if image_path is not None:
            importance_vis = (importance_map * 255).astype(np.uint8)
            importance_vis = cv2.applyColorMap(importance_vis, cv2.COLORMAP_JET)
            output_dir = os.path.dirname(get_output_path(image_path, 'geoda'))
            os.makedirs(output_dir, exist_ok=True)
            importance_path = os.path.join(output_dir, 'importance_map.png')
            cv2.imwrite(importance_path, importance_vis)
            print(f"Saved importance map to {importance_path}")
    
    # Create GeoDA Attack
    attack = GeoDA(
        estimator=classifier,
        batch_size=batch_size,
        norm=norm,
        sub_dim=sub_dim,
        max_iter=max_iter,
        bin_search_tol=bin_search_tol,
        lambda_param=lambda_param,
        sigma=sigma,
        verbose=True
    )
    
    # Preprocess image using utility function
    img_tensor = preprocess_image_for_attack(image)
    
    # Generate adversarial example
    print(f"Generating adversarial example with GeoDA Attack:")
    print(f"- Batch size: {batch_size}")
    print(f"- Norm: {norm}")
    print(f"- Subspace dimension: {sub_dim}")
    print(f"- Max iterations: {max_iter}")
    print(f"- Binary search tolerance: {bin_search_tol}")
    print(f"- Lambda parameter: {lambda_param}")
    print(f"- Sigma: {sigma}")
    print(f"- SSIM threshold: {ssim_threshold}")
    
    # For untargeted attack
    adv_image = attack.generate(x=img_tensor)
    
    # Convert back to uint8 format
    adv_image = adv_image[0].transpose(1, 2, 0)
    adv_image = np.clip(adv_image, 0, 1) * 255
    adv_image = adv_image.astype(np.uint8)
    
    # Resize back to original dimensions
    adv_image = cv2.resize(adv_image, (image.shape[1], image.shape[0]))
    
    # Apply perceptual constraint
    current_ssim = calculate_ssim(image, adv_image)
    print(f"Initial SSIM: {current_ssim:.4f}")
    
    # Apply threshold-optimized constraints to get SSIM close to the threshold
    if current_ssim < ssim_threshold or abs(current_ssim - ssim_threshold) < 0.05:
        print(f"Applying perceptual constraints to achieve target SSIM of {ssim_threshold:.4f}...")
        adv_image = apply_threshold_optimized_constraints(
            image, adv_image, ssim_threshold=ssim_threshold
        )
        final_ssim = calculate_ssim(image, adv_image)
        print(f"Final SSIM: {final_ssim:.4f} (target: {ssim_threshold:.4f})")
    
    # Apply targeted perturbation using importance map
    if importance_map is not None:
        print("Applying targeted perturbation using importance map...")
        
        # Calculate perturbation
        perturbation = adv_image.astype(np.float32) - image.astype(np.float32)
        
        # Apply importance map to perturbation (amplify important regions)
        importance_map_3ch = np.stack([importance_map] * 3, axis=2)
        weighted_perturbation = perturbation * (1 + (importance_map_3ch * 2.0))
        
        # Apply weighted perturbation to original image
        targeted_adv_image = image.astype(np.float32) + weighted_perturbation
        targeted_adv_image = np.clip(targeted_adv_image, 0, 255).astype(np.uint8)
        
        # Check if targeted image still meets SSIM constraint
        targeted_ssim = calculate_ssim(image, targeted_adv_image)
        if targeted_ssim >= ssim_threshold:
            print(f"Using targeted perturbation (SSIM: {targeted_ssim:.4f})")
            adv_image = targeted_adv_image
        else:
            print(f"Targeted perturbation violates SSIM constraint (SSIM: {targeted_ssim:.4f}), using original adversarial image")
    
    return adv_image


def main():
    parser = argparse.ArgumentParser(description="Generate adversarial examples using Geometric Decision-based Attack (GeoDA) with perceptual constraints")
    parser.add_argument("--image_path", type=str, 
                        default="data/test_extracted/chart/20231114102825506748.png",
                        help="Path to the input image")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for model inference (default: 64)")
    parser.add_argument("--norm", type=str, default="2", choices=["1", "2", "inf"],
                        help="Norm for the adversarial perturbation (default: 2)")
    parser.add_argument("--sub_dim", type=int, default=10,
                        help="Dimensionality of 2D frequency space (DCT) (default: 10)")
    parser.add_argument("--max_iter", type=int, default=1000,
                        help="Maximum number of iterations (default: 1000)")
    parser.add_argument("--bin_search_tol", type=float, default=0.1,
                        help="Maximum remaining L2 perturbation defining binary search convergence (default: 0.1)")
    parser.add_argument("--lambda_param", type=float, default=0.6,
                        help="Lambda parameter for iteration distribution (default: 0.6)")
    parser.add_argument("--sigma", type=float, default=0.0002,
                        help="Variance of the Gaussian perturbation (default: 0.0002)")
    parser.add_argument("--ssim_threshold", type=float, default=0.85,
                        help="SSIM threshold for perceptual constraint (default: 0.85)")
    args = parser.parse_args()
    
    # Convert norm string to appropriate format
    if args.norm == "inf":
        norm = np.inf
    else:
        norm = int(args.norm)
    
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
    
    # Apply GeoDA Attack with perceptual constraints
    adv_image = geoda_attack_with_perceptual_constraints(
        image, classifier, args.image_path, args.batch_size, norm, args.sub_dim,
        args.max_iter, args.bin_search_tol, args.lambda_param, args.sigma,
        args.ssim_threshold, importance_map
    )
    
    # Get output path using utility function
    output_path = get_output_path(args.image_path, 'geoda')
    
    # Save adversarial image
    save_image(adv_image, output_path)
    
    # Print attack information
    print_attack_info(output_path, image, adv_image, 'geoda')
    
    # Print additional attack-specific information
    perturbation = np.abs(image.astype(np.float32) - adv_image.astype(np.float32))
    print(f"Max perturbation: {np.max(perturbation)}")
    print(f"Mean perturbation: {np.mean(perturbation)}")
    print(f"SSIM: {calculate_ssim(image, adv_image):.4f}")
    
    # Estimate total queries used
    estimated_queries = args.max_iter
    print(f"Estimated total queries: approximately {estimated_queries}")


if __name__ == "__main__":
    main()
