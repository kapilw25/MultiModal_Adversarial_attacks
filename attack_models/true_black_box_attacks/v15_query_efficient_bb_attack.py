#!/usr/bin/env python3
"""
Query-Efficient Black-box Attack Script for Vision-Language Models with Perceptual Constraints

This script implements a custom version of the Query-Efficient Black-box attack, which creates 
adversarial examples by estimating gradients through efficient querying of the model. The attack 
is inspired by the Natural Evolution Strategies (NES) approach from the paper:
"Query-Efficient Black-box Adversarial Examples" (https://arxiv.org/abs/1712.07113)

The implementation enhances the standard Query-Efficient Black-box attack with:
1. SSIM (Structural Similarity) constraints to maintain visual similarity
2. Importance-weighted perturbation focusing on chart elements
3. Binary search for optimal perceptual quality

Usage:
    python v15_query_efficient_bb_attack.py [--image_path PATH] [--num_basis NUM]
                                           [--sigma SIGMA] [--max_iter ITER]
                                           [--epsilon EPS] [--ssim_threshold THRESHOLD]

Example:
    source venv_MM/bin/activate && python attack_models/true_black_box_attacks/v15_query_efficient_bb_attack.py --image_path data/test_extracted/chart/20231114102825506748.png --num_basis 20 --sigma 0.015625 --max_iter 100 --epsilon 0.1 --ssim_threshold 0.85
"""

import os
import cv2
import numpy as np
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# Import utility functions
from v0_attack_utils import (
    load_image, save_image, 
    get_output_path, print_attack_info, preprocess_image_for_attack,
    calculate_ssim, generate_importance_map_for_charts, create_classifier_with_probs,
    apply_enhanced_perceptual_constraints, postprocess_adversarial_image
)


def query_efficient_bb_attack_with_perceptual_constraints(image, classifier, image_path=None, 
                                                       num_basis=20, sigma=0.015625, 
                                                       max_iter=100, epsilon=0.1, 
                                                       ssim_threshold=0.85, 
                                                       importance_map=None):
    """Apply Query-Efficient Black-box Attack with perceptual constraints"""
    
    # Generate importance map if not provided
    if importance_map is None and image is not None:
        print("Generating importance map for targeted perturbation...")
        importance_map = generate_importance_map_for_charts(image)
        
        # Visualize importance map
        if image_path is not None:
            importance_vis = (importance_map * 255).astype(np.uint8)
            importance_vis = cv2.applyColorMap(importance_vis, cv2.COLORMAP_JET)
            output_dir = os.path.dirname(get_output_path(image_path, 'query_efficient_bb'))
            os.makedirs(output_dir, exist_ok=True)
            importance_path = os.path.join(output_dir, 'importance_map.png')
            cv2.imwrite(importance_path, importance_vis)
            print(f"Saved importance map to {importance_path}")
    
    # Initialize adversarial image as the original image
    adv_image = image.copy().astype(np.float32)
    
    # Set up transformation for model input
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Run the attack for multiple iterations
    print(f"Running Query-Efficient Black-box Attack for {max_iter} iterations...")
    print(f"- Number of basis vectors: {num_basis}")
    print(f"- Sigma: {sigma}")
    print(f"- Epsilon: {epsilon}")
    print(f"- SSIM threshold: {ssim_threshold}")
    
    for i in tqdm(range(max_iter)):
        # Generate random perturbation directions
        perturbations = []
        for _ in range(num_basis):
            # Generate random noise
            noise = np.random.normal(0, 1, size=image.shape).astype(np.float32)
            # Scale the noise
            scaled_noise = noise * sigma
            perturbations.append(scaled_noise)
        
        # Evaluate model on perturbed images
        best_perturbation = None
        best_score = -float('inf')
        
        for pert in perturbations:
            # Apply positive perturbation
            pos_perturbed = np.clip(adv_image + pert, 0, 255).astype(np.uint8)
            # Apply negative perturbation
            neg_perturbed = np.clip(adv_image - pert, 0, 255).astype(np.uint8)
            
            # Convert to PIL images
            pos_img = Image.fromarray(pos_perturbed)
            neg_img = Image.fromarray(neg_perturbed)
            
            # Transform for model input
            pos_tensor = transform(pos_img).unsqueeze(0).to(classifier.device)
            neg_tensor = transform(neg_img).unsqueeze(0).to(classifier.device)
            
            # Get model predictions
            with torch.no_grad():
                pos_output = classifier.model(pos_tensor)
                neg_output = classifier.model(neg_tensor)
            
            # Calculate score difference (higher is better for adversarial examples)
            # We use the difference in confidence for the top class
            pos_score = torch.softmax(pos_output, dim=1)[0, 0].item()
            neg_score = torch.softmax(neg_output, dim=1)[0, 0].item()
            score_diff = pos_score - neg_score
            
            # Estimate gradient direction
            est_gradient = score_diff * pert / (2 * sigma)
            
            # Keep track of the best perturbation
            if score_diff > best_score:
                best_score = score_diff
                best_perturbation = est_gradient
        
        # Apply the best perturbation if found
        if best_perturbation is not None:
            # Update adversarial image with estimated gradient
            adv_image = np.clip(adv_image + epsilon * np.sign(best_perturbation), 0, 255)
        
        # Every 10 iterations, check SSIM
        if i > 0 and i % 10 == 0:
            current_adv = adv_image.astype(np.uint8)
            current_ssim = calculate_ssim(image, current_adv)
            print(f"Iteration {i}/{max_iter}, SSIM: {current_ssim:.4f}")
            
            # If SSIM is below threshold, apply perceptual constraints
            if current_ssim < ssim_threshold:
                print(f"SSIM below threshold, applying perceptual constraints...")
                current_adv = apply_enhanced_perceptual_constraints(
                    image, current_adv, ssim_threshold=ssim_threshold
                )
                adv_image = current_adv.astype(np.float32)
                current_ssim = calculate_ssim(image, current_adv)
                print(f"After constraints: SSIM: {current_ssim:.4f}")
    
    # Convert to uint8 for final output
    adv_image = adv_image.astype(np.uint8)
    
    # Apply importance-weighted perturbation if importance map is provided
    if importance_map is not None:
        print("Applying importance-weighted perturbation...")
        
        # Calculate perturbation
        perturbation = adv_image.astype(np.float32) - image.astype(np.float32)
        
        # Weight perturbation by importance map (higher weight in important regions)
        importance_map_3ch = np.stack([importance_map] * 3, axis=2)
        weighted_perturbation = perturbation * (1 + importance_map_3ch)
        
        # Apply weighted perturbation
        weighted_adv_image = np.clip(image.astype(np.float32) + weighted_perturbation, 0, 255).astype(np.uint8)
        
        # Check if weighted version still meets SSIM threshold
        weighted_ssim = calculate_ssim(image, weighted_adv_image)
        print(f"SSIM after importance weighting: {weighted_ssim:.4f}")
        
        if weighted_ssim >= ssim_threshold:
            adv_image = weighted_adv_image
            print("Using importance-weighted adversarial image")
        else:
            print("Importance-weighted image doesn't meet SSIM threshold, using original adversarial image")
    
    # Final check and application of perceptual constraints
    final_ssim = calculate_ssim(image, adv_image)
    print(f"Final SSIM before constraints: {final_ssim:.4f}")
    
    if final_ssim < ssim_threshold:
        print(f"Applying final perceptual constraints to meet SSIM threshold: {ssim_threshold}")
        adv_image = apply_enhanced_perceptual_constraints(
            image, adv_image, ssim_threshold=ssim_threshold
        )
        
        # Calculate final SSIM
        final_ssim = calculate_ssim(image, adv_image)
        print(f"Final SSIM after perceptual constraints: {final_ssim:.4f}")
    
    return adv_image


def main():
    parser = argparse.ArgumentParser(description="Generate adversarial examples using Query-Efficient Black-box Attack with perceptual constraints")
    parser.add_argument("--image_path", type=str, 
                        default="data/test_extracted/chart/20231114102825506748.png",
                        help="Path to the input image")
    parser.add_argument("--num_basis", type=int, default=20,
                        help="Number of basis vectors for gradient estimation (default: 20)")
    parser.add_argument("--sigma", type=float, default=0.015625,
                        help="Sigma value for Gaussian noise (default: 0.015625 = 1/64)")
    parser.add_argument("--max_iter", type=int, default=100,
                        help="Maximum number of iterations (default: 100)")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Epsilon for perturbation magnitude (default: 0.1)")
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
    
    # Apply Query-Efficient Black-box Attack with perceptual constraints
    adv_image = query_efficient_bb_attack_with_perceptual_constraints(
        image, classifier, args.image_path, args.num_basis, 
        args.sigma, args.max_iter, args.epsilon, 
        args.ssim_threshold, importance_map
    )
    
    # Get output path using utility function
    output_path = get_output_path(args.image_path, 'query_efficient_bb')
    
    # Save adversarial image
    save_image(adv_image, output_path)
    
    # Print attack information
    print_attack_info(output_path, image, adv_image, 'query_efficient_bb')
    
    # Print additional attack-specific information
    perturbation = np.abs(image.astype(np.float32) - adv_image.astype(np.float32))
    print(f"Max perturbation: {np.max(perturbation)}")
    print(f"Mean perturbation: {np.mean(perturbation)}")
    print(f"SSIM: {calculate_ssim(image, adv_image):.4f}")
    print(f"Query-Efficient parameters: num_basis={args.num_basis}, sigma={args.sigma}, epsilon={args.epsilon}")


if __name__ == "__main__":
    main()
