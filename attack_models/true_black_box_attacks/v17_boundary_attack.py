#!/usr/bin/env python3
"""
Boundary Attack Script for Vision-Language Models

This script applies the Boundary Attack with perceptual constraints to create
adversarial examples that are both effective and visually imperceptible to humans.
Boundary Attack is a powerful black-box attack that only requires access to the
model's final decision (class label), not probabilities or gradients.

Usage:
    python v17_boundary_attack.py [--image_path PATH] [--targeted] [--target_class CLASS]
                                 [--delta DELTA] [--epsilon EPSILON] [--step_adapt STEP_ADAPT]
                                 [--max_iter MAX_ITER] [--num_trial NUM_TRIAL] [--sample_size SAMPLE_SIZE]
                                 [--init_size INIT_SIZE] [--min_epsilon MIN_EPSILON]
                                 [--ssim_threshold THRESHOLD]

Example:
    source venv_MM/bin/activate && python attack_models/true_black_box_attacks/v17_boundary_attack.py --image_path data/test_extracted/chart/20231114102825506748.png --delta 0.1 --epsilon 0.1 --max_iter 1000 --ssim_threshold 0.85
"""

import os
import cv2
import numpy as np
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
from art.attacks.evasion import BoundaryAttack
from art.estimators.classification import PyTorchClassifier
from tqdm import tqdm

# Import utility functions
from v0_attack_utils import (
    load_image, create_classifier, save_image, 
    get_output_path, print_attack_info, preprocess_image_for_attack,
    calculate_ssim, generate_importance_map_for_charts,
    apply_threshold_optimized_constraints
)


def boundary_attack_with_perceptual_constraints(image, classifier, image_path=None, targeted=False, 
                                              target_class=None, delta=0.1, epsilon=0.1, 
                                              step_adapt=0.667, max_iter=1000, num_trial=25,
                                              sample_size=20, init_size=100, min_epsilon=0.0,
                                              ssim_threshold=0.85, importance_map=None):
    """Apply Boundary Attack with perceptual constraints"""
    
    # Generate importance map if not provided
    if importance_map is None and image is not None:
        print("Generating importance map for targeted perturbation...")
        importance_map = generate_importance_map_for_charts(image)
        
        # Visualize importance map
        if image_path is not None:
            importance_vis = (importance_map * 255).astype(np.uint8)
            importance_vis = cv2.applyColorMap(importance_vis, cv2.COLORMAP_JET)
            output_dir = os.path.dirname(get_output_path(image_path, 'boundary'))
            os.makedirs(output_dir, exist_ok=True)
            importance_path = os.path.join(output_dir, 'importance_map.png')
            cv2.imwrite(importance_path, importance_vis)
            print(f"Saved importance map to {importance_path}")
    
    # Create Boundary Attack
    attack = BoundaryAttack(
        estimator=classifier,
        targeted=targeted,
        delta=delta,
        epsilon=epsilon,
        step_adapt=step_adapt,
        max_iter=max_iter,
        num_trial=num_trial,
        sample_size=sample_size,
        init_size=init_size,
        min_epsilon=min_epsilon,
        batch_size=1,
        verbose=True
    )
    
    # Preprocess image using utility function
    img_tensor = preprocess_image_for_attack(image)
    
    # Generate adversarial example
    print(f"Generating adversarial example with Boundary Attack:")
    print(f"- Targeted: {targeted}")
    print(f"- Delta: {delta}")
    print(f"- Epsilon: {epsilon}")
    print(f"- Step adapt: {step_adapt}")
    print(f"- Max iterations: {max_iter}")
    print(f"- Number of trials: {num_trial}")
    print(f"- Sample size: {sample_size}")
    print(f"- Init size: {init_size}")
    print(f"- Min epsilon: {min_epsilon}")
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
    parser = argparse.ArgumentParser(description="Generate adversarial examples using Boundary Attack with perceptual constraints")
    parser.add_argument("--image_path", type=str, 
                        default="data/test_extracted/chart/20231114102825506748.png",
                        help="Path to the input image")
    parser.add_argument("--targeted", action="store_true",
                        help="Use targeted attack instead of untargeted")
    parser.add_argument("--target_class", type=int, default=None,
                        help="Target class for targeted attack (default: None)")
    parser.add_argument("--delta", type=float, default=0.1,
                        help="Initial step size for the orthogonal step (default: 0.1)")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Initial step size for the step towards the target (default: 0.1)")
    parser.add_argument("--step_adapt", type=float, default=0.667,
                        help="Factor by which the step sizes are multiplied or divided (default: 0.667)")
    parser.add_argument("--max_iter", type=int, default=1000,
                        help="Maximum number of iterations (default: 1000)")
    parser.add_argument("--num_trial", type=int, default=25,
                        help="Maximum number of trials per iteration (default: 25)")
    parser.add_argument("--sample_size", type=int, default=20,
                        help="Number of samples per trial (default: 20)")
    parser.add_argument("--init_size", type=int, default=100,
                        help="Maximum number of trials for initial generation (default: 100)")
    parser.add_argument("--min_epsilon", type=float, default=0.0,
                        help="Stop attack if perturbation is smaller than this value (default: 0.0)")
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
    
    # Apply Boundary Attack with perceptual constraints
    adv_image = boundary_attack_with_perceptual_constraints(
        image, classifier, args.image_path, args.targeted, args.target_class,
        args.delta, args.epsilon, args.step_adapt, args.max_iter, args.num_trial,
        args.sample_size, args.init_size, args.min_epsilon, args.ssim_threshold,
        importance_map
    )
    
    # Get output path using utility function
    output_path = get_output_path(args.image_path, 'boundary')
    
    # Save adversarial image
    save_image(adv_image, output_path)
    
    # Print attack information
    print_attack_info(output_path, image, adv_image, 'boundary')
    
    # Print additional attack-specific information
    perturbation = np.abs(image.astype(np.float32) - adv_image.astype(np.float32))
    print(f"Max perturbation: {np.max(perturbation)}")
    print(f"Mean perturbation: {np.mean(perturbation)}")
    print(f"SSIM: {calculate_ssim(image, adv_image):.4f}")
    
    # Estimate total queries used
    estimated_queries = args.max_iter * args.num_trial * args.sample_size
    print(f"Estimated total queries: approximately {estimated_queries}")


if __name__ == "__main__":
    main()
