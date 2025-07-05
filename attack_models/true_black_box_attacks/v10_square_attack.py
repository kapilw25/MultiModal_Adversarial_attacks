#!/usr/bin/env python3
"""
Enhanced Perceptually-Constrained Square Attack Script for Vision-Language Models

This script applies the Square Attack with multiple perceptual constraints to create
adversarial examples that are both effective and visually imperceptible to humans.
It enhances the pure black-box Square Attack with:

1. SSIM (Structural Similarity) constraints
2. LPIPS (Learned Perceptual Image Patch Similarity) constraints
3. CLIP similarity metrics for semantic preservation
4. Adaptive multi-stage blending to maintain visual quality

Usage:
    python v10_square_attack.py [--image_path PATH] [--eps EPSILON] [--norm NORM]
                               [--max_iter ITERATIONS] [--p_init INITIAL_PROB]
                               [--targeted] [--target_class CLASS]
                               [--ssim_threshold THRESHOLD] [--lpips_threshold THRESHOLD]
                               [--clip_threshold THRESHOLD]

Example:
    python v10_square_attack.py --image_path data/test_extracted/chart/20231114102825506748.png --eps 0.05 --norm inf --max_iter 100 --ssim_threshold 0.95 --lpips_threshold 0.05 --clip_threshold 0.9
"""

import os
import cv2
import numpy as np
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
from art.attacks.evasion import SquareAttack
from art.estimators.classification import PyTorchClassifier
import lpips
import clip
from tqdm import tqdm

# Import utility functions
from v0_attack_utils import (
    load_image, create_classifier, save_image, 
    get_output_path, print_attack_info, preprocess_image_for_attack,
    calculate_ssim, calculate_lpips, calculate_clip_similarity,
    apply_enhanced_perceptual_constraints
)


def square_attack_with_enhanced_constraints(image, classifier, image_path, eps=0.05, norm='inf', 
                                           max_iter=100, p_init=0.8, targeted=False, target_class=None,
                                           ssim_threshold=0.95, lpips_threshold=0.05, clip_threshold=0.9):
    """Apply Square Attack with enhanced perceptual constraints"""
    # Initialize models for perceptual metrics
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Initialize LPIPS model
    print("Initializing LPIPS model...")
    lpips_model = lpips.LPIPS(net='alex').to(device)
    
    # Initialize CLIP model
    print("Initializing CLIP model...")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    
    # Convert norm string to appropriate format
    if norm == 'inf':
        norm_type = np.inf
    elif norm == '2':
        norm_type = 2
    else:
        raise ValueError("Norm must be either 'inf' or '2'")
    
    # Create Square Attack
    attack = SquareAttack(
        estimator=classifier,
        norm=norm_type,
        max_iter=max_iter,
        eps=eps,
        p_init=p_init,
        nb_restarts=1,
        batch_size=1,
        verbose=True
    )
    
    # Preprocess image using utility function
    img_tensor = preprocess_image_for_attack(image)
    
    # Generate adversarial example
    print(f"Generating adversarial example with Square Attack:")
    print(f"- Epsilon: {eps}")
    print(f"- Norm: {norm}")
    print(f"- Max iterations: {max_iter}")
    print(f"- Initial probability: {p_init}")
    print(f"- Targeted: {targeted}")
    print(f"- SSIM threshold: {ssim_threshold}")
    print(f"- LPIPS threshold: {lpips_threshold}")
    print(f"- CLIP threshold: {clip_threshold}")
    
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
    
    # Apply enhanced perceptual constraints
    adv_image = apply_enhanced_perceptual_constraints(
        image, adv_image, ssim_threshold, lpips_threshold, clip_threshold,
        lpips_model, clip_model, clip_preprocess
    )
    
    return adv_image


def main():
    parser = argparse.ArgumentParser(description="Generate adversarial examples using Square Attack with enhanced perceptual constraints")
    parser.add_argument("--image_path", type=str, 
                        default="data/test_extracted/chart/20231114102825506748.png",
                        help="Path to the input image")
    parser.add_argument("--eps", type=float, default=0.05,
                        help="Maximum perturbation (default: 0.05)")
    parser.add_argument("--norm", type=str, default='inf', choices=['inf', '2'],
                        help="Norm to use for the attack: 'inf' or '2' (default: 'inf')")
    parser.add_argument("--max_iter", type=int, default=100,
                        help="Maximum number of iterations (default: 100)")
    parser.add_argument("--p_init", type=float, default=0.8,
                        help="Initial probability for square size (default: 0.8)")
    parser.add_argument("--targeted", action="store_true",
                        help="Use targeted attack instead of untargeted")
    parser.add_argument("--target_class", type=int, default=None,
                        help="Target class for targeted attack (default: None)")
    parser.add_argument("--ssim_threshold", type=float, default=0.95,
                        help="SSIM threshold for perceptual constraint (default: 0.95)")
    parser.add_argument("--lpips_threshold", type=float, default=0.05,
                        help="LPIPS threshold for perceptual constraint (default: 0.05)")
    parser.add_argument("--clip_threshold", type=float, default=0.9,
                        help="CLIP similarity threshold (default: 0.9)")
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
    
    # Apply Square Attack with enhanced perceptual constraints
    adv_image = square_attack_with_enhanced_constraints(
        image, classifier, args.image_path, args.eps, args.norm, 
        args.max_iter, args.p_init, args.targeted, args.target_class,
        args.ssim_threshold, args.lpips_threshold, args.clip_threshold
    )
    
    # Get output path using utility function
    output_path = get_output_path(args.image_path, 'square')
    
    # Save adversarial image
    save_image(adv_image, output_path)
    
    # Print attack information
    print_attack_info(output_path, image, adv_image, 'square')
    
    # Print additional attack-specific information
    perturbation = np.abs(image.astype(np.float32) - adv_image.astype(np.float32))
    print(f"Max perturbation: {np.max(perturbation)}")
    print(f"Mean perturbation: {np.mean(perturbation)}")
    print(f"SSIM: {calculate_ssim(image, adv_image):.4f}")
    
    # Print query efficiency information
    print(f"Total queries used: approximately {args.max_iter}")


if __name__ == "__main__":
    main()
