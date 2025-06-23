#!/usr/bin/env python3
"""
FGSM Attack Script for Vision-Language Models

This script applies a Fast Gradient Sign Method (FGSM) adversarial attack to images
to test the robustness of vision-language models. FGSM was introduced by Goodfellow et al.
in the paper "Explaining and Harnessing Adversarial Examples" (2014).

The FGSM attack is defined mathematically as:
- For untargeted attacks: x' = x + ε · sign(∇ₓJ(θ, x, y))
- For targeted attacks:   x' = x - ε · sign(∇ₓJ(θ, x, t))

Where:
- x is the original input image
- x' is the adversarial example
- ε is the perturbation magnitude (controls how much each pixel can change)
- J is the loss function
- θ represents the model parameters
- y is the true label
- t is the target label
- ∇ₓJ represents the gradient of the loss with respect to the input x
- sign() is the sign function that returns -1, 0, or 1 depending on the sign of its input

Usage:
    python v3_fgsm_attack.py [--image_path PATH] [--eps EPSILON] [--targeted] [--target_class CLASS]

Example:
    # Untargeted attack
    python v3_fgsm_attack.py --image_path data/test_extracted/chart/image.png --eps 0.03
    
    # Targeted attack
    python v3_fgsm_attack.py --image_path data/test_extracted/chart/image.png --eps 0.03 --targeted --target_class 20
"""

import os
import numpy as np
import argparse
import torch
from torchvision import transforms

# Import utility functions
from v0_attack_utils import (
    load_image, create_classifier, save_image, 
    get_output_path, print_attack_info, preprocess_image_for_attack
)
from art.attacks.evasion import FastGradientMethod


def fgsm_attack(image, classifier, eps=8/255, targeted=False, target_class=None):
    """Apply FGSM attack to the image
    
    The FGSM attack is defined as:
    - For untargeted attacks: x' = x + ε · sign(∇ₓJ(θ, x, y))
    - For targeted attacks:   x' = x - ε · sign(∇ₓJ(θ, x, t))
    
    Where:
    - x is the original input image
    - x' is the adversarial example
    - ε is the perturbation magnitude
    - J is the loss function
    - θ represents the model parameters
    - y is the true label
    - t is the target label
    
    Args:
        image: Original image to attack
        classifier: Model to attack
        eps: Epsilon parameter controlling perturbation magnitude
        targeted: Whether to perform a targeted attack
        target_class: Target class for targeted attack (ignored if targeted=False)
    
    Returns:
        Adversarial image
    """
    # Preprocess image
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Convert to tensor and add batch dimension
    img_tensor = transform(image).unsqueeze(0).numpy()
    
    # For targeted attacks, we need to specify the target class
    if targeted and target_class is None:
        # Get original prediction
        preds = classifier.predict(img_tensor)
        original_class = np.argmax(preds, axis=1)[0]
        
        # Choose a random class different from the original
        target_class = np.random.randint(0, 1000)
        while target_class == original_class:
            target_class = np.random.randint(0, 1000)
        
        print(f"Original class: {original_class}, randomly selected target class: {target_class}")
    
    # Create FGSM attack
    attack = FastGradientMethod(
        estimator=classifier,
        norm=np.inf,  # L∞ norm as specified in the paper
        eps=eps,
        targeted=targeted,
        batch_size=1,
        minimal=False
    )
    
    # Generate adversarial example
    attack_type = "targeted" if targeted else "untargeted"
    print(f"Generating {attack_type} adversarial example with eps={eps}")
    
    if targeted:
        # For targeted attacks, we need to provide the target class
        target = np.zeros((1, 1000))
        target[0, target_class] = 1
        adv_image = attack.generate(x=img_tensor, y=target)
    else:
        # For untargeted attacks, we don't need to provide a target
        adv_image = attack.generate(x=img_tensor)
    
    # Convert back to uint8 format
    adv_image = adv_image[0].transpose(1, 2, 0)
    adv_image = np.clip(adv_image, 0, 1) * 255
    adv_image = adv_image.astype(np.uint8)
    
    # Resize back to original dimensions
    adv_image = cv2.resize(adv_image, (image.shape[1], image.shape[0]))
    
    return adv_image


def print_fgsm_info(targeted=False):
    """Print information about the FGSM attack"""
    print("\nFGSM Attack Information:")
    print("- Fast Gradient Sign Method (FGSM) is a one-step attack optimized for the L∞ distance metric")
    print("- Mathematical formulation:")
    if targeted:
        print("  x' = x - ε · sign(∇ₓJ(θ, x, t))  # Targeted attack")
    else:
        print("  x' = x + ε · sign(∇ₓJ(θ, x, y))  # Untargeted attack")
    print("- Designed primarily for speed rather than producing minimal perturbations")
    print("- Uses a single step in the direction of the gradient sign")
    
    print("\nComparison with PGD attack:")
    print("- FGSM: Single-step attack (faster but less effective)")
    print("- PGD: Multi-step attack (slower but more effective)")
    print("- PGD can be viewed as an iterative version of FGSM with smaller step sizes")
    print("- PGD formula: xₜ₊₁' = Proj_ε(xₜ' + α · sign(∇ₓJ(θ, xₜ', y)))")
    print("- PGD explores the loss landscape more thoroughly, finding better adversarial examples")
    print("- FGSM is more suitable for fast adversarial training, while PGD is better for evaluation")


def main():
    parser = argparse.ArgumentParser(description="Generate adversarial examples using FGSM attack")
    parser.add_argument("--image_path", type=str, 
                        default="data/test_extracted/chart/20231114102825506748.png",
                        help="Path to the input image")
    parser.add_argument("--eps", type=float, default=8/255,
                        help="Perturbation magnitude (default: 8/255)")
    parser.add_argument("--targeted", action="store_true",
                        help="Perform targeted attack instead of untargeted")
    parser.add_argument("--target_class", type=int, default=None,
                        help="Target class for targeted attack (default: random)")
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
    
    # Apply FGSM attack
    adv_image = fgsm_attack(image, classifier, args.eps, args.targeted, args.target_class)
    
    # Get output path
    output_path = get_output_path(args.image_path, 'fgsm')
    
    # Save adversarial image
    save_image(adv_image, output_path)
    
    # Print attack information
    print_attack_info(output_path, image, adv_image, 'fgsm')
    print_fgsm_info(args.targeted)


if __name__ == "__main__":
    import cv2  # Import here to avoid circular import with v0_attack_utils
    main()
