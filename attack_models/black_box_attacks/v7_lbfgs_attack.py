#!/usr/bin/env python3
"""
L-BFGS Attack Script for Vision-Language Models

This script applies an L-BFGS adversarial attack to images to test the robustness of 
vision-language models. The L-BFGS attack was one of the first methods for generating 
adversarial examples, introduced by Szegedy et al. in their paper "Intriguing properties 
of neural networks".

Usage:
    python v7_lbfgs_attack.py [--image_path PATH] [--target_class CLASS] [--c_init C_INIT] [--max_iter ITERATIONS]

Example:
    python v7_lbfgs_attack.py --image_path data/test_extracted/chart/image.png --target_class 20 --c_init 0.1 --max_iter 10
"""

import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

# Import utility functions
from v0_attack_utils import (
    load_image, create_classifier, save_image, 
    get_output_path, print_attack_info
)


class LBFGS_Attack:
    """
    Implementation of the L-BFGS attack.
    
    The L-BFGS attack formulates the problem as:
    minimize c * ||x' - x||_2^2 + loss(x', target_class)
    subject to x' ∈ [0, 1]^n
    
    Where:
    - x is the original input
    - x' is the adversarial example
    - loss is the loss function (typically cross-entropy)
    - c is a constant that balances the two objectives
    """
    def __init__(self, model, device='cuda:0'):
        self.model = model
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()
    
    def attack(self, image, target_class, c_init=0.1, max_iter=10, binary_search_steps=5):
        """
        Generate an adversarial example using L-BFGS attack
        
        Args:
            image: Original image tensor (1, C, H, W)
            target_class: Target class for the adversarial example
            c_init: Initial value of the constant c
            max_iter: Maximum number of iterations for L-BFGS
            binary_search_steps: Number of binary search steps to find optimal c
        
        Returns:
            Adversarial example as a tensor
        """
        # Move image to device
        image = image.to(self.device)
        
        # Get original prediction
        with torch.no_grad():
            original_output = self.model(image)
            original_class = torch.argmax(original_output, dim=1).item()
        
        print(f"Original class: {original_class}, Target class: {target_class}")
        
        # If target is the same as original, choose a different target
        if target_class == original_class:
            target_class = (target_class + 1) % 1000
            print(f"Target class is the same as original, changing to: {target_class}")
        
        # Create target tensor
        target = torch.tensor([target_class], device=self.device)
        
        # Binary search for the optimal c value
        c_lower = 0
        c_upper = 1e10
        c = c_init
        
        best_adv_image = None
        best_l2_dist = float('inf')
        
        for binary_step in range(binary_search_steps):
            print(f"Binary search step {binary_step+1}/{binary_search_steps}, c = {c}")
            
            # Create a copy of the image that requires gradient
            adv_image = image.clone().detach().requires_grad_(True)
            
            # Use L-BFGS optimizer
            optimizer = optim.LBFGS([adv_image], lr=1, max_iter=max_iter)
            
            # Define closure function for L-BFGS
            def closure():
                optimizer.zero_grad()
                
                # Calculate loss
                output = self.model(adv_image)
                l2_dist = torch.norm(adv_image - image, p=2)
                ce_loss = self.loss_fn(output, target)
                total_loss = c * l2_dist + ce_loss
                
                # Backward pass
                total_loss.backward()
                
                return total_loss
            
            # Run optimization
            optimizer.step(closure)
            
            # Check if attack was successful
            with torch.no_grad():
                output = self.model(adv_image)
                adv_class = torch.argmax(output, dim=1).item()
                l2_dist = torch.norm(adv_image - image, p=2).item()
            
            print(f"  L2 distance: {l2_dist:.4f}, Predicted class: {adv_class}")
            
            # Update best adversarial example if this one is better
            if adv_class == target_class and l2_dist < best_l2_dist:
                best_adv_image = adv_image.clone().detach()
                best_l2_dist = l2_dist
            
            # Update c using binary search
            if adv_class == target_class:
                # Attack succeeded, try to reduce distortion
                c_upper = c
                c = (c_lower + c_upper) / 2
            else:
                # Attack failed, increase c to prioritize target class
                c_lower = c
                c = (c_lower + c_upper) / 2 if c_upper < 1e10 else c * 10
        
        # If attack failed, return the last adversarial example
        if best_adv_image is None:
            best_adv_image = adv_image.detach()
        
        # Ensure pixel values are in valid range [0, 1]
        best_adv_image = torch.clamp(best_adv_image, 0, 1)
        
        return best_adv_image


def lbfgs_attack(image, classifier, target_class=None, c_init=0.1, max_iter=10):
    """Apply L-BFGS attack to the image"""
    # Preprocess image
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Convert to tensor and add batch dimension
    img_tensor = transform(image).unsqueeze(0)
    
    # Get device
    device = next(classifier.model.parameters()).device
    
    # If target class is not specified, use a random class
    if target_class is None:
        # Get original prediction
        with torch.no_grad():
            output = classifier.model(img_tensor.to(device))
            original_class = torch.argmax(output, dim=1).item()
        
        # Choose a random class different from the original
        target_class = np.random.randint(0, 1000)
        while target_class == original_class:
            target_class = np.random.randint(0, 1000)
    
    # Create attack
    attack = LBFGS_Attack(classifier.model, device)
    
    # Generate adversarial example
    print(f"Generating adversarial example with target_class={target_class}, c_init={c_init}, max_iter={max_iter}")
    adv_tensor = attack.attack(img_tensor, target_class, c_init, max_iter)
    
    # Convert back to numpy array
    adv_image = adv_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
    adv_image = np.clip(adv_image, 0, 1) * 255
    adv_image = adv_image.astype(np.uint8)
    
    # Resize back to original dimensions
    adv_image = np.zeros_like(image) if adv_image is None else cv2.resize(adv_image, (image.shape[1], image.shape[0]))
    
    return adv_image


def print_lbfgs_info():
    """Print information about the L-BFGS attack"""
    print("\nL-BFGS Attack Information:")
    print("- One of the first methods for generating adversarial examples")
    print("- Optimizes for minimal L2 distance while achieving target classification")
    print("- Uses box-constrained L-BFGS optimization with line search")
    print("- Typically produces visually imperceptible perturbations")
    print("- More computationally intensive than gradient-based methods like FGSM")


def main():
    parser = argparse.ArgumentParser(description="Generate adversarial examples using L-BFGS attack")
    parser.add_argument("--image_path", type=str, 
                        default="data/test_extracted/chart/20231114102825506748.png",
                        help="Path to the input image")
    parser.add_argument("--target_class", type=int, default=None,
                        help="Target class for the adversarial example (default: random)")
    parser.add_argument("--c_init", type=float, default=0.1,
                        help="Initial value of the constant c (default: 0.1)")
    parser.add_argument("--max_iter", type=int, default=10,
                        help="Maximum number of iterations for L-BFGS (default: 10)")
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
    
    # Apply L-BFGS attack
    adv_image = lbfgs_attack(image, classifier, args.target_class, args.c_init, args.max_iter)
    
    # Get output path
    output_path = get_output_path(args.image_path, 'lbfgs')
    
    # Save adversarial image
    save_image(adv_image, output_path)
    
    # Print attack information
    print_attack_info(output_path, image, adv_image, 'lbfgs')
    print_lbfgs_info()


if __name__ == "__main__":
    import cv2  # Import here to avoid circular import with v0_attack_utils
    main()
