#!/usr/bin/env python3
"""
DeepFool Attack Script for Vision-Language Models

This script applies a DeepFool adversarial attack to images to test the robustness of 
vision-language models. DeepFool was introduced by Moosavi-Dezfooli et al. in their paper 
"DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks" and is designed to find
the minimal perturbation needed to cause misclassification.

Usage:
    python v9_deepfool_attack.py [--image_path PATH] [--max_iter ITERATIONS] [--overshoot OVERSHOOT]

Example:
    python v9_deepfool_attack.py --image_path data/test_extracted/chart/image.png --max_iter 50 --overshoot 0.02
"""

import os
import numpy as np
import argparse
import torch
import torch.nn as nn
from torchvision import transforms

# Import utility functions
from v0_attack_utils import (
    load_image, create_classifier, save_image, 
    get_output_path, print_attack_info
)


class DeepFool_Attack:
    """
    Implementation of the DeepFool attack.
    
    DeepFool works by iteratively finding the closest decision boundary to the input sample
    and then pushing the sample across that boundary. It approximates the classifier as a
    linear model at each iteration and moves toward the closest decision boundary.
    """
    def __init__(self, model, device='cuda:0', num_classes=1000):
        self.model = model
        self.device = device
        self.num_classes = num_classes
    
    def attack(self, image, max_iter=50, overshoot=0.02):
        """
        Generate an adversarial example using DeepFool attack
        
        Args:
            image: Original image tensor (1, C, H, W)
            max_iter: Maximum number of iterations
            overshoot: Overshoot parameter (typically 0.02)
            
        Returns:
            Adversarial example as a tensor
        """
        # Move image to device and make a copy that requires gradient
        image = image.to(self.device)
        adv_image = image.clone().detach().requires_grad_(True)
        
        # Get original prediction
        with torch.no_grad():
            output = self.model(image)
            original_class = torch.argmax(output, dim=1).item()
            
        print(f"Original class: {original_class}")
        
        # Initialize variables
        current_class = original_class
        iteration = 0
        total_perturbation = torch.zeros_like(image)
        
        # Main attack loop
        while current_class == original_class and iteration < max_iter:
            # Forward pass
            output = self.model(adv_image)
            
            # Get current class
            current_class = torch.argmax(output, dim=1).item()
            
            if current_class != original_class:
                break
                
            # Get logits for original class
            f_original = output[0, original_class]
            
            # Initialize variables for finding closest boundary
            min_distance = float('inf')
            closest_perturbation = None
            
            # For each class (except the original)
            for k in range(self.num_classes):
                if k == original_class:
                    continue
                    
                # Zero gradients
                if adv_image.grad is not None:
                    adv_image.grad.zero_()
                
                # Get logits for class k
                f_k = output[0, k]
                
                # Compute gradient of (f_k - f_original) with respect to the image
                loss = f_k - f_original
                loss.backward(retain_graph=True)
                
                # Get gradient
                grad = adv_image.grad.clone()
                
                # Compute perturbation to reach the decision boundary
                # w_k = grad of (f_k - f_original)
                # f_k(x) - f_original(x) = w_k^T * r
                # We want to find the smallest r such that f_k(x+r) >= f_original(x+r)
                # This is given by: r = -[(f_k(x) - f_original(x)) / ||w_k||^2] * w_k
                
                # Compute norm of gradient
                grad_norm = torch.norm(grad) + 1e-10  # Add small constant to avoid division by zero
                
                # Compute perturbation
                perturbation = -loss.item() * grad / (grad_norm ** 2)
                
                # Compute distance to decision boundary
                distance = abs(loss.item()) / grad_norm
                
                # Check if this is the closest boundary
                if distance < min_distance:
                    min_distance = distance
                    closest_perturbation = perturbation
            
            # Apply the perturbation
            if closest_perturbation is not None:
                adv_image = adv_image.detach() + closest_perturbation
                total_perturbation += closest_perturbation
                
                # Ensure pixel values are in valid range [0, 1]
                adv_image = torch.clamp(adv_image, 0, 1)
                adv_image.requires_grad_(True)
            
            # Print progress every 10 iterations
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{max_iter}, Current class: {current_class}, Min distance: {min_distance:.6f}")
                
            iteration += 1
        
        # Apply overshoot
        adv_image = image + (1 + overshoot) * total_perturbation
        adv_image = torch.clamp(adv_image, 0, 1)
        
        # Final check
        with torch.no_grad():
            output = self.model(adv_image)
            final_class = torch.argmax(output, dim=1).item()
        
        print(f"Final class: {final_class}, Original class: {original_class}")
        print(f"Attack {'succeeded' if final_class != original_class else 'failed'} after {iteration} iterations")
        print(f"L2 perturbation norm: {torch.norm(adv_image - image).item():.6f}")
        
        return adv_image


def deepfool_attack(image, classifier, max_iter=50, overshoot=0.02):
    """Apply DeepFool attack to the image"""
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
    
    # Create attack
    attack = DeepFool_Attack(classifier.model, device)
    
    # Generate adversarial example
    print(f"Generating adversarial example with max_iter={max_iter}, overshoot={overshoot}")
    adv_tensor = attack.attack(img_tensor, max_iter, overshoot)
    
    # Convert back to numpy array
    adv_image = adv_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
    adv_image = np.clip(adv_image, 0, 1) * 255
    adv_image = adv_image.astype(np.uint8)
    
    # Resize back to original dimensions
    adv_image = cv2.resize(adv_image, (image.shape[1], image.shape[0]))
    
    return adv_image


def print_deepfool_info():
    """Print information about the DeepFool attack"""
    print("\nDeepFool Attack Information:")
    print("- Designed to find the minimal perturbation needed to cause misclassification")
    print("- Works by iteratively finding the closest decision boundary")
    print("- Approximates the classifier as a linear model at each iteration")
    print("- Typically produces smaller perturbations than FGSM (2-10x smaller)")
    print("- Untargeted attack (pushes sample across nearest decision boundary)")
    print("- Optimized for L2 norm (Euclidean distance)")
    print("- Useful for measuring model robustness")


def main():
    parser = argparse.ArgumentParser(description="Generate adversarial examples using DeepFool attack")
    parser.add_argument("--image_path", type=str, 
                        default="data/test_extracted/chart/20231114102825506748.png",
                        help="Path to the input image")
    parser.add_argument("--max_iter", type=int, default=50,
                        help="Maximum number of iterations (default: 50)")
    parser.add_argument("--overshoot", type=float, default=0.02,
                        help="Overshoot parameter (default: 0.02)")
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
    
    # Apply DeepFool attack
    adv_image = deepfool_attack(image, classifier, args.max_iter, args.overshoot)
    
    # Get output path
    output_path = get_output_path(args.image_path, 'deepfool')
    
    # Save adversarial image
    save_image(adv_image, output_path)
    
    # Print attack information
    print_attack_info(output_path, image, adv_image, 'deepfool')
    print_deepfool_info()


if __name__ == "__main__":
    import cv2  # Import here to avoid circular import with v0_attack_utils
    main()
