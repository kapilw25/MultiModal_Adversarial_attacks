#!/usr/bin/env python3
"""
Jacobian-based Saliency Map Attack (JSMA) Script for Vision-Language Models

This script applies a JSMA adversarial attack to images to test the robustness of 
vision-language models. JSMA was introduced by Papernot et al. in their paper 
"The Limitations of Deep Learning in Adversarial Settings" and is optimized for the L0 
distance metric, aiming to modify the fewest possible pixels.

Usage:
    python v8_jsma_attack.py [--image_path PATH] [--target_class CLASS] [--max_iter ITERATIONS] [--theta THETA] [--use_logits]

Example:
    python v8_jsma_attack.py --image_path data/test_extracted/chart/image.png --target_class 20 --max_iter 100 --theta 1.0
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


class JSMA_Attack:
    """
    Implementation of the Jacobian-based Saliency Map Attack (JSMA).
    
    JSMA is a greedy algorithm that iteratively selects and modifies pixels that have 
    the highest impact on the classification outcome. It uses the gradient information 
    to construct a "saliency map" that quantifies how influential each pixel is for 
    achieving the target classification.
    """
    def __init__(self, model, device='cuda:0', use_logits=True):
        self.model = model
        self.device = device
        self.use_logits = use_logits  # Whether to use logits (Z) or softmax outputs (F)
    
    def compute_jacobian(self, image, target_class, num_classes=1000):
        """
        Compute the Jacobian matrix of the model's output with respect to the input image.
        
        Args:
            image: Input image tensor (1, C, H, W)
            target_class: Target class index
            num_classes: Number of output classes
            
        Returns:
            target_grad: Gradients for target class
            other_grad: Sum of gradients for all other classes
        """
        image.requires_grad_(True)
        
        # Forward pass
        output = self.model(image)
        
        # For target class
        if self.use_logits:
            # Use logits (Z) - JSMA-Z variant
            output[0, target_class].backward(retain_graph=True)
        else:
            # Use softmax outputs (F) - JSMA-F variant
            softmax_output = torch.nn.functional.softmax(output, dim=1)
            softmax_output[0, target_class].backward(retain_graph=True)
        
        # Store target gradients
        target_grad = image.grad.clone()
        
        # Reset gradients
        image.grad.zero_()
        
        # For all other classes (combined)
        if self.use_logits:
            # Create a mask for all classes except target
            mask = torch.ones_like(output)
            mask[0, target_class] = 0
            masked_output = output * mask
            masked_output.sum().backward()
        else:
            # For softmax, we need to handle differently
            softmax_output = torch.nn.functional.softmax(output, dim=1)
            mask = torch.ones_like(softmax_output)
            mask[0, target_class] = 0
            masked_output = softmax_output * mask
            masked_output.sum().backward()
        
        # Store other gradients
        other_grad = image.grad.clone()
        
        # Reset gradients and detach
        image.grad.zero_()
        image.requires_grad_(False)
        
        return target_grad, other_grad
    
    def compute_saliency_scores(self, target_grad, other_grad):
        """
        Compute saliency scores for each pixel.
        
        Args:
            target_grad: Gradients for target class
            other_grad: Sum of gradients for all other classes
            
        Returns:
            Saliency scores for each pixel
        """
        # Flatten gradients
        target_grad_flat = target_grad.flatten()
        other_grad_flat = other_grad.flatten()
        
        # Compute saliency scores
        # Higher score means more influential for classification
        saliency = torch.zeros_like(target_grad_flat)
        
        # Condition: target_grad > 0 and other_grad < 0
        mask = (target_grad_flat > 0) & (other_grad_flat < 0)
        saliency[mask] = target_grad_flat[mask] * (-other_grad_flat[mask])
        
        return saliency
    
    def attack(self, image, target_class, max_iter=100, theta=1.0):
        """
        Generate an adversarial example using JSMA attack
        
        Args:
            image: Original image tensor (1, C, H, W)
            target_class: Target class for the adversarial example
            max_iter: Maximum number of iterations (pixel modifications)
            theta: Maximum distortion per pixel (0-1 range)
            
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
        
        # Get image dimensions
        _, channels, height, width = image.shape
        
        # Create a copy of the image that we'll modify
        adv_image = image.clone()
        
        # Track modified pixels
        modified_pixels = set()
        
        # Main attack loop
        for iteration in range(max_iter):
            # Check if we've already succeeded
            with torch.no_grad():
                output = self.model(adv_image)
                current_class = torch.argmax(output, dim=1).item()
                
            if current_class == target_class:
                print(f"Attack succeeded after {iteration} iterations!")
                break
            
            # Compute Jacobian
            target_grad, other_grad = self.compute_jacobian(adv_image, target_class)
            
            # Compute saliency scores
            saliency = self.compute_saliency_scores(target_grad, other_grad)
            
            # Find the most influential pixel
            if torch.max(saliency) == 0:
                print(f"No valid pixels found at iteration {iteration}")
                break
                
            # Get top pixels (we'll modify one at a time for memory efficiency)
            _, indices = torch.topk(saliency, k=10)
            
            # Try each pixel until we find one that hasn't been modified
            modified = False
            for idx in indices:
                idx = idx.item()
                
                # Convert flat index to 3D indices
                c = idx % channels
                h = (idx // channels) % height
                w = idx // (channels * height)
                
                # Skip if this pixel has already been modified
                if (c, h, w) in modified_pixels:
                    continue
                
                # Modify the pixel
                adv_image[0, c, h, w] = torch.clamp(adv_image[0, c, h, w] + theta, 0, 1)
                
                # Add to modified pixels set
                modified_pixels.add((c, h, w))
                modified = True
                break
            
            # If we couldn't find any unmodified pixels in the top k, break
            if not modified:
                print(f"No unmodified pixels found in top candidates at iteration {iteration}")
                break
            
            # Print progress every 10 iterations
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{max_iter}, Modified pixels: {len(modified_pixels)}")
        
        # Final check
        with torch.no_grad():
            output = self.model(adv_image)
            final_class = torch.argmax(output, dim=1).item()
        
        print(f"Final class: {final_class}, Target class: {target_class}")
        print(f"Total modified pixels: {len(modified_pixels)}")
        
        return adv_image


def jsma_attack(image, classifier, target_class=None, max_iter=100, theta=1.0, use_logits=True):
    """Apply JSMA attack to the image"""
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
    attack = JSMA_Attack(classifier.model, device, use_logits=use_logits)
    
    # Generate adversarial example
    print(f"Generating adversarial example with target_class={target_class}, max_iter={max_iter}, theta={theta}")
    adv_tensor = attack.attack(img_tensor, target_class, max_iter, theta)
    
    # Convert back to numpy array
    adv_image = adv_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
    adv_image = np.clip(adv_image, 0, 1) * 255
    adv_image = adv_image.astype(np.uint8)
    
    # Resize back to original dimensions
    adv_image = cv2.resize(adv_image, (image.shape[1], image.shape[0]))
    
    return adv_image


def print_jsma_info(use_logits=True):
    """Print information about the JSMA attack"""
    variant = "JSMA-Z" if use_logits else "JSMA-F"
    
    print(f"\n{variant} Attack Information:")
    print("- Optimized for the L0 distance metric (minimizing the number of modified pixels)")
    print("- Uses a saliency map to identify the most influential pixels for classification")
    print("- Greedy algorithm that iteratively modifies pixels to achieve target classification")
    print("- Typically produces sparse but potentially visible perturbations")
    print("- Computationally expensive due to Jacobian matrix calculation")
    if use_logits:
        print("- Uses logits (Z) - output before softmax - to calculate gradients")
    else:
        print("- Uses softmax outputs (F) to calculate gradients (useful for defensively distilled networks)")


def main():
    parser = argparse.ArgumentParser(description="Generate adversarial examples using JSMA attack")
    parser.add_argument("--image_path", type=str, 
                        default="data/test_extracted/chart/20231114102825506748.png",
                        help="Path to the input image")
    parser.add_argument("--target_class", type=int, default=None,
                        help="Target class for the adversarial example (default: random)")
    parser.add_argument("--max_iter", type=int, default=100,
                        help="Maximum number of iterations (pixel modifications) (default: 100)")
    parser.add_argument("--theta", type=float, default=1.0,
                        help="Maximum distortion per pixel (0-1 range) (default: 1.0)")
    parser.add_argument("--use_logits", action="store_true", default=True,
                        help="Use logits (Z) instead of softmax outputs (F) for gradient calculation")
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
    
    # Apply JSMA attack
    adv_image = jsma_attack(image, classifier, args.target_class, args.max_iter, args.theta, args.use_logits)
    
    # Get output path
    output_path = get_output_path(args.image_path, 'jsma')
    
    # Save adversarial image
    save_image(adv_image, output_path)
    
    # Print attack information
    print_attack_info(output_path, image, adv_image, 'jsma')
    print_jsma_info(args.use_logits)


if __name__ == "__main__":
    import cv2  # Import here to avoid circular import with v0_attack_utils
    main()
