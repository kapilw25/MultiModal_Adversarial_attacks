#!/usr/bin/env python3
"""
Carlini-Wagner L∞ Attack Script for Vision-Language Models

This script applies a Carlini-Wagner L∞ adversarial attack to images
to test the robustness of vision-language models. The CW-L∞ attack is described in the paper
"Towards Evaluating the Robustness of Neural Networks" by Carlini and Wagner.

The L∞ attack minimizes the maximum change to any pixel in the image.

Usage:
    python v6_cw_linf_attack.py [--image_path PATH] [--max_iter ITERATIONS] [--confidence CONF] [--binary_steps STEPS]

Example:
    python v6_cw_linf_attack.py --image_path data/test_extracted/chart/image.png --confidence 5 --binary_steps 10
"""

import os
import cv2
import numpy as np
import argparse
from PIL import Image
import torch
from torchvision import transforms, models
from art.estimators.classification import PyTorchClassifier
import time


def load_image(image_path):
    """Load and preprocess an image for the model"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def create_classifier(device='cuda:0'):
    """Create a PyTorch classifier for the attack"""
    # Use a pre-trained ResNet model as a substitute model for the attack
    # This is because we don't have direct access to the VLM's vision encoder
    model = models.resnet50(pretrained=True)
    model.to(device).eval()
    
    # Define preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Create ART classifier
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0.0, 1.0),
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=(3, 224, 224),
        nb_classes=1000,
        preprocessing=(mean, std),
        device_type=device
    )
    
    return classifier


def cw_linf_attack(image, classifier, max_iter=100, confidence=5.0, binary_steps=10):
    """Apply Carlini-Wagner L∞ attack to the image
    
    The CW-L∞ attack aims to minimize the maximum change to any pixel.
    It uses binary search to find the smallest epsilon that produces a successful adversarial example.
    
    Implementation based on the algorithm described in:
    "Towards Evaluating the Robustness of Neural Networks" by Carlini and Wagner
    """
    # Preprocess image
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Convert to tensor and add batch dimension
    img_tensor = transform(image).unsqueeze(0).numpy()
    
    # Get the original prediction
    original_pred = np.argmax(classifier.predict(img_tensor)[0])
    
    # Target is any class other than the original (for untargeted attack)
    target = (original_pred + 1) % classifier.nb_classes
    
    print(f"Starting CW-L∞ attack with max_iter={max_iter}, confidence={confidence}")
    print(f"Original prediction: {original_pred}, Target: {target}")
    
    # Initialize binary search for the smallest epsilon
    lower_bound = 0.0
    upper_bound = 1.0
    best_adv = None
    best_epsilon = float('inf')
    
    # Binary search for the smallest epsilon
    for binary_step in range(binary_steps):
        current_epsilon = (lower_bound + upper_bound) / 2.0
        print(f"Binary search step {binary_step+1}/{binary_steps}, epsilon = {current_epsilon:.6f}")
        
        # Initialize adversarial example
        adv_image = img_tensor.copy()
        
        # Perform gradient descent to find adversarial example
        for iteration in range(max_iter):
            # Compute gradients
            with torch.enable_grad():
                x = torch.tensor(adv_image, requires_grad=True, device=classifier._device)
                logits = classifier.model(x)
                
                # Compute loss that encourages misclassification with high confidence
                target_logit = logits[0, target]
                other_logits = torch.cat([logits[0, :target], logits[0, target+1:]])
                max_other = torch.max(other_logits)
                
                loss = max_other - target_logit + confidence
                loss.backward()
                
                # Get gradients
                grads = x.grad.cpu().numpy()[0]
            
            # Update the adversarial example using sign of gradients (like FGSM)
            # but constrained by the current epsilon
            perturbation = np.sign(grads) * current_epsilon
            
            # Apply perturbation
            new_adv = img_tensor + perturbation
            
            # Clip to valid image range
            new_adv = np.clip(new_adv, 0.0, 1.0)
            
            # Ensure the perturbation is within L∞ constraint
            delta = new_adv - img_tensor
            delta = np.clip(delta, -current_epsilon, current_epsilon)
            new_adv = img_tensor + delta
            
            # Update the adversarial example
            adv_image = new_adv
            
            # Check if the attack is successful
            new_pred = np.argmax(classifier.predict(adv_image)[0])
            
            if new_pred != original_pred:
                print(f"Found adversarial example at iteration {iteration+1} with epsilon = {current_epsilon:.6f}")
                
                # Update best adversarial example if this epsilon is smaller
                if current_epsilon < best_epsilon:
                    best_epsilon = current_epsilon
                    best_adv = adv_image.copy()
                
                # Update binary search bounds
                upper_bound = current_epsilon
                break
                
            if iteration == max_iter - 1:
                # If we reach the maximum iterations without success, increase epsilon
                lower_bound = current_epsilon
    
    if best_adv is None:
        print("Failed to find an adversarial example")
        return image
    
    # Convert back to uint8 format
    adv_image = best_adv[0].transpose(1, 2, 0)
    adv_image = np.clip(adv_image, 0, 1) * 255
    adv_image = adv_image.astype(np.uint8)
    
    # Resize back to original dimensions
    adv_image = cv2.resize(adv_image, (image.shape[1], image.shape[0]))
    
    return adv_image


def save_image(image, output_path):
    """Save the image to the specified path"""
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save image
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"Saved adversarial image to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate adversarial examples using Carlini-Wagner L∞ attack")
    parser.add_argument("--image_path", type=str, 
                        default="data/test_extracted/chart/20231114102825506748.png",
                        help="Path to the input image")
    parser.add_argument("--max_iter", type=int, default=50,
                        help="Maximum number of iterations per binary search step (default: 50)")
    parser.add_argument("--confidence", type=float, default=5.0,
                        help="Confidence parameter for attack (higher values produce stronger attacks)")
    parser.add_argument("--binary_steps", type=int, default=10,
                        help="Number of binary search steps (default: 10)")
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
    
    # Apply CW-L∞ attack
    start_time = time.time()
    adv_image = cw_linf_attack(image, classifier, args.max_iter, args.confidence, args.binary_steps)
    end_time = time.time()
    
    # Create output path
    input_dir = os.path.dirname(args.image_path)
    filename = os.path.basename(args.image_path)
    output_dir = input_dir.replace('test_extracted', 'test_extracted_adv_cw_linf')
    output_path = os.path.join(output_dir, filename)
    
    # Save adversarial image
    save_image(adv_image, output_path)
    
    # Print perturbation statistics
    perturbation = np.abs(image.astype(np.float32) - adv_image.astype(np.float32))
    print(f"L∞ norm (max perturbation): {np.max(perturbation)}")
    print(f"Mean perturbation: {np.mean(perturbation)}")
    print(f"Attack took {end_time - start_time:.2f} seconds")
    
    print("\nTo use this adversarial image in evaluation:")
    print(f"1. The image is saved at: {output_path}")
    print("2. When running eval_model.py, the script will use the original path")
    print("3. To use adversarial images, modify the img_path in eval_model.py:")
    print("   Change: img_path = 'data/test_extracted/' + data['image']")
    print("   To:     img_path = 'test_extracted_adv_cw_linf/' + data['image']")
    
    print("\nCW-L∞ Attack Information:")
    print("- Optimizes for minimal maximum change to any pixel")
    print("- Uses binary search to find the smallest epsilon that produces a successful attack")
    print("- Produces uniform perturbations across the image")
    print("- Compared to L2 and L0 attacks, L∞ attacks often produce more visible but evenly distributed changes")


if __name__ == "__main__":
    main()
