#!/usr/bin/env python3
"""
Carlini-Wagner L0 Attack Script for Vision-Language Models

This script applies a Carlini-Wagner L0 adversarial attack to images
to test the robustness of vision-language models. The CW-L0 attack is described in the paper
"Towards Evaluating the Robustness of Neural Networks" by Carlini and Wagner.

The L0 attack minimizes the number of pixels changed in the image.

Usage:
    python v5_cw_l0_attack.py [--image_path PATH] [--max_iter ITERATIONS] [--confidence CONF]

Example:
    python v5_cw_l0_attack.py --image_path data/test_extracted/chart/image.png --max_iter 50 --confidence 10
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


def cw_l0_attack(image, classifier, max_iter=100, confidence=10.0, initial_const=10.0):
    """Apply Carlini-Wagner L0 attack to the image
    
    The CW-L0 attack aims to minimize the number of pixels changed.
    It uses an iterative approach that starts with all pixels and gradually
    reduces the set of pixels that can be modified.
    
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
    
    # Create a copy of the image to work with
    adv_image = img_tensor.copy()
    
    # Get image dimensions
    _, channels, height, width = img_tensor.shape
    
    # Initialize a mask of pixels that can be changed (all True initially)
    pixel_mask = np.ones((channels, height, width), dtype=bool)
    
    # Track the best adversarial example found
    best_adv = None
    best_l0 = float('inf')
    
    print(f"Starting CW-L0 attack with max_iter={max_iter}, confidence={confidence}")
    print(f"Original prediction: {original_pred}")
    
    # Implement the iterative algorithm to find minimal pixel changes
    for iteration in range(max_iter):
        print(f"Iteration {iteration+1}/{max_iter}")
        
        # Use the L2 attack as a subroutine to find important pixels
        # We'll simulate this by computing gradients directly
        with torch.enable_grad():
            x = torch.tensor(adv_image, requires_grad=True, device=classifier._device)
            logits = classifier.model(x)
            
            # Target is any class other than the original
            target = (original_pred + 1) % classifier.nb_classes
            
            # Compute loss that encourages misclassification with high confidence
            target_logit = logits[0, target]
            other_logits = torch.cat([logits[0, :target], logits[0, target+1:]])
            max_other = torch.max(other_logits)
            
            loss = max_other - target_logit + confidence
            loss.backward()
            
            # Get gradients
            grads = x.grad.cpu().numpy()[0]
        
        # Compute importance of each pixel based on gradient magnitude
        importance = np.sum(np.abs(grads), axis=0)
        
        # Apply the current mask
        importance = importance * pixel_mask[0]  # Assuming same mask for all channels
        
        # Find the least important pixels
        if np.sum(pixel_mask) <= 1:
            # If only one pixel is left, we're done
            break
            
        # Remove the least important pixels (reduce by 10% each iteration)
        num_pixels = np.sum(pixel_mask[0])
        num_to_remove = max(1, int(0.1 * num_pixels))
        
        # Get indices of least important pixels
        flat_importance = importance.flatten()
        flat_importance[flat_importance == 0] = float('inf')  # Ignore already masked pixels
        indices = np.argsort(flat_importance)[:num_to_remove]
        
        # Convert flat indices back to 2D coordinates
        y_indices, x_indices = np.unravel_index(indices, (height, width))
        
        # Update the mask
        for y, x in zip(y_indices, x_indices):
            pixel_mask[:, y, x] = False
        
        # Create a new adversarial image with only the allowed pixels changed
        new_adv = img_tensor.copy()
        
        # Apply changes only to unmasked pixels
        for c in range(channels):
            for y in range(height):
                for x in range(width):
                    if pixel_mask[c, y, x]:
                        # Set to a value that maximizes misclassification
                        # This is a simplification; in practice, you'd optimize this value
                        if grads[c, y, x] > 0:
                            new_adv[0, c, y, x] = 1.0  # Max value
                        else:
                            new_adv[0, c, y, x] = 0.0  # Min value
        
        # Check if this is a successful adversarial example
        new_pred = np.argmax(classifier.predict(new_adv)[0])
        
        if new_pred != original_pred:
            # Count the number of changed pixels (L0 norm)
            changed_pixels = np.sum(np.any(new_adv != img_tensor, axis=1))
            
            print(f"Found adversarial example with {changed_pixels} pixels changed")
            
            if changed_pixels < best_l0:
                best_l0 = changed_pixels
                best_adv = new_adv.copy()
                
                # Update the working adversarial example
                adv_image = new_adv.copy()
    
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
    parser = argparse.ArgumentParser(description="Generate adversarial examples using Carlini-Wagner L0 attack")
    parser.add_argument("--image_path", type=str, 
                        default="data/test_extracted/chart/20231114102825506748.png",
                        help="Path to the input image")
    parser.add_argument("--max_iter", type=int, default=50,
                        help="Maximum number of iterations (default: 50)")
    parser.add_argument("--confidence", type=float, default=10.0,
                        help="Confidence parameter for attack (higher values produce stronger attacks)")
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
    
    # Apply CW-L0 attack
    start_time = time.time()
    adv_image = cw_l0_attack(image, classifier, args.max_iter, args.confidence)
    end_time = time.time()
    
    # Create output path
    input_dir = os.path.dirname(args.image_path)
    filename = os.path.basename(args.image_path)
    output_dir = input_dir.replace('test_extracted', 'test_extracted_adv_cw_l0')
    output_path = os.path.join(output_dir, filename)
    
    # Save adversarial image
    save_image(adv_image, output_path)
    
    # Print perturbation statistics
    perturbation = np.abs(image.astype(np.float32) - adv_image.astype(np.float32))
    changed_pixels = np.sum(np.any(perturbation > 0, axis=2))
    print(f"Number of pixels changed (L0 norm): {changed_pixels}")
    print(f"Max perturbation: {np.max(perturbation)}")
    print(f"Attack took {end_time - start_time:.2f} seconds")
    
    print("\nTo use this adversarial image in evaluation:")
    print(f"1. The image is saved at: {output_path}")
    print("2. When running eval_model.py, the script will use the original path")
    print("3. To use adversarial images, modify the img_path in eval_model.py:")
    print("   Change: img_path = 'data/test_extracted/' + data['image']")
    print("   To:     img_path = 'test_extracted_adv_cw_l0/' + data['image']")
    
    print("\nCW-L0 Attack Information:")
    print("- Optimizes for minimal number of pixels changed")
    print("- Uses an iterative approach to identify and modify only the most important pixels")
    print("- Produces sparse perturbations that may be more visible but affect fewer pixels")


if __name__ == "__main__":
    main()
