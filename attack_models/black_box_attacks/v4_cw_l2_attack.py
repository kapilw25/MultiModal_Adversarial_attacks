#!/usr/bin/env python3
"""
Carlini-Wagner L2 Attack Script for Vision-Language Models

This script applies a Carlini-Wagner L2 adversarial attack to images
to test the robustness of vision-language models. The CW attack is described in the paper
"Towards Evaluating the Robustness of Neural Networks" by Carlini and Wagner.

Usage:
    python v4_cw_l2_attack.py [--image_path PATH] [--confidence CONF] [--max_iter ITERATIONS] [--learning_rate LR]

Example:
    python v4_cw_l2_attack.py --image_path data/test_extracted/chart/image.png --confidence 5 --max_iter 100
"""

import os
import cv2
import numpy as np
import argparse
from PIL import Image
import torch
from torchvision import transforms, models
from art.attacks.evasion import CarliniL2Method
from art.estimators.classification import PyTorchClassifier


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


def cw_l2_attack(image, classifier, confidence=5.0, max_iter=100, learning_rate=0.01):
    """Apply Carlini-Wagner L2 attack to the image
    
    The CW-L2 attack formulates the problem as an optimization:
    minimize ||x' - x||_2^2 + c * f(x')
    
    Where:
    - x is the original input
    - x' is the adversarial example
    - f(x') is a function that measures how successful the adversarial example is
    - c is a constant that balances the two objectives
    """
    # Create CW attack
    attack = CarliniL2Method(
        classifier=classifier,
        confidence=confidence,
        targeted=False,
        max_iter=max_iter,
        binary_search_steps=10,
        learning_rate=learning_rate,
        initial_const=0.01,
        verbose=True
    )
    
    # Preprocess image
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Convert to tensor and add batch dimension
    img_tensor = transform(image).unsqueeze(0).numpy()
    
    # Generate adversarial example
    print(f"Generating adversarial example with confidence={confidence}, max_iter={max_iter}, learning_rate={learning_rate}")
    adv_image = attack.generate(x=img_tensor)
    
    # Convert back to uint8 format
    adv_image = adv_image[0].transpose(1, 2, 0)
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
    parser = argparse.ArgumentParser(description="Generate adversarial examples using Carlini-Wagner L2 attack")
    parser.add_argument("--image_path", type=str, 
                        default="data/test_extracted/chart/20231114102825506748.png",
                        help="Path to the input image")
    parser.add_argument("--confidence", type=float, default=5.0,
                        help="Confidence parameter for CW attack (higher values produce stronger attacks)")
    parser.add_argument("--max_iter", type=int, default=100,
                        help="Maximum number of iterations (default: 100)")
    parser.add_argument("--learning_rate", type=float, default=0.01,
                        help="Learning rate for the optimization (default: 0.01)")
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
    
    # Apply CW-L2 attack
    adv_image = cw_l2_attack(image, classifier, args.confidence, args.max_iter, args.learning_rate)
    
    # Create output path
    input_dir = os.path.dirname(args.image_path)
    filename = os.path.basename(args.image_path)
    output_dir = input_dir.replace('test_extracted', 'test_extracted_adv_cw_l2')
    output_path = os.path.join(output_dir, filename)
    
    # Save adversarial image
    save_image(adv_image, output_path)
    
    # Print perturbation statistics
    perturbation = np.abs(image.astype(np.float32) - adv_image.astype(np.float32))
    print(f"Max perturbation: {np.max(perturbation)}")
    print(f"Mean perturbation: {np.mean(perturbation)}")
    print(f"L2 norm of perturbation: {np.linalg.norm(perturbation)}")
    
    print("\nTo use this adversarial image in evaluation:")
    print(f"1. The image is saved at: {output_path}")
    print("2. When running eval_model.py, the script will use the original path")
    print("3. To use adversarial images, modify the img_path in eval_model.py:")
    print("   Change: img_path = 'data/test_extracted/' + data['image']")
    print("   To:     img_path = 'test_extracted_adv_cw_l2/' + data['image']")
    
    print("\nCW-L2 Attack Information:")
    print("- Optimizes for minimal L2 (Euclidean) distance between original and adversarial images")
    print("- Balances distortion and attack success using the confidence parameter")
    print("- Generally produces visually imperceptible perturbations")


if __name__ == "__main__":
    main()
