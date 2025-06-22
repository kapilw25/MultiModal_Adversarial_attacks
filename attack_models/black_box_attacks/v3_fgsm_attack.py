#!/usr/bin/env python3
"""
FGSM Attack Script for Vision-Language Models

This script applies a Fast Gradient Sign Method (FGSM) adversarial attack to images
to test the robustness of vision-language models. FGSM is a one-step attack method
described in the paper "Towards Deep Learning Models Resistant to Adversarial Attacks"
by Madry et al.

Usage:
    python v3_fgsm_attack.py [--image_path PATH] [--eps EPSILON]

Example:
    python v3_fgsm_attack.py --image_path data/test_extracted/chart/20231114102825506748.png --eps 0.03
"""

import os
import cv2
import numpy as np
import argparse
from PIL import Image
import torch
from torchvision import transforms, models
from art.attacks.evasion import FastGradientMethod
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


def fgsm_attack(image, classifier, eps=8/255):
    """Apply FGSM attack to the image
    
    The FGSM attack is defined as:
    x_adv = x + eps * sign(∇_x L(θ, x, y))
    
    Where:
    - x is the original input
    - eps is the perturbation magnitude
    - L is the loss function
    - θ is the model parameters
    - y is the true label
    """
    # Create FGSM attack
    attack = FastGradientMethod(
        estimator=classifier,
        norm=np.inf,
        eps=eps,
        targeted=False,
        batch_size=1,
        minimal=False
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
    print(f"Generating adversarial example with eps={eps}")
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
    parser = argparse.ArgumentParser(description="Generate adversarial examples using FGSM attack")
    parser.add_argument("--image_path", type=str, 
                        default="data/test_extracted/chart/20231114102825506748.png",
                        help="Path to the input image")
    parser.add_argument("--eps", type=float, default=8/255,
                        help="Perturbation magnitude (default: 8/255)")
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
    adv_image = fgsm_attack(image, classifier, args.eps)
    
    # Create output path
    input_dir = os.path.dirname(args.image_path)
    filename = os.path.basename(args.image_path)
    output_dir = input_dir.replace('test_extracted', 'test_extracted_adv_fgsm')
    output_path = os.path.join(output_dir, filename)
    
    # Save adversarial image
    save_image(adv_image, output_path)
    
    # Print perturbation statistics
    perturbation = np.abs(image.astype(np.float32) - adv_image.astype(np.float32))
    print(f"Max perturbation: {np.max(perturbation)}")
    print(f"Mean perturbation: {np.mean(perturbation)}")
    
    print("\nTo use this adversarial image in evaluation:")
    print(f"1. The image is saved at: {output_path}")
    print("2. When running eval_model.py, the script will use the original path")
    print("3. To use adversarial images, modify the img_path in eval_model.py:")
    print("   Change: img_path = 'data/test_extracted/' + data['image']")
    print("   To:     img_path = 'data/test_extracted_adv_fgsm/' + data['image']")
    
    print("\nComparison with PGD attack:")
    print("- FGSM is a one-step attack (faster but potentially less effective)")
    print("- PGD is an iterative attack (slower but typically more powerful)")
    print("- Both attacks use the same epsilon parameter for maximum perturbation")
    print("- To compare results, run both attacks and evaluate model performance on each")


if __name__ == "__main__":
    main()
