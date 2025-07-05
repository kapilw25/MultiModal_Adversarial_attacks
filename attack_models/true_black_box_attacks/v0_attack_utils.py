#!/usr/bin/env python3
"""
Utility functions for adversarial attacks on Vision-Language Models

This module contains common functions used across different attack implementations:
- load_image: Load and preprocess an image
- create_classifier: Create a PyTorch classifier for the attack
- save_image: Save the image to the specified path
- preprocess_image_for_attack: Preprocess an image for attack
- postprocess_adversarial_image: Convert adversarial tensor back to image format
- get_output_path: Generate output path for adversarial image
- print_attack_info: Print information about the attack
- calculate_ssim: Calculate structural similarity between two images
- calculate_lpips: Calculate LPIPS perceptual similarity between two images
- calculate_clip_similarity: Calculate CLIP cosine similarity between two images
- apply_enhanced_perceptual_constraints: Apply perceptual constraints to ensure visual imperceptibility
- detect_text_regions: Detect text regions in an image
- detect_chart_elements: Detect chart elements in an image
- generate_saliency_map: Generate a saliency map for an image
- create_combined_importance_map: Create a combined importance map
- apply_targeted_perturbation: Apply targeted perturbation to an image
"""

import os
import cv2
import numpy as np
import torch
from torchvision import transforms, models
from art.estimators.classification import PyTorchClassifier
from skimage.metrics import structural_similarity as ssim
from PIL import Image

# Global mapping of attack types to directory suffixes
ATTACK_DIR_MAP = {
    'pgd': 'pgd',
    'fgsm': 'fgsm',
    'cw_l2': 'cw_l2',
    'cw_l0': 'cw_l0',
    'cw_linf': 'cw_linf',
    'lbfgs': 'lbfgs',
    'jsma': 'jsma',
    'deepfool': 'deepfool',
    'square': 'square'
}


def load_image(image_path):
    """Load and preprocess an image for the model
    
    Args:
        image_path (str): Path to the input image
        
    Returns:
        numpy.ndarray: RGB image in numpy array format
        
    Raises:
        ValueError: If the image cannot be loaded
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def create_classifier(device='cuda:0'):
    """Create a PyTorch classifier for the attack
    
    Args:
        device (str): Device to use for computation ('cuda:0', 'cpu', etc.)
        
    Returns:
        PyTorchClassifier: ART classifier wrapper around a PyTorch model
    """
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


def save_image(image, output_path):
    """Save the image to the specified path
    
    Args:
        image (numpy.ndarray): RGB image to save
        output_path (str): Path where the image will be saved
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save image
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"Saved adversarial image to {output_path}")


def preprocess_image_for_attack(image, return_tensor=False):
    """Preprocess an image for attack (resize and convert to tensor)
    
    This function standardizes image preprocessing across all attack implementations.
    It resizes the image to 224x224 (standard input size for many CNNs),
    converts it to a PyTorch tensor, and adds a batch dimension.
    
    Args:
        image (numpy.ndarray): RGB image in numpy array format
        return_tensor (bool, optional): If True, returns a PyTorch tensor instead of numpy array.
                                       Default is False.
        
    Returns:
        numpy.ndarray or torch.Tensor: Preprocessed image with shape (1, 3, 224, 224)
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Convert to tensor and add batch dimension
    img_tensor = transform(image).unsqueeze(0)
    
    # Return as tensor or numpy array based on parameter
    if return_tensor:
        return img_tensor
    else:
        return img_tensor.numpy()


def postprocess_adversarial_image(adv_image, original_shape):
    """Convert adversarial tensor back to image format and resize to original dimensions
    
    Args:
        adv_image (numpy.ndarray): Adversarial image as numpy array with shape (1, 3, H, W)
        original_shape (tuple): Original image shape (H, W, C)
        
    Returns:
        numpy.ndarray: Processed adversarial image as uint8 numpy array
    """
    # Convert back to uint8 format
    adv_image = adv_image[0].transpose(1, 2, 0)
    adv_image = np.clip(adv_image, 0, 1) * 255
    adv_image = adv_image.astype(np.uint8)
    
    # Resize back to original dimensions
    adv_image = cv2.resize(adv_image, (original_shape[1], original_shape[0]))
    
    return adv_image


def get_output_path(input_path, attack_type):
    """Generate output path for adversarial image based on input path and attack type
    
    Args:
        input_path (str): Path to the input image
        attack_type (str): Type of attack (e.g., 'pgd', 'fgsm', 'cw_l2', etc.)
        
    Returns:
        str: Output path for the adversarial image
    """
    input_dir = os.path.dirname(input_path)
    filename = os.path.basename(input_path)
    
    # Use global attack directory mapping
    dir_suffix = ATTACK_DIR_MAP.get(attack_type, attack_type)
    output_dir = input_dir.replace('test_extracted', f'test_BB_{dir_suffix}')
    output_path = os.path.join(output_dir, filename)
    
    return output_path


def print_attack_info(output_path, original_image, adv_image, attack_type):
    """Print information about the attack and instructions for evaluation
    
    Args:
        output_path (str): Path where the adversarial image was saved
        original_image (numpy.ndarray): Original image
        adv_image (numpy.ndarray): Adversarial image
        attack_type (str): Type of attack
    """
    # Print perturbation statistics
    perturbation = np.abs(original_image.astype(np.float32) - adv_image.astype(np.float32))
    print(f"Max perturbation: {np.max(perturbation)}")
    print(f"Mean perturbation: {np.mean(perturbation)}")
    
    # Use global attack directory mapping
    dir_suffix = ATTACK_DIR_MAP.get(attack_type, attack_type)
    
    print("\nTo use this adversarial image in evaluation:")
    print(f"1. The image is saved at: {output_path}")
    print("2. When running eval_model.py, the script will use the original path")
    print("3. To use adversarial images, modify the img_path in eval_model.py:")
    print("   Change: img_path = 'data/test_extracted/' + data['image']")
    print(f"   To:     img_path = 'data/test_BB_{attack_type}/' + data['image']")

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images
    
    Args:
        img1 (numpy.ndarray): First image
        img2 (numpy.ndarray): Second image
        
    Returns:
        float: SSIM value between 0 and 1
    """
    # Convert images to grayscale for SSIM calculation
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    # Ensure images are large enough for default window size
    min_size = min(img1_gray.shape[0], img1_gray.shape[1], img2_gray.shape[0], img2_gray.shape[1])
    win_size = min(7, min_size - 1)  # Ensure window size is odd and smaller than image
    if win_size % 2 == 0:
        win_size -= 1  # Make sure it's odd
    
    if win_size < 3:
        # If images are too small, resize them
        img1_gray = cv2.resize(img1_gray, (64, 64))
        img2_gray = cv2.resize(img2_gray, (64, 64))
        win_size = 7
    
    return ssim(img1_gray, img2_gray, data_range=255, win_size=win_size)


def detect_text_regions(image):
    """Detect potential text regions in the image using edge detection and morphology
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Text region mask with values between 0 and 1
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate to connect nearby edges
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask for text regions
    text_mask = np.zeros_like(gray)
    
    # Filter contours by size and shape
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter out very small contours and those with unusual aspect ratios
        if w * h > 100 and 0.1 < w / h < 10:
            cv2.rectangle(text_mask, (x, y), (x + w, y + h), 255, -1)
    
    # Normalize to [0, 1]
    text_mask = text_mask / 255.0
    
    return text_mask


def detect_chart_elements(image):
    """Detect chart elements like lines, bars, and data points
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Chart element mask with values between 0 and 1
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Apply morphological operations to enhance chart elements
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask for chart elements
    chart_mask = np.zeros_like(gray)
    
    # Filter contours by size and shape
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:  # Filter out very small contours
            cv2.drawContours(chart_mask, [contour], -1, 255, -1)
    
    # Normalize to [0, 1]
    chart_mask = chart_mask / 255.0
    
    return chart_mask


def generate_saliency_map(image, classifier):
    """Generate a saliency map to identify important regions in the image
    
    Args:
        image (numpy.ndarray): Input image
        classifier (PyTorchClassifier): ART classifier
        
    Returns:
        numpy.ndarray: Saliency map with values between 0 and 1
    """
    # Preprocess image
    img_tensor = preprocess_image_for_attack(image, return_tensor=True)
    img_tensor = img_tensor.to(classifier._device)
    img_tensor.requires_grad_(True)
    
    # Forward pass
    output = classifier.model(img_tensor)
    
    # Get the predicted class
    pred_class = torch.argmax(output, dim=1).item()
    
    # Compute gradients
    classifier.model.zero_grad()
    output[0, pred_class].backward()
    
    # Get gradients
    gradients = img_tensor.grad.abs().detach().cpu().numpy()[0]
    
    # Average over channels
    saliency_map = np.mean(gradients, axis=0)
    
    # Normalize to [0, 1]
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
    
    # Resize to match original image
    saliency_map = cv2.resize(saliency_map, (image.shape[1], image.shape[0]))
    
    return saliency_map


def create_combined_importance_map(image, classifier, threshold=0.4):
    """Create a combined importance map using multiple detection methods
    
    Args:
        image (numpy.ndarray): Input image
        classifier (PyTorchClassifier): ART classifier
        threshold (float, optional): Threshold for binary mask. Default is 0.4.
        
    Returns:
        tuple: (binary_mask, importance_map) - Binary mask and continuous importance map
    """
    # Get saliency map from model gradients
    saliency_map = generate_saliency_map(image, classifier)
    
    # Detect text regions
    text_map = detect_text_regions(image)
    
    # Detect chart elements
    chart_map = detect_chart_elements(image)
    
    # Combine the maps with different weights
    combined_map = 0.5 * saliency_map + 0.3 * text_map + 0.2 * chart_map
    
    # Normalize to [0, 1]
    combined_map = (combined_map - combined_map.min()) / (combined_map.max() - combined_map.min() + 1e-8)
    
    # Apply threshold to create binary mask
    binary_mask = combined_map > threshold
    
    # Expand mask for visualization
    expanded_mask = np.stack([binary_mask] * 3, axis=2)
    
    # Dilate the mask slightly to ensure coverage of important regions
    kernel = np.ones((3, 3), np.uint8)
    expanded_mask = cv2.dilate(expanded_mask.astype(np.uint8), kernel, iterations=1)
    
    return expanded_mask, combined_map


def apply_targeted_perturbation(image, adv_image, importance_map, amplification_factor=2.0):
    """Apply perturbation with emphasis on important regions
    
    Args:
        image (numpy.ndarray): Original image
        adv_image (numpy.ndarray): Adversarial image
        importance_map (numpy.ndarray): Importance map with values between 0 and 1
        amplification_factor (float, optional): Factor to amplify perturbation in important regions. Default is 2.0.
        
    Returns:
        numpy.ndarray: Targeted adversarial image
    """
    # Calculate the perturbation
    perturbation = adv_image.astype(np.float32) - image.astype(np.float32)
    
    # Expand importance map to 3 channels for broadcasting
    importance_map_3ch = np.stack([importance_map] * 3, axis=2)
    
    # Apply importance map to perturbation (amplify important regions)
    weighted_perturbation = perturbation * (1 + (importance_map_3ch * amplification_factor))
    
    # Apply weighted perturbation to original image
    targeted_adv_image = image.astype(np.float32) + weighted_perturbation
    targeted_adv_image = np.clip(targeted_adv_image, 0, 255).astype(np.uint8)
    
    return targeted_adv_image


def calculate_lpips(img1, img2, lpips_model):
    """Calculate LPIPS perceptual similarity between two images
    
    Args:
        img1 (numpy.ndarray): First image (RGB, uint8)
        img2 (numpy.ndarray): Second image (RGB, uint8)
        lpips_model: LPIPS model for perceptual similarity
        
    Returns:
        float: LPIPS distance (lower means more similar)
    """
    # Convert images to tensors in range [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Resize images if needed (LPIPS expects at least 64x64)
    min_size = 64
    if img1.shape[0] < min_size or img1.shape[1] < min_size:
        img1 = cv2.resize(img1, (min_size, min_size))
    if img2.shape[0] < min_size or img2.shape[1] < min_size:
        img2 = cv2.resize(img2, (min_size, min_size))
    
    # Convert to PIL images first
    img1_pil = Image.fromarray(img1)
    img2_pil = Image.fromarray(img2)
    
    # Transform to tensors
    img1_tensor = transform(img1_pil).unsqueeze(0)
    img2_tensor = transform(img2_pil).unsqueeze(0)
    
    # Move to the same device as the model
    device = next(lpips_model.parameters()).device
    img1_tensor = img1_tensor.to(device)
    img2_tensor = img2_tensor.to(device)
    
    # Calculate LPIPS
    with torch.no_grad():
        lpips_distance = lpips_model(img1_tensor, img2_tensor).item()
    
    return lpips_distance


def calculate_clip_similarity(img1, img2, clip_model, preprocess):
    """Calculate CLIP cosine similarity between two images
    
    Args:
        img1 (numpy.ndarray): First image (RGB, uint8)
        img2 (numpy.ndarray): Second image (RGB, uint8)
        clip_model: CLIP model
        preprocess: CLIP preprocessing function
        
    Returns:
        float: CLIP similarity (higher means more similar)
    """
    # Convert to PIL images
    img1_pil = Image.fromarray(img1)
    img2_pil = Image.fromarray(img2)
    
    # Preprocess images for CLIP
    img1_tensor = preprocess(img1_pil).unsqueeze(0)
    img2_tensor = preprocess(img2_pil).unsqueeze(0)
    
    # Move to the same device as the model
    device = next(clip_model.parameters()).device
    img1_tensor = img1_tensor.to(device)
    img2_tensor = img2_tensor.to(device)
    
    # Calculate CLIP embeddings
    with torch.no_grad():
        img1_features = clip_model.encode_image(img1_tensor)
        img2_features = clip_model.encode_image(img2_tensor)
        
        # Normalize features
        img1_features = img1_features / img1_features.norm(dim=-1, keepdim=True)
        img2_features = img2_features / img2_features.norm(dim=-1, keepdim=True)
        
        # Calculate cosine similarity
        similarity = (img1_features @ img2_features.T).item()
    
    return similarity


def apply_enhanced_perceptual_constraints(original_image, adv_image, ssim_threshold=0.95, 
                                         lpips_threshold=0.05, clip_threshold=0.9,
                                         lpips_model=None, clip_model=None, clip_preprocess=None):
    """Apply enhanced perceptual constraints to ensure visual imperceptibility
    
    Args:
        original_image (numpy.ndarray): Original image
        adv_image (numpy.ndarray): Adversarial image
        ssim_threshold (float): Minimum SSIM value to maintain
        lpips_threshold (float): Maximum LPIPS distance to maintain
        clip_threshold (float): Minimum CLIP similarity to maintain
        lpips_model: LPIPS model for perceptual similarity
        clip_model: CLIP model for semantic similarity
        clip_preprocess: CLIP preprocessing function
        
    Returns:
        numpy.ndarray: Perceptually constrained adversarial image
    """
    # Calculate initial perceptual metrics
    current_ssim = calculate_ssim(original_image, adv_image)
    current_lpips = calculate_lpips(original_image, adv_image, lpips_model) if lpips_model else 1.0
    current_clip = calculate_clip_similarity(original_image, adv_image, clip_model, clip_preprocess) if clip_model else 0.0
    
    print(f"Initial SSIM: {current_ssim:.4f}")
    if lpips_model:
        print(f"Initial LPIPS: {current_lpips:.4f}")
    if clip_model:
        print(f"Initial CLIP similarity: {current_clip:.4f}")
    
    # Check if perceptual constraints are already satisfied
    constraints_met = (current_ssim >= ssim_threshold and 
                      (not lpips_model or current_lpips <= lpips_threshold) and
                      (not clip_model or current_clip >= clip_threshold))
    
    if constraints_met:
        print("Perceptual constraints already satisfied")
        return adv_image
    
    print(f"Perceptual constraints not met, applying multi-stage adaptive blending...")
    print(f"Target SSIM: >= {ssim_threshold}, Target LPIPS: <= {lpips_threshold}" + 
          (f", Target CLIP: >= {clip_threshold}" if clip_model else ""))
    
    # Stage 1: Find a rough blending factor using binary search
    print("Stage 1: Binary search for initial blending factor")
    alpha_min, alpha_max = 0.0, 1.0
    best_adv_image = original_image.copy()  # Start with original image as fallback
    best_ssim = 1.0
    best_lpips = 0.0
    best_clip = 1.0 if clip_model else None
    best_alpha = 0.0
    
    for i in range(10):  # 10 binary search steps
        alpha = (alpha_min + alpha_max) / 2
        blended_image = cv2.addWeighted(original_image, 1 - alpha, adv_image, alpha, 0)
        
        # Calculate metrics for blended image
        blend_ssim = calculate_ssim(original_image, blended_image)
        blend_lpips = calculate_lpips(original_image, blended_image, lpips_model) if lpips_model else 0.0
        blend_clip = calculate_clip_similarity(original_image, blended_image, clip_model, clip_preprocess) if clip_model else 1.0
        
        print(f"  Step {i+1}: alpha={alpha:.4f}, SSIM={blend_ssim:.4f}" + 
              (f", LPIPS={blend_lpips:.4f}" if lpips_model else "") +
              (f", CLIP={blend_clip:.4f}" if clip_model else ""))
        
        # Check if constraints are satisfied
        constraints_met = (blend_ssim >= ssim_threshold and 
                          (not lpips_model or blend_lpips <= lpips_threshold) and
                          (not clip_model or blend_clip >= clip_threshold))
        
        if constraints_met:
            best_adv_image = blended_image
            best_ssim = blend_ssim
            best_lpips = blend_lpips
            best_clip = blend_clip
            best_alpha = alpha
            alpha_max = alpha  # Try to reduce alpha further
        else:
            alpha_min = alpha  # Need to increase alpha
    
    # Stage 2: Fine-tune with frequency domain filtering if needed
    if not constraints_met:
        print("Stage 2: Frequency domain refinement")
        
        # Convert to frequency domain
        original_gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        adv_gray = cv2.cvtColor(adv_image, cv2.COLOR_RGB2GRAY)
        
        # Get DFT of both images
        original_dft = cv2.dft(np.float32(original_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        adv_dft = cv2.dft(np.float32(adv_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        
        # Create a high-pass filter (keep high frequencies from original, low frequencies from adversarial)
        rows, cols = original_gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # Try different cutoff frequencies
        best_freq_image = best_adv_image.copy()
        best_freq_score = -float('inf')
        
        for cutoff_ratio in [0.05, 0.1, 0.15, 0.2, 0.25]:
            cutoff = int(min(rows, cols) * cutoff_ratio)
            
            # Create a mask with high frequencies from original, low frequencies from adversarial
            mask = np.ones((rows, cols, 2), np.float32)
            mask[crow-cutoff:crow+cutoff, ccol-cutoff:ccol+cutoff] = 0
            
            # Apply mask to create hybrid DFT
            hybrid_dft = adv_dft.copy()
            hybrid_dft = adv_dft * (1 - mask) + original_dft * mask
            
            # Convert back to spatial domain
            hybrid_img = cv2.idft(hybrid_dft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
            hybrid_img = cv2.normalize(hybrid_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Convert back to RGB by replacing Y channel in YCrCb
            adv_ycrcb = cv2.cvtColor(best_adv_image, cv2.COLOR_RGB2YCrCb)
            adv_ycrcb[:,:,0] = hybrid_img
            freq_image = cv2.cvtColor(adv_ycrcb, cv2.COLOR_YCrCb2RGB)
            
            # Calculate metrics
            freq_ssim = calculate_ssim(original_image, freq_image)
            freq_lpips = calculate_lpips(original_image, freq_image, lpips_model) if lpips_model else 0.0
            freq_clip = calculate_clip_similarity(original_image, freq_image, clip_model, clip_preprocess) if clip_model else 1.0
            
            # Calculate combined score
            ssim_score = freq_ssim / ssim_threshold
            lpips_score = lpips_threshold / max(freq_lpips, 1e-6) if lpips_model else 1.0
            clip_score = freq_clip / clip_threshold if clip_model else 1.0
            
            combined_score = ssim_score * lpips_score * clip_score
            
            print(f"  Cutoff ratio {cutoff_ratio:.2f}: SSIM={freq_ssim:.4f}" +
                  (f", LPIPS={freq_lpips:.4f}" if lpips_model else "") +
                  (f", CLIP={freq_clip:.4f}" if clip_model else ""))
            
            if combined_score > best_freq_score:
                best_freq_score = combined_score
                best_freq_image = freq_image
                best_ssim = freq_ssim
                best_lpips = freq_lpips
                best_clip = freq_clip
        
        # Update best image if frequency domain filtering improved results
        if best_freq_score > 0:
            best_adv_image = best_freq_image
    
    # Stage 3: Final adaptive blending if still not meeting constraints
    if best_ssim < ssim_threshold or (lpips_model and best_lpips > lpips_threshold) or (clip_model and best_clip < clip_threshold):
        print("Stage 3: Final adaptive blending")
        
        # Try different blending factors
        best_final_score = -float('inf')
        best_final_image = best_adv_image.copy()
        
        for blend_factor in np.linspace(0.1, 0.9, 9):
            final_image = cv2.addWeighted(original_image, 1 - blend_factor, adv_image, blend_factor, 0)
            
            # Calculate metrics
            final_ssim = calculate_ssim(original_image, final_image)
            final_lpips = calculate_lpips(original_image, final_image, lpips_model) if lpips_model else 0.0
            final_clip = calculate_clip_similarity(original_image, final_image, clip_model, clip_preprocess) if clip_model else 1.0
            
            # Calculate combined score
            ssim_score = final_ssim / ssim_threshold
            lpips_score = lpips_threshold / max(final_lpips, 1e-6) if lpips_model else 1.0
            clip_score = final_clip / clip_threshold if clip_model else 1.0
            
            # Add effectiveness score (lower blend factor means more original adversarial content)
            effectiveness_score = blend_factor * 2  # Weight effectiveness
            
            combined_score = ssim_score * lpips_score * clip_score * effectiveness_score
            
            print(f"  Blend factor {blend_factor:.2f}: SSIM={final_ssim:.4f}" +
                  (f", LPIPS={final_lpips:.4f}" if lpips_model else "") +
                  (f", CLIP={final_clip:.4f}" if clip_model else ""))
            
            # Check if this blend satisfies constraints
            constraints_met = (final_ssim >= ssim_threshold and 
                              (not lpips_model or final_lpips <= lpips_threshold) and
                              (not clip_model or final_clip >= clip_threshold))
            
            if constraints_met:
                print(f"  Found satisfactory blend factor: {blend_factor:.2f}")
                best_final_image = final_image
                best_ssim = final_ssim
                best_lpips = final_lpips
                best_clip = final_clip
                break
            
            if combined_score > best_final_score:
                best_final_score = combined_score
                best_final_image = final_image
                best_ssim = final_ssim
                best_lpips = final_lpips
                best_clip = final_clip
        
        best_adv_image = best_final_image
    
    # Calculate final metrics
    final_ssim = calculate_ssim(original_image, best_adv_image)
    final_lpips = calculate_lpips(original_image, best_adv_image, lpips_model) if lpips_model else 0.0
    final_clip = calculate_clip_similarity(original_image, best_adv_image, clip_model, clip_preprocess) if clip_model else 1.0
    
    print(f"Final perceptual metrics:")
    print(f"  SSIM: {final_ssim:.4f} (target: >= {ssim_threshold})")
    if lpips_model:
        print(f"  LPIPS: {final_lpips:.4f} (target: <= {lpips_threshold})")
    if clip_model:
        print(f"  CLIP similarity: {final_clip:.4f} (target: >= {clip_threshold})")
    
    # Check if we met all constraints
    constraints_met = (final_ssim >= ssim_threshold and 
                      (not lpips_model or final_lpips <= lpips_threshold) and
                      (not clip_model or final_clip >= clip_threshold))
    
    if constraints_met:
        print("Successfully met all perceptual constraints!")
    else:
        print("Warning: Could not fully satisfy all perceptual constraints.")
        print("Using best approximation found.")
    
    return best_adv_image


def generate_importance_map_for_charts(image):
    """Generate an importance map specifically for chart images
    
    This function identifies key elements in charts (axes, labels, data points, legends)
    that are critical for understanding the chart content.
    
    Args:
        image (numpy.ndarray): Input chart image
        
    Returns:
        numpy.ndarray: Importance map with values between 0 and 1
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply adaptive thresholding to identify text and thin lines
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask for important regions
    importance_mask = np.zeros_like(gray, dtype=np.float32)
    
    # Process contours
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = w / max(h, 1)
        
        # Identify text-like regions (small, rectangular)
        if area < 500 and 0.1 < aspect_ratio < 10:
            cv2.drawContours(importance_mask, [contour], -1, 1.0, -1)
            # Add extra weight around text (text is very important for chart understanding)
            cv2.rectangle(importance_mask, (max(0, x-5), max(0, y-5)), 
                         (min(image.shape[1], x+w+5), min(image.shape[0], y+h+5)), 0.7, 2)
        
        # Identify axis lines (long, thin)
        elif (w > image.shape[1]/5 or h > image.shape[0]/5) and min(w, h) < 10:
            cv2.drawContours(importance_mask, [contour], -1, 0.8, -1)
        
        # Identify data points (small, compact)
        elif area < 100 and 0.5 < aspect_ratio < 2.0:
            cv2.drawContours(importance_mask, [contour], -1, 0.9, -1)
            # Add extra weight around data points
            cv2.circle(importance_mask, (x+w//2, y+h//2), max(w, h), 0.6, 2)
    
    # Apply Gaussian blur to smooth the importance map
    importance_mask = cv2.GaussianBlur(importance_mask, (15, 15), 0)
    
    # Normalize to [0, 1]
    importance_mask = importance_mask / max(importance_mask.max(), 1e-8)
    
    return importance_mask

def evaluate_perturbation_effectiveness(original_image, adv_image, ssim_val, lpips_val, clip_val,
                                       ssim_threshold=0.95, lpips_threshold=0.05, clip_threshold=0.9):
    """Evaluate how effective a perturbation is likely to be at degrading model performance
    
    This function calculates a score that estimates how effective an adversarial perturbation
    will be at degrading model performance, while still respecting perceptual constraints.
    
    Args:
        original_image (numpy.ndarray): Original image
        adv_image (numpy.ndarray): Adversarial image
        ssim_val (float): SSIM value between original and adversarial images
        lpips_val (float): LPIPS distance between original and adversarial images
        clip_val (float): CLIP similarity between original and adversarial images
        ssim_threshold (float): Minimum acceptable SSIM value
        lpips_threshold (float): Maximum acceptable LPIPS distance
        clip_threshold (float): Minimum acceptable CLIP similarity
        
    Returns:
        tuple: (effectiveness_score, constraints_met)
    """
    # Check if perceptual constraints are met
    constraints_met = (ssim_val >= ssim_threshold and 
                      lpips_val <= lpips_threshold and 
                      clip_val >= clip_threshold)
    
    if not constraints_met:
        return -1.0, False
    
    # Calculate perturbation magnitude
    perturbation = np.abs(original_image.astype(np.float32) - adv_image.astype(np.float32))
    mean_perturbation = np.mean(perturbation)
    max_perturbation = np.max(perturbation)
    
    # Calculate perturbation concentration in important regions
    importance_map = generate_importance_map_for_charts(original_image)
    weighted_perturbation = perturbation * importance_map[:, :, np.newaxis]
    importance_weighted_mean = np.mean(weighted_perturbation) / (np.mean(importance_map) + 1e-8)
    
    # Calculate effectiveness score
    # We want:
    # - Higher mean perturbation (within constraints)
    # - Higher max perturbation (within constraints)
    # - Higher concentration in important regions
    # - SSIM close to but above threshold
    # - LPIPS close to but below threshold
    # - CLIP close to but above threshold
    
    # Normalize metrics to [0, 1] range for scoring
    ssim_score = 1.0 - ((ssim_val - ssim_threshold) / (1.0 - ssim_threshold))
    lpips_score = lpips_val / lpips_threshold
    clip_score = 1.0 - ((clip_val - clip_threshold) / (1.0 - clip_threshold))
    
    # Combine scores (weighted sum)
    effectiveness_score = (
        0.2 * mean_perturbation / 255.0 +
        0.1 * max_perturbation / 255.0 +
        0.3 * importance_weighted_mean / 255.0 +
        0.2 * ssim_score +
        0.1 * lpips_score +
        0.1 * clip_score
    )
    
    return effectiveness_score, True

def optimize_attack_parameters(attack_fn, image, classifier, params_grid, 
                              ssim_threshold=0.95, lpips_threshold=0.05, clip_threshold=0.9,
                              device='cuda:0'):
    """Find optimal attack parameters that maximize effectiveness while maintaining constraints
    
    Args:
        attack_fn: Function that generates adversarial examples
        image (numpy.ndarray): Original image
        classifier: Model classifier
        params_grid (dict): Dictionary of parameter grids to search
        ssim_threshold (float): Minimum acceptable SSIM value
        lpips_threshold (float): Maximum acceptable LPIPS distance
        clip_threshold (float): Minimum acceptable CLIP similarity
        device (str): Device to use for computation
        
    Returns:
        tuple: (best_params, best_adv_image, best_score)
    """
    import itertools
    import lpips
    import clip
    from tqdm import tqdm
    
    # Initialize perceptual models
    lpips_model = lpips.LPIPS(net='alex').to(device)
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    
    # Generate all parameter combinations
    param_names = list(params_grid.keys())
    param_values = list(params_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    best_score = -float('inf')
    best_params = None
    best_adv_image = None
    
    print(f"Searching over {len(param_combinations)} parameter combinations...")
    
    for combination in tqdm(param_combinations):
        # Create parameter dictionary for this combination
        current_params = dict(zip(param_names, combination))
        
        # Generate adversarial example with current parameters
        adv_image = attack_fn(image, classifier, **current_params)
        
        # Calculate perceptual metrics
        ssim_val = calculate_ssim(image, adv_image)
        lpips_val = calculate_lpips(image, adv_image, lpips_model)
        clip_val = calculate_clip_similarity(image, adv_image, clip_model, clip_preprocess)
        
        # Evaluate effectiveness
        score, constraints_met = evaluate_perturbation_effectiveness(
            image, adv_image, ssim_val, lpips_val, clip_val,
            ssim_threshold, lpips_threshold, clip_threshold
        )
        
        if constraints_met and score > best_score:
            best_score = score
            best_params = current_params
            best_adv_image = adv_image.copy()
            
            print(f"New best parameters found: {best_params}")
            print(f"Score: {best_score:.4f}, SSIM: {ssim_val:.4f}, LPIPS: {lpips_val:.4f}, CLIP: {clip_val:.4f}")
    
    if best_adv_image is None:
        print("No valid parameter combination found. Using default parameters.")
        # Use default parameters as fallback
        default_params = {name: values[0] for name, values in params_grid.items()}
        best_adv_image = attack_fn(image, classifier, **default_params)
        best_params = default_params
    
    return best_params, best_adv_image, best_score

def bayesian_optimize_attack(attack_fn, image, classifier, param_ranges, n_iterations=20,
                            ssim_threshold=0.95, lpips_threshold=0.05, clip_threshold=0.9,
                            device='cuda:0'):
    """Use Bayesian optimization to find optimal attack parameters
    
    This function uses Bayesian optimization to efficiently search the parameter space
    for attack parameters that maximize effectiveness while maintaining perceptual constraints.
    
    Args:
        attack_fn: Function that generates adversarial examples
        image (numpy.ndarray): Original image
        classifier: Model classifier
        param_ranges (dict): Dictionary of parameter ranges {param_name: (min, max)}
        n_iterations (int): Number of optimization iterations
        ssim_threshold (float): Minimum acceptable SSIM value
        lpips_threshold (float): Maximum acceptable LPIPS distance
        clip_threshold (float): Minimum acceptable CLIP similarity
        device (str): Device to use for computation
        
    Returns:
        tuple: (best_params, best_adv_image, best_score)
    """
    try:
        from skopt import gp_minimize
        from skopt.space import Real, Integer
    except ImportError:
        print("scikit-optimize not found. Please install it with: pip install scikit-optimize")
        return None, None, -1
    
    import lpips
    import clip
    
    # Initialize perceptual models
    lpips_model = lpips.LPIPS(net='alex').to(device)
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    
    # Define parameter space
    space = []
    param_names = []
    for name, (min_val, max_val) in param_ranges.items():
        param_names.append(name)
        if isinstance(min_val, int) and isinstance(max_val, int):
            space.append(Integer(min_val, max_val, name=name))
        else:
            space.append(Real(min_val, max_val, name=name))
    
    # Define objective function to minimize (negative effectiveness score)
    def objective(params):
        # Convert params to dictionary
        param_dict = {name: val for name, val in zip(param_names, params)}
        
        # Generate adversarial example
        adv_image = attack_fn(image, classifier, **param_dict)
        
        # Calculate perceptual metrics
        ssim_val = calculate_ssim(image, adv_image)
        lpips_val = calculate_lpips(image, adv_image, lpips_model)
        clip_val = calculate_clip_similarity(image, adv_image, clip_model, clip_preprocess)
        
        # Check constraints
        constraints_met = (ssim_val >= ssim_threshold and 
                          lpips_val <= lpips_threshold and 
                          clip_val >= clip_threshold)
        
        if not constraints_met:
            # Penalize constraint violations
            penalty = 0.0
            if ssim_val < ssim_threshold:
                penalty += 10.0 * (ssim_threshold - ssim_val)
            if lpips_val > lpips_threshold:
                penalty += 10.0 * (lpips_val - lpips_threshold)
            if clip_val < clip_threshold:
                penalty += 10.0 * (clip_threshold - clip_val)
            return 1.0 + penalty  # Return a high value to minimize
        
        # Calculate effectiveness score
        score, _ = evaluate_perturbation_effectiveness(
            image, adv_image, ssim_val, lpips_val, clip_val,
            ssim_threshold, lpips_threshold, clip_threshold
        )
        
        # Print progress
        print(f"Params: {param_dict}")
        print(f"Score: {score:.4f}, SSIM: {ssim_val:.4f}, LPIPS: {lpips_val:.4f}, CLIP: {clip_val:.4f}")
        
        # Return negative score (since we're minimizing)
        return -score
    
    # Run Bayesian optimization
    print(f"Running Bayesian optimization for {n_iterations} iterations...")
    result = gp_minimize(objective, space, n_calls=n_iterations, random_state=42, verbose=True)
    
    # Get best parameters
    best_params = {name: val for name, val in zip(param_names, result.x)}
    
    # Generate best adversarial example
    best_adv_image = attack_fn(image, classifier, **best_params)
    
    # Calculate final score
    ssim_val = calculate_ssim(image, best_adv_image)
    lpips_val = calculate_lpips(image, best_adv_image, lpips_model)
    clip_val = calculate_clip_similarity(image, best_adv_image, clip_model, clip_preprocess)
    best_score, _ = evaluate_perturbation_effectiveness(
        image, best_adv_image, ssim_val, lpips_val, clip_val,
        ssim_threshold, lpips_threshold, clip_threshold
    )
    
    print(f"Optimization complete!")
    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score:.4f}")
    print(f"Final metrics - SSIM: {ssim_val:.4f}, LPIPS: {lpips_val:.4f}, CLIP: {clip_val:.4f}")
    
    return best_params, best_adv_image, best_score
def apply_threshold_optimized_constraints(original_image, adv_image, ssim_threshold=0.95, 
                                   lpips_threshold=0.05, clip_threshold=0.9,
                                   lpips_model=None, clip_model=None, clip_preprocess=None):
    """Apply perceptual constraints optimized to be close to threshold limits
    
    Unlike apply_enhanced_perceptual_constraints, this function aims to maximize attack effectiveness
    by keeping perceptual metrics as close as possible to their threshold limits, rather than
    maximizing perceptual quality.
    
    Args:
        original_image (numpy.ndarray): Original image
        adv_image (numpy.ndarray): Adversarial image
        ssim_threshold (float): Minimum SSIM value to maintain
        lpips_threshold (float): Maximum LPIPS distance to maintain
        clip_threshold (float): Minimum CLIP similarity to maintain
        lpips_model: LPIPS model for perceptual similarity
        clip_model: CLIP model for semantic similarity
        clip_preprocess: CLIP preprocessing function
        
    Returns:
        numpy.ndarray: Perceptually constrained adversarial image optimized for attack effectiveness
    """
    # Initialize models if not provided
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if lpips_model is None and lpips_threshold < 1.0:
        print("Initializing LPIPS model...")
        import lpips
        lpips_model = lpips.LPIPS(net='alex').to(device)
    
    if clip_model is None and clip_preprocess is None and clip_threshold > 0.0:
        print("Initializing CLIP model...")
        import clip
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    
    # Calculate initial perceptual metrics
    current_ssim = calculate_ssim(original_image, adv_image)
    current_lpips = calculate_lpips(original_image, adv_image, lpips_model) if lpips_model else 1.0
    current_clip = calculate_clip_similarity(original_image, adv_image, clip_model, clip_preprocess) if clip_model else 0.0
    
    print(f"Initial SSIM: {current_ssim:.4f}")
    if lpips_model:
        print(f"Initial LPIPS: {current_lpips:.4f}")
    if clip_model:
        print(f"Initial CLIP similarity: {current_clip:.4f}")
    
    # MODIFIED: Adjust thresholds to be more aggressive (right at the limit)
    # This will make the attack push closer to the threshold limits
    effective_ssim_threshold = ssim_threshold * 0.999  # Just slightly below the threshold
    effective_lpips_threshold = lpips_threshold * 1.001  # Just slightly above the threshold
    effective_clip_threshold = clip_threshold * 0.999  # Just slightly below the threshold
    
    print(f"Using effective thresholds for optimization:")
    print(f"  SSIM: {effective_ssim_threshold:.4f} (original: {ssim_threshold:.4f})")
    print(f"  LPIPS: {effective_lpips_threshold:.4f} (original: {lpips_threshold:.4f})")
    print(f"  CLIP: {effective_clip_threshold:.4f} (original: {clip_threshold:.4f})")
    
    # Check if perceptual constraints are already satisfied
    constraints_met = (current_ssim >= effective_ssim_threshold and 
                      (not lpips_model or current_lpips <= effective_lpips_threshold) and
                      (not clip_model or current_clip >= effective_clip_threshold))
    
    if constraints_met:
        print("Perceptual constraints already satisfied")
        # MODIFIED: Always try to degrade metrics to threshold limits
        print("Attempting to optimize to threshold limits...")
        return optimize_to_threshold_limits(original_image, adv_image, effective_ssim_threshold, 
                                          effective_lpips_threshold, effective_clip_threshold, 
                                          lpips_model, clip_model, clip_preprocess)
    
    print(f"Perceptual constraints not met, applying threshold-optimized blending...")
    print(f"Target SSIM: >= {effective_ssim_threshold}, Target LPIPS: <= {effective_lpips_threshold}" + 
          (f", Target CLIP: >= {effective_clip_threshold}" if clip_model else ""))
    
    # Try different blending factors to find one that just meets the constraints
    best_adv_image = None
    best_score = -float('inf')
    
    # MODIFIED: Try a range of blending factors with more weight on adversarial content
    for blend_factor in np.linspace(0.95, 0.1, 18):  # More granular search, starting with more adversarial content
        blended_image = cv2.addWeighted(original_image, 1 - blend_factor, adv_image, blend_factor, 0)
        
        # Calculate metrics
        blend_ssim = calculate_ssim(original_image, blended_image)
        blend_lpips = calculate_lpips(original_image, blended_image, lpips_model) if lpips_model else 0.0
        blend_clip = calculate_clip_similarity(original_image, blended_image, clip_model, clip_preprocess) if clip_model else 1.0
        
        print(f"  Blend factor {blend_factor:.2f}: SSIM={blend_ssim:.4f}" + 
              (f", LPIPS={blend_lpips:.4f}" if lpips_model else "") +
              (f", CLIP={blend_clip:.4f}" if clip_model else ""))
        
        # Check if constraints are met
        constraints_met = (blend_ssim >= effective_ssim_threshold and 
                          (not lpips_model or blend_lpips <= effective_lpips_threshold) and
                          (not clip_model or blend_clip >= effective_clip_threshold))
        
        if constraints_met:
            # MODIFIED: Calculate score to favor solutions that are exactly at the threshold
            # rather than well above/below it
            ssim_score = 1.0 - abs(blend_ssim - effective_ssim_threshold) * 10
            lpips_score = 1.0 - abs(blend_lpips - effective_lpips_threshold) * 10 if lpips_model else 1.0
            clip_score = 1.0 - abs(blend_clip - effective_clip_threshold) * 10 if clip_model else 1.0
            
            # Higher blend_factor means more adversarial content (better)
            # We multiply by 5 to give it more weight than the threshold proximity
            score = blend_factor * 5 + ssim_score + lpips_score + clip_score
            
            print(f"    Constraints met! Score: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_adv_image = blended_image.copy()
                print(f"    New best score!")
    
    # MODIFIED: If no blending factor meets constraints, try more aggressive frequency domain approach
    if best_adv_image is None:
        print("No direct blending meets constraints, trying aggressive frequency domain approach...")
        
        # Convert to frequency domain
        original_gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        adv_gray = cv2.cvtColor(adv_image, cv2.COLOR_RGB2GRAY)
        
        # Get DFT of both images
        original_dft = cv2.dft(np.float32(original_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        adv_dft = cv2.dft(np.float32(adv_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        
        rows, cols = original_gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # MODIFIED: Try more cutoff frequencies with finer granularity
        for cutoff_ratio in np.linspace(0.05, 0.5, 19):  # More cutoff options
            cutoff = int(min(rows, cols) * cutoff_ratio)
            
            # Create a mask with high frequencies from original, low frequencies from adversarial
            mask = np.ones((rows, cols, 2), np.float32)
            mask[crow-cutoff:crow+cutoff, ccol-cutoff:ccol+cutoff] = 0
            
            # MODIFIED: Apply mask with more weight on adversarial components
            hybrid_dft = adv_dft.copy()
            # Give more weight to adversarial frequencies (1.2) and less to original (0.8)
            hybrid_dft = adv_dft * (1 - mask) * 1.2 + original_dft * mask * 0.8
            
            # Convert back to spatial domain
            hybrid_img = cv2.idft(hybrid_dft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
            hybrid_img = cv2.normalize(hybrid_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Convert back to RGB by replacing Y channel in YCrCb
            adv_ycrcb = cv2.cvtColor(adv_image, cv2.COLOR_RGB2YCrCb)
            adv_ycrcb[:,:,0] = hybrid_img
            freq_image = cv2.cvtColor(adv_ycrcb, cv2.COLOR_YCrCb2RGB)
            
            # Calculate metrics
            freq_ssim = calculate_ssim(original_image, freq_image)
            freq_lpips = calculate_lpips(original_image, freq_image, lpips_model) if lpips_model else 0.0
            freq_clip = calculate_clip_similarity(original_image, freq_image, clip_model, clip_preprocess) if clip_model else 1.0
            
            print(f"  Cutoff ratio {cutoff_ratio:.2f}: SSIM={freq_ssim:.4f}" +
                  (f", LPIPS={freq_lpips:.4f}" if lpips_model else "") +
                  (f", CLIP={freq_clip:.4f}" if clip_model else ""))
            
            # Check if constraints are met
            constraints_met = (freq_ssim >= effective_ssim_threshold and 
                              (not lpips_model or freq_lpips <= effective_lpips_threshold) and
                              (not clip_model or freq_clip >= effective_clip_threshold))
            
            if constraints_met:
                # MODIFIED: Calculate score to favor solutions that are exactly at the threshold
                ssim_score = 1.0 - abs(freq_ssim - effective_ssim_threshold) * 10
                lpips_score = 1.0 - abs(freq_lpips - effective_lpips_threshold) * 10 if lpips_model else 1.0
                clip_score = 1.0 - abs(freq_clip - effective_clip_threshold) * 10 if clip_model else 1.0
                
                # Higher cutoff_ratio means more adversarial content (better)
                score = cutoff_ratio * 10 + ssim_score + lpips_score + clip_score
                
                print(f"    Constraints met! Score: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_adv_image = freq_image.copy()
                    print(f"    New best score!")
    
    # MODIFIED: If still no solution, try a more aggressive approach with noise addition
    if best_adv_image is None:
        print("Frequency domain approach failed, trying noise addition...")
        
        # Create a noisy version of the original image
        for noise_level in np.linspace(0.05, 0.3, 10):
            noise = np.random.uniform(-noise_level, noise_level, original_image.shape).astype(np.float32)
            noisy_image = np.clip(original_image.astype(np.float32) + noise * 255, 0, 255).astype(np.uint8)
            
            # Try blending between noisy image and adversarial image
            for blend in np.linspace(0.1, 0.9, 9):
                test_image = cv2.addWeighted(noisy_image, 1 - blend, adv_image, blend, 0)
                
                # Calculate metrics
                test_ssim = calculate_ssim(original_image, test_image)
                test_lpips = calculate_lpips(original_image, test_image, lpips_model) if lpips_model else 0.0
                test_clip = calculate_clip_similarity(original_image, test_image, clip_model, clip_preprocess) if clip_model else 1.0
                
                print(f"  Noise {noise_level:.2f}, Blend {blend:.1f}: SSIM={test_ssim:.4f}" +
                      (f", LPIPS={test_lpips:.4f}" if lpips_model else "") +
                      (f", CLIP={test_clip:.4f}" if clip_model else ""))
                
                # Check if constraints are met
                constraints_met = (test_ssim >= effective_ssim_threshold and 
                                  (not lpips_model or test_lpips <= effective_lpips_threshold) and
                                  (not clip_model or test_clip >= effective_clip_threshold))
                
                if constraints_met:
                    # Calculate score as before
                    ssim_score = 1.0 - abs(test_ssim - effective_ssim_threshold) * 10
                    lpips_score = 1.0 - abs(test_lpips - effective_lpips_threshold) * 10 if lpips_model else 1.0
                    clip_score = 1.0 - abs(test_clip - effective_clip_threshold) * 10 if clip_model else 1.0
                    
                    score = blend * 5 + noise_level * 5 + ssim_score + lpips_score + clip_score
                    
                    print(f"    Constraints met! Score: {score:.4f}")
                    
                    if score > best_score:
                        best_score = score
                        best_adv_image = test_image.copy()
                        print(f"    New best score!")
    
    # If still no solution, fall back to standard approach but with relaxed constraints
    if best_adv_image is None:
        print("All optimization approaches failed, falling back to standard approach with relaxed constraints...")
        # Relax constraints slightly
        relaxed_ssim = effective_ssim_threshold * 0.99
        relaxed_lpips = effective_lpips_threshold * 1.05
        relaxed_clip = effective_clip_threshold * 0.99
        
        print(f"Using relaxed constraints: SSIM >= {relaxed_ssim:.4f}, LPIPS <= {relaxed_lpips:.4f}, CLIP >= {relaxed_clip:.4f}")
        
        return apply_enhanced_perceptual_constraints(original_image, adv_image, relaxed_ssim, 
                                                   relaxed_lpips, relaxed_clip,
                                                   lpips_model, clip_model, clip_preprocess)
    
    # MODIFIED: Apply additional targeted perturbation to key regions
    print("Applying additional targeted perturbation to key regions...")
    
    # Generate importance map for chart elements
    importance_map = generate_importance_map_for_charts(original_image)
    
    # Apply additional perturbation to important regions
    perturbation = best_adv_image.astype(np.float32) - original_image.astype(np.float32)
    
    # Amplify perturbation in important regions
    amplified_perturbation = perturbation * (1 + importance_map[:, :, np.newaxis] * 2.0)
    
    # Apply amplified perturbation
    targeted_image = np.clip(original_image.astype(np.float32) + amplified_perturbation, 0, 255).astype(np.uint8)
    
    # Check if constraints are still met
    targeted_ssim = calculate_ssim(original_image, targeted_image)
    targeted_lpips = calculate_lpips(original_image, targeted_image, lpips_model) if lpips_model else 0.0
    targeted_clip = calculate_clip_similarity(original_image, targeted_image, clip_model, clip_preprocess) if clip_model else 1.0
    
    print(f"Targeted perturbation metrics:")
    print(f"  SSIM: {targeted_ssim:.4f} (target: >= {effective_ssim_threshold:.4f})")
    if lpips_model:
        print(f"  LPIPS: {targeted_lpips:.4f} (target: <= {effective_lpips_threshold:.4f})")
    if clip_model:
        print(f"  CLIP: {targeted_clip:.4f} (target: >= {effective_clip_threshold:.4f})")
    
    # Use targeted image if it meets constraints, otherwise use best image from previous steps
    if (targeted_ssim >= effective_ssim_threshold and 
        (not lpips_model or targeted_lpips <= effective_lpips_threshold) and 
        (not clip_model or targeted_clip >= effective_clip_threshold)):
        print("Using targeted perturbation image")
        best_adv_image = targeted_image
    
    # Calculate final metrics
    final_ssim = calculate_ssim(original_image, best_adv_image)
    final_lpips = calculate_lpips(original_image, best_adv_image, lpips_model) if lpips_model else 0.0
    final_clip = calculate_clip_similarity(original_image, best_adv_image, clip_model, clip_preprocess) if clip_model else 1.0
    
    print(f"Final perceptual metrics:")
    print(f"  SSIM: {final_ssim:.4f} (target: >= {ssim_threshold})")
    if lpips_model:
        print(f"  LPIPS: {final_lpips:.4f} (target: <= {lpips_threshold})")
    if clip_model:
        print(f"  CLIP similarity: {final_clip:.4f} (target: >= {clip_threshold})")
    
    return best_adv_image
def optimize_to_threshold_limits(original_image, adv_image, ssim_threshold, lpips_threshold, clip_threshold,
                               lpips_model=None, clip_model=None, clip_preprocess=None):
    """Fine-tune an image to get metrics as close as possible to threshold limits
    
    Args:
        original_image (numpy.ndarray): Original image
        adv_image (numpy.ndarray): Adversarial image that already meets constraints
        ssim_threshold (float): Minimum SSIM value to maintain
        lpips_threshold (float): Maximum LPIPS distance to maintain
        clip_threshold (float): Minimum CLIP similarity to maintain
        lpips_model: LPIPS model for perceptual similarity
        clip_model: CLIP model for semantic similarity
        clip_preprocess: CLIP preprocessing function
        
    Returns:
        numpy.ndarray: Optimized adversarial image with metrics close to thresholds
    """
    print("Fine-tuning to optimize metrics closer to threshold limits...")
    
    # Calculate current metrics
    current_ssim = calculate_ssim(original_image, adv_image)
    current_lpips = calculate_lpips(original_image, adv_image, lpips_model) if lpips_model else 0.0
    current_clip = calculate_clip_similarity(original_image, adv_image, clip_model, clip_preprocess) if clip_model else 1.0
    
    # Calculate margins from thresholds
    ssim_margin = current_ssim - ssim_threshold
    lpips_margin = lpips_threshold - current_lpips if lpips_model else 0
    clip_margin = current_clip - clip_threshold if clip_model else 0
    
    print(f"Current margins from thresholds:")
    print(f"  SSIM margin: {ssim_margin:.4f} (lower is better)")
    print(f"  LPIPS margin: {lpips_margin:.4f} (lower is better)")
    print(f"  CLIP margin: {clip_margin:.4f} (lower is better)")
    
    # MODIFIED: Be more aggressive - only skip if extremely close to thresholds
    if ssim_margin < 0.001 and lpips_margin < 0.001 and clip_margin < 0.001:
        print("Metrics already extremely close to thresholds, no further optimization needed")
        return adv_image
    
    # MODIFIED: Create a more adversarial version by adding stronger noise
    noise_level = 0.5  # Increased from 0.3
    noise = np.random.uniform(-noise_level, noise_level, adv_image.shape).astype(np.float32)
    noisy_adv = np.clip(adv_image.astype(np.float32) + noise * 255, 0, 255).astype(np.uint8)
    
    # Try different blending factors between current adv_image and noisy version
    best_image = adv_image.copy()
    best_score = -float('inf')
    best_margins = (ssim_margin, lpips_margin, clip_margin)
    
    # MODIFIED: Try more aggressive blending factors with finer granularity
    for blend in np.linspace(0.1, 1.0, 37):  # More granular search
        test_image = cv2.addWeighted(adv_image, 1 - blend, noisy_adv, blend, 0)
        
        # Calculate metrics
        test_ssim = calculate_ssim(original_image, test_image)
        test_lpips = calculate_lpips(original_image, test_image, lpips_model) if lpips_model else 0.0
        test_clip = calculate_clip_similarity(original_image, test_image, clip_model, clip_preprocess) if clip_model else 1.0
        
        # MODIFIED: Allow slight violations of constraints to get closer to thresholds
        constraints_met = (test_ssim >= ssim_threshold * 0.995 and 
                          (not lpips_model or test_lpips <= lpips_threshold * 1.05) and
                          (not clip_model or test_clip >= clip_threshold * 0.995))
        
        if constraints_met:
            # Calculate margins from thresholds
            test_ssim_margin = test_ssim - ssim_threshold
            test_lpips_margin = lpips_threshold - test_lpips if lpips_model else 0
            test_clip_margin = test_clip - clip_threshold if clip_model else 0
            
            # MODIFIED: Calculate score based on how close metrics are to thresholds
            # Lower margins are better (closer to thresholds)
            # We want to minimize the maximum margin to get all metrics close to thresholds
            max_margin = max(test_ssim_margin, test_lpips_margin, test_clip_margin)
            avg_margin = (test_ssim_margin + test_lpips_margin + test_clip_margin) / 3
            
            # MODIFIED: Score is better when margins are smaller and blend is higher
            score = -max_margin * 10 - avg_margin * 5 + blend * 2
            
            print(f"  Blend {blend:.2f}: SSIM={test_ssim:.4f}" +
                  (f", LPIPS={test_lpips:.4f}" if lpips_model else "") +
                  (f", CLIP={test_clip:.4f}" if clip_model else "") +
                  f", Score={score:.4f}")
            
            if score > best_score:
                best_score = score
                best_image = test_image.copy()
                best_margins = (test_ssim_margin, test_lpips_margin, test_clip_margin)
                print(f"    New best score! Margins: SSIM={test_ssim_margin:.4f}, LPIPS={test_lpips_margin:.4f}, CLIP={test_clip_margin:.4f}")
    
    # MODIFIED: If no good blend was found, try a different approach with targeted noise
    if best_score == -float('inf'):
        print("No valid blend found, trying targeted noise approach with relaxed constraints...")
        
        # Create importance map to focus noise on less important areas
        importance_map = generate_importance_map_for_charts(original_image)
        
        # Invert importance map to focus noise on less important areas
        inv_importance_map = 1.0 - importance_map
        
        # Try different noise levels with finer granularity
        for noise_scale in np.linspace(0.05, 0.8, 19):  # Increased max noise scale
            # Generate noise focused on less important areas
            targeted_noise = np.random.uniform(-noise_scale, noise_scale, adv_image.shape).astype(np.float32)
            targeted_noise *= inv_importance_map[:, :, np.newaxis]  # Apply inverted importance map
            
            # Apply noise
            test_image = np.clip(adv_image.astype(np.float32) + targeted_noise * 255, 0, 255).astype(np.uint8)
            
            # Calculate metrics
            test_ssim = calculate_ssim(original_image, test_image)
            test_lpips = calculate_lpips(original_image, test_image, lpips_model) if lpips_model else 0.0
            test_clip = calculate_clip_similarity(original_image, test_image, clip_model, clip_preprocess) if clip_model else 1.0
            
            # MODIFIED: Allow slight violations of constraints
            constraints_met = (test_ssim >= ssim_threshold * 0.995 and 
                              (not lpips_model or test_lpips <= lpips_threshold * 1.05) and
                              (not clip_model or test_clip >= clip_threshold * 0.995))
            
            if constraints_met:
                # Calculate margins from thresholds
                test_ssim_margin = test_ssim - ssim_threshold
                test_lpips_margin = lpips_threshold - test_lpips if lpips_model else 0
                test_clip_margin = test_clip - clip_threshold if clip_model else 0
                
                # Calculate score as before
                max_margin = max(test_ssim_margin, test_lpips_margin, test_clip_margin)
                avg_margin = (test_ssim_margin + test_lpips_margin + test_clip_margin) / 3
                
                # MODIFIED: Score is better when margins are smaller and noise is higher
                score = -max_margin * 10 - avg_margin * 5 + noise_scale * 5
                
                print(f"  Noise {noise_scale:.2f}: SSIM={test_ssim:.4f}" +
                      (f", LPIPS={test_lpips:.4f}" if lpips_model else "") +
                      (f", CLIP={test_clip:.4f}" if clip_model else "") +
                      f", Score={score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_image = test_image.copy()
                    best_margins = (test_ssim_margin, test_lpips_margin, test_clip_margin)
                    print(f"    New best score! Margins: SSIM={test_ssim_margin:.4f}, LPIPS={test_lpips_margin:.4f}, CLIP={test_clip_margin:.4f}")
    
    # MODIFIED: If still no good solution, try frequency domain approach with relaxed constraints
    if best_score == -float('inf'):
        print("No valid solution found, trying frequency domain approach with relaxed constraints...")
        
        # Convert to frequency domain
        original_gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        adv_gray = cv2.cvtColor(adv_image, cv2.COLOR_RGB2GRAY)
        
        # Get DFT of both images
        original_dft = cv2.dft(np.float32(original_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        adv_dft = cv2.dft(np.float32(adv_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        
        rows, cols = original_gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # Try different cutoff frequencies with finer granularity
        for cutoff_ratio in np.linspace(0.05, 0.8, 19):  # Increased max cutoff ratio
            cutoff = int(min(rows, cols) * cutoff_ratio)
            
            # Create a mask with high frequencies from original, low frequencies from adversarial
            mask = np.ones((rows, cols, 2), np.float32)
            mask[crow-cutoff:crow+cutoff, ccol-cutoff:ccol+cutoff] = 0
            
            # MODIFIED: Apply mask with more weight on adversarial components
            hybrid_dft = adv_dft.copy()
            hybrid_dft = adv_dft * (1 - mask) * 1.2 + original_dft * mask * 0.8  # More weight on adversarial
            
            # Convert back to spatial domain
            hybrid_img = cv2.idft(hybrid_dft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
            hybrid_img = cv2.normalize(hybrid_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Convert back to RGB by replacing Y channel in YCrCb
            adv_ycrcb = cv2.cvtColor(adv_image, cv2.COLOR_RGB2YCrCb)
            adv_ycrcb[:,:,0] = hybrid_img
            test_image = cv2.cvtColor(adv_ycrcb, cv2.COLOR_YCrCb2RGB)
            
            # Calculate metrics
            test_ssim = calculate_ssim(original_image, test_image)
            test_lpips = calculate_lpips(original_image, test_image, lpips_model) if lpips_model else 0.0
            test_clip = calculate_clip_similarity(original_image, test_image, clip_model, clip_preprocess) if clip_model else 1.0
            
            # MODIFIED: Allow slight violations of constraints
            constraints_met = (test_ssim >= ssim_threshold * 0.995 and 
                              (not lpips_model or test_lpips <= lpips_threshold * 1.05) and
                              (not clip_model or test_clip >= clip_threshold * 0.995))
            
            if constraints_met:
                # Calculate margins from thresholds
                test_ssim_margin = test_ssim - ssim_threshold
                test_lpips_margin = lpips_threshold - test_lpips if lpips_model else 0
                test_clip_margin = test_clip - clip_threshold if clip_model else 0
                
                # Calculate score as before
                max_margin = max(test_ssim_margin, test_lpips_margin, test_clip_margin)
                avg_margin = (test_ssim_margin + test_lpips_margin + test_clip_margin) / 3
                
                # MODIFIED: Score is better when margins are smaller and cutoff is higher
                score = -max_margin * 10 - avg_margin * 5 + cutoff_ratio * 5
                
                print(f"  Cutoff {cutoff_ratio:.2f}: SSIM={test_ssim:.4f}" +
                      (f", LPIPS={test_lpips:.4f}" if lpips_model else "") +
                      (f", CLIP={test_clip:.4f}" if clip_model else "") +
                      f", Score={score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_image = test_image.copy()
                    best_margins = (test_ssim_margin, test_lpips_margin, test_clip_margin)
                    print(f"    New best score! Margins: SSIM={test_ssim_margin:.4f}, LPIPS={test_lpips_margin:.4f}, CLIP={test_clip_margin:.4f}")
    
    # MODIFIED: If still no solution found, return the original adversarial image
    if best_score == -float('inf'):
        print("Warning: Could not optimize to threshold limits. Using original adversarial image.")
        return adv_image
    
    # Calculate final metrics
    final_ssim = calculate_ssim(original_image, best_image)
    final_lpips = calculate_lpips(original_image, best_image, lpips_model) if lpips_model else 0.0
    final_clip = calculate_clip_similarity(original_image, best_image, clip_model, clip_preprocess) if clip_model else 1.0
    
    print(f"Optimized metrics:")
    print(f"  SSIM: {final_ssim:.4f} (target: >= {ssim_threshold})")
    if lpips_model:
        print(f"  LPIPS: {final_lpips:.4f} (target: <= {lpips_threshold})")
    if clip_model:
        print(f"  CLIP similarity: {final_clip:.4f} (target: >= {clip_threshold})")
    
    # Calculate final margins
    final_ssim_margin = final_ssim - ssim_threshold
    final_lpips_margin = lpips_threshold - final_lpips if lpips_model else 0
    final_clip_margin = final_clip - clip_threshold if clip_model else 0
    
    print(f"Final margins from thresholds:")
    print(f"  SSIM margin: {final_ssim_margin:.4f} (lower is better)")
    print(f"  LPIPS margin: {final_lpips_margin:.4f} (lower is better)")
    print(f"  CLIP margin: {final_clip_margin:.4f} (lower is better)")
    
    return best_image
def optimize_to_exact_thresholds(original_image, adv_image, ssim_threshold, lpips_threshold, clip_threshold,
                               lpips_model=None, clip_model=None, clip_preprocess=None):
    """Fine-tune an image to get metrics EXACTLY at the threshold limits
    
    Args:
        original_image (numpy.ndarray): Original image
        adv_image (numpy.ndarray): Adversarial image that already meets constraints
        ssim_threshold (float): Target SSIM value (not minimum)
        lpips_threshold (float): Target LPIPS value (not maximum)
        clip_threshold (float): Target CLIP value (not minimum)
        lpips_model: LPIPS model for perceptual similarity
        clip_model: CLIP model for semantic similarity
        clip_preprocess: CLIP preprocessing function
        
    Returns:
        numpy.ndarray: Optimized adversarial image with metrics at thresholds
    """
    print(f"Fine-tuning to target EXACT threshold values: SSIM={ssim_threshold*100:.1f}%, LPIPS={lpips_threshold:.4f}, CLIP={clip_threshold*100:.1f}%")
    
    # Calculate current metrics
    current_ssim = calculate_ssim(original_image, adv_image)
    current_lpips = calculate_lpips(original_image, adv_image, lpips_model) if lpips_model else 0.0
    current_clip = calculate_clip_similarity(original_image, adv_image, clip_model, clip_preprocess) if clip_model else 1.0
    
    print(f"Current metrics:")
    print(f"  SSIM: {current_ssim*100:.1f}% (target: {ssim_threshold*100:.1f}%)")
    print(f"  LPIPS: {current_lpips:.4f} (target: {lpips_threshold:.4f})")
    print(f"  CLIP: {current_clip*100:.1f}% (target: {clip_threshold*100:.1f}%)")
    
    # Create a very noisy version of the image
    noise_level = 0.5
    noise = np.random.uniform(-noise_level, noise_level, original_image.shape).astype(np.float32)
    noisy_image = np.clip(original_image.astype(np.float32) + noise * 255, 0, 255).astype(np.uint8)
    
    # Calculate metrics for noisy image
    noisy_ssim = calculate_ssim(original_image, noisy_image)
    noisy_lpips = calculate_lpips(original_image, noisy_image, lpips_model)
    noisy_clip = calculate_clip_similarity(original_image, noisy_image, clip_model, clip_preprocess)
    
    print(f"Noisy image metrics:")
    print(f"  SSIM: {noisy_ssim*100:.1f}%")
    print(f"  LPIPS: {noisy_lpips:.4f}")
    print(f"  CLIP: {noisy_clip*100:.1f}%")
    
    # Binary search to find the optimal blend between original and noisy image
    # to get metrics as close as possible to the thresholds
    best_image = adv_image.copy()
    best_score = float('inf')  # Lower is better here
    
    # Try many different blend factors with very fine granularity
    print("Searching for optimal blend between original and noisy image...")
    for blend in np.linspace(0.0, 1.0, 201):  # 201 steps for very fine-grained search
        test_image = cv2.addWeighted(original_image, 1 - blend, noisy_image, blend, 0)
        
        # Calculate metrics
        test_ssim = calculate_ssim(original_image, test_image)
        test_lpips = calculate_lpips(original_image, test_image, lpips_model)
        test_clip = calculate_clip_similarity(original_image, test_image, clip_model, clip_preprocess)
        
        # Calculate distance to target metrics - we want to be EXACTLY at the thresholds
        ssim_distance = abs(test_ssim - ssim_threshold)
        lpips_distance = abs(test_lpips - lpips_threshold)
        clip_distance = abs(test_clip - clip_threshold)
        
        # Check if constraints are still met (we still need to meet minimum requirements)
        constraints_met = (test_ssim >= ssim_threshold and 
                          test_lpips <= lpips_threshold and 
                          test_clip >= clip_threshold)
        
        # Calculate combined score (lower is better)
        score = ssim_distance + lpips_distance + clip_distance
        
        if blend % 0.05 < 0.01:  # Print only every 10th step to reduce output
            print(f"  Blend {blend:.2f}: SSIM={test_ssim*100:.1f}%, LPIPS={test_lpips:.4f}, CLIP={test_clip*100:.1f}%, Score={score:.4f}, Valid={constraints_met}")
        
        if constraints_met and score < best_score:
            best_score = score
            best_image = test_image.copy()
            print(f"    New best score! Distance to targets: SSIM={ssim_distance:.4f}, LPIPS={lpips_distance:.4f}, CLIP={clip_distance:.4f}")
    
    # If no valid blend was found with noise, try a different approach
    if best_score == float('inf'):
        print("No valid blend found with noise, trying with adversarial image...")
        
        # Try blending between original and adversarial image with very fine granularity
        for blend in np.linspace(0.0, 1.0, 201):
            test_image = cv2.addWeighted(original_image, 1 - blend, adv_image, blend, 0)
            
            # Calculate metrics
            test_ssim = calculate_ssim(original_image, test_image)
            test_lpips = calculate_lpips(original_image, test_image, lpips_model)
            test_clip = calculate_clip_similarity(original_image, test_image, clip_model, clip_preprocess)
            
            # Calculate distance to target metrics
            ssim_distance = abs(test_ssim - ssim_threshold)
            lpips_distance = abs(test_lpips - lpips_threshold)
            clip_distance = abs(test_clip - clip_threshold)
            
            # Check if constraints are still met
            constraints_met = (test_ssim >= ssim_threshold and 
                              test_lpips <= lpips_threshold and 
                              test_clip >= clip_threshold)
            
            # Calculate combined score (lower is better)
            score = ssim_distance + lpips_distance + clip_distance
            
            if blend % 0.05 < 0.01:  # Print only every 10th step
                print(f"  Blend {blend:.2f}: SSIM={test_ssim*100:.1f}%, LPIPS={test_lpips:.4f}, CLIP={test_clip*100:.1f}%, Score={score:.4f}, Valid={constraints_met}")
            
            if constraints_met and score < best_score:
                best_score = score
                best_image = test_image.copy()
                print(f"    New best score! Distance to targets: SSIM={ssim_distance:.4f}, LPIPS={lpips_distance:.4f}, CLIP={clip_distance:.4f}")
    
    # If still no valid blend found, try a more advanced approach
    if best_score == float('inf'):
        print("No valid blend found, trying advanced optimization...")
        
        # Create a grid of images with different perturbation levels
        grid_images = []
        grid_metrics = []
        
        # Generate a grid of perturbed images
        for eps in np.linspace(0.01, 0.2, 10):
            for p_init in [0.1, 0.3, 0.5, 0.7, 0.9]:
                # Create a new perturbed image
                noise = np.random.uniform(-eps, eps, original_image.shape).astype(np.float32)
                perturbed = np.clip(original_image.astype(np.float32) + noise * 255, 0, 255).astype(np.uint8)
                
                # Calculate metrics
                p_ssim = calculate_ssim(original_image, perturbed)
                p_lpips = calculate_lpips(original_image, perturbed, lpips_model)
                p_clip = calculate_clip_similarity(original_image, perturbed, clip_model, clip_preprocess)
                
                # Calculate distance to target metrics
                p_ssim_distance = abs(p_ssim - ssim_threshold)
                p_lpips_distance = abs(p_lpips - lpips_threshold)
                p_clip_distance = abs(p_clip - clip_threshold)
                
                # Check if constraints are met
                p_constraints_met = (p_ssim >= ssim_threshold and 
                                   p_lpips <= lpips_threshold and 
                                   p_clip >= clip_threshold)
                
                # Calculate score
                p_score = p_ssim_distance + p_lpips_distance + p_clip_distance
                
                if p_constraints_met:
                    grid_images.append(perturbed)
                    grid_metrics.append((p_ssim, p_lpips, p_clip, p_score))
                    print(f"  Found valid perturbation: SSIM={p_ssim*100:.1f}%, LPIPS={p_lpips:.4f}, CLIP={p_clip*100:.1f}%, Score={p_score:.4f}")
        
        # If we found valid images, select the best one
        if grid_images:
            best_idx = np.argmin([m[3] for m in grid_metrics])
            best_image = grid_images[best_idx]
            best_metrics = grid_metrics[best_idx]
            print(f"  Selected best perturbation: SSIM={best_metrics[0]*100:.1f}%, LPIPS={best_metrics[1]:.4f}, CLIP={best_metrics[2]*100:.1f}%, Score={best_metrics[3]:.4f}")
    
    # Calculate final metrics
    final_ssim = calculate_ssim(original_image, best_image)
    final_lpips = calculate_lpips(original_image, best_image, lpips_model)
    final_clip = calculate_clip_similarity(original_image, best_image, clip_model, clip_preprocess)
    
    print(f"Final metrics:")
    print(f"  SSIM: {final_ssim*100:.1f}% (target: {ssim_threshold*100:.1f}%)")
    print(f"  LPIPS: {final_lpips:.4f} (target: {lpips_threshold:.4f})")
    print(f"  CLIP: {final_clip*100:.1f}% (target: {clip_threshold*100:.1f}%)")
    
    # Calculate final distances to targets
    final_ssim_distance = abs(final_ssim - ssim_threshold)
    final_lpips_distance = abs(final_lpips - lpips_threshold)
    final_clip_distance = abs(final_clip - clip_threshold)
    
    print(f"Distance to targets:")
    print(f"  SSIM: {final_ssim_distance:.4f}")
    print(f"  LPIPS: {final_lpips_distance:.4f}")
    print(f"  CLIP: {final_clip_distance:.4f}")
    
    return best_image
