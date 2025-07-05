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
