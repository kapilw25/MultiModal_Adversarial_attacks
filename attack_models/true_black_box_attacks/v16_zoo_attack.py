#!/usr/bin/env python3
"""
Zeroth Order Optimization (ZOO) Attack Script for Vision-Language Models

This script applies the ZOO Attack with perceptual constraints to create
adversarial examples that are both effective and visually imperceptible to humans.
ZOO is a black-box attack that uses zeroth-order optimization to estimate gradients
and generate adversarial examples without requiring access to model gradients.

Usage:
    python v16_zoo_attack.py [--image_path PATH] [--confidence CONFIDENCE] 
                            [--learning_rate LEARNING_RATE] [--max_iter MAX_ITER]
                            [--binary_search_steps BINARY_SEARCH_STEPS] [--initial_const INITIAL_CONST]
                            [--abort_early] [--use_resize] [--use_importance]
                            [--nb_parallel NB_PARALLEL] [--variable_h VARIABLE_H]
                            [--targeted] [--target_class CLASS]
                            [--ssim_threshold THRESHOLD]

Example:
    source venv_MM/bin/activate && python attack_models/true_black_box_attacks/v16_zoo_attack.py --image_path data/test_extracted/chart/20231114102825506748.png --confidence 0.0 --learning_rate 1e-2 --max_iter 10 --binary_search_steps 1 --initial_const 1e-3 --nb_parallel 128 --variable_h 1e-4 --ssim_threshold 0.85
"""

import os
import cv2
import numpy as np
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# Import ZooAttack from ART
from art.attacks.evasion.zoo import ZooAttack
from art.estimators.classification import PyTorchClassifier

# Import utility functions
from v0_attack_utils import (
    load_image, create_classifier, save_image, 
    get_output_path, print_attack_info, preprocess_image_for_attack,
    calculate_ssim, generate_importance_map_for_charts,
    apply_threshold_optimized_constraints
)


def zoo_attack_with_perceptual_constraints(image, classifier, image_path=None, confidence=0.0, 
                                          learning_rate=1e-2, max_iter=10, binary_search_steps=1,
                                          initial_const=1e-3, abort_early=True, use_resize=True,
                                          use_importance=True, nb_parallel=128, variable_h=1e-4,
                                          targeted=False, target_class=None, ssim_threshold=0.85,
                                          importance_map=None):
    """Apply ZOO Attack with perceptual constraints"""
    
    # Generate importance map if not provided
    if importance_map is None and image is not None:
        print("Generating importance map for targeted perturbation...")
        importance_map = generate_importance_map_for_charts(image)
        
        # Visualize importance map
        if image_path is not None:
            importance_vis = (importance_map * 255).astype(np.uint8)
            importance_vis = cv2.applyColorMap(importance_vis, cv2.COLORMAP_JET)
            output_dir = os.path.dirname(get_output_path(image_path, 'zoo'))
            os.makedirs(output_dir, exist_ok=True)
            importance_path = os.path.join(output_dir, 'importance_map.png')
            cv2.imwrite(importance_path, importance_vis)
            print(f"Saved importance map to {importance_path}")
    
    # Create ZOO Attack
    attack = ZooAttack(
        classifier=classifier,
        confidence=confidence,
        targeted=targeted,
        learning_rate=learning_rate,
        max_iter=max_iter,
        binary_search_steps=binary_search_steps,
        initial_const=initial_const,
        abort_early=abort_early,
        use_resize=use_resize,
        use_importance=use_importance,
        nb_parallel=nb_parallel,
        batch_size=1,
        variable_h=variable_h,
        verbose=True
    )
    
    # Preprocess image using utility function
    img_tensor = preprocess_image_for_attack(image)
    
    # Generate adversarial example
    print(f"Generating adversarial example with ZOO Attack:")
    print(f"- Confidence: {confidence}")
    print(f"- Learning rate: {learning_rate}")
    print(f"- Max iterations: {max_iter}")
    print(f"- Binary search steps: {binary_search_steps}")
    print(f"- Initial const: {initial_const}")
    print(f"- Abort early: {abort_early}")
    print(f"- Use resize: {use_resize}")
    print(f"- Use importance: {use_importance}")
    print(f"- Parallel coordinates: {nb_parallel}")
    print(f"- Variable h: {variable_h}")
    print(f"- Targeted: {targeted}")
    print(f"- SSIM threshold: {ssim_threshold}")
    
    if targeted and target_class is not None:
        # For targeted attack, create one-hot encoded target
        target = np.zeros((1, classifier.nb_classes))
        target[0, target_class] = 1
        adv_image = attack.generate(x=img_tensor, y=target)
    else:
        # For untargeted attack
        adv_image = attack.generate(x=img_tensor)
    
    # Convert back to uint8 format
    adv_image = adv_image[0].transpose(1, 2, 0)
    adv_image = np.clip(adv_image, 0, 1) * 255
    adv_image = adv_image.astype(np.uint8)
    
    # Resize back to original dimensions
    adv_image = cv2.resize(adv_image, (image.shape[1], image.shape[0]))
    
    # Apply perceptual constraint
    current_ssim = calculate_ssim(image, adv_image)
    print(f"Initial SSIM: {current_ssim:.4f}")
    
    # Apply threshold-optimized constraints to get SSIM close to the threshold
    if current_ssim < ssim_threshold or abs(current_ssim - ssim_threshold) < 0.05:
        print(f"Applying perceptual constraints to achieve target SSIM of {ssim_threshold:.4f}...")
        adv_image = apply_threshold_optimized_constraints(
            image, adv_image, ssim_threshold=ssim_threshold
        )
        final_ssim = calculate_ssim(image, adv_image)
        print(f"Final SSIM: {final_ssim:.4f} (target: {ssim_threshold:.4f})")
    
    # Apply targeted perturbation using importance map
    if importance_map is not None:
        print("Applying targeted perturbation using importance map...")
        
        # Calculate perturbation
        perturbation = adv_image.astype(np.float32) - image.astype(np.float32)
        
        # Apply importance map to perturbation (amplify important regions)
        importance_map_3ch = np.stack([importance_map] * 3, axis=2)
        weighted_perturbation = perturbation * (1 + (importance_map_3ch * 2.0))
        
        # Apply weighted perturbation to original image
        targeted_adv_image = image.astype(np.float32) + weighted_perturbation
        targeted_adv_image = np.clip(targeted_adv_image, 0, 255).astype(np.uint8)
        
        # Check if targeted image still meets SSIM constraint
        targeted_ssim = calculate_ssim(image, targeted_adv_image)
        if targeted_ssim >= ssim_threshold:
            print(f"Using targeted perturbation (SSIM: {targeted_ssim:.4f})")
            adv_image = targeted_adv_image
        else:
            print(f"Targeted perturbation violates SSIM constraint (SSIM: {targeted_ssim:.4f}), using original adversarial image")
    
    return adv_image


def main():
    parser = argparse.ArgumentParser(description="Generate adversarial examples using ZOO Attack with perceptual constraints")
    parser.add_argument("--image_path", type=str, 
                        default="data/test_extracted/chart/20231114102825506748.png",
                        help="Path to the input image")
    parser.add_argument("--confidence", type=float, default=0.0,
                        help="Confidence of adversarial examples (default: 0.0)")
    parser.add_argument("--learning_rate", type=float, default=1e-2,
                        help="Learning rate for the attack algorithm (default: 1e-2)")
    parser.add_argument("--max_iter", type=int, default=10,
                        help="Maximum number of iterations (default: 10)")
    parser.add_argument("--binary_search_steps", type=int, default=1,
                        help="Number of binary search steps (default: 1)")
    parser.add_argument("--initial_const", type=float, default=1e-3,
                        help="Initial trade-off constant (default: 1e-3)")
    parser.add_argument("--abort_early", action="store_true", default=True,
                        help="Abort early if gradient descent gets stuck")
    parser.add_argument("--use_resize", action="store_true", default=True,
                        help="Use the resizing strategy from the paper")
    parser.add_argument("--use_importance", action="store_true", default=True,
                        help="Use importance sampling when choosing coordinates to update")
    parser.add_argument("--nb_parallel", type=int, default=128,
                        help="Number of coordinate updates to run in parallel (default: 128)")
    parser.add_argument("--variable_h", type=float, default=1e-4,
                        help="Step size for numerical estimation of derivatives (default: 1e-4)")
    parser.add_argument("--targeted", action="store_true",
                        help="Use targeted attack instead of untargeted")
    parser.add_argument("--target_class", type=int, default=None,
                        help="Target class for targeted attack (default: None)")
    parser.add_argument("--ssim_threshold", type=float, default=0.85,
                        help="SSIM threshold for perceptual constraint (default: 0.85)")
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
    
    # Generate importance map for chart understanding
    print("Generating importance map for chart elements...")
    importance_map = generate_importance_map_for_charts(image)
    
    # Apply ZOO Attack with perceptual constraints
    adv_image = zoo_attack_with_perceptual_constraints(
        image, classifier, args.image_path, args.confidence, args.learning_rate,
        args.max_iter, args.binary_search_steps, args.initial_const, args.abort_early,
        args.use_resize, args.use_importance, args.nb_parallel, args.variable_h,
        args.targeted, args.target_class, args.ssim_threshold, importance_map
    )
    
    # Get output path using utility function
    output_path = get_output_path(args.image_path, 'zoo')
    
    # Save adversarial image
    save_image(adv_image, output_path)
    
    # Print attack information
    print_attack_info(output_path, image, adv_image, 'zoo')
    
    # Print additional attack-specific information
    perturbation = np.abs(image.astype(np.float32) - adv_image.astype(np.float32))
    print(f"Max perturbation: {np.max(perturbation)}")
    print(f"Mean perturbation: {np.mean(perturbation)}")
    print(f"SSIM: {calculate_ssim(image, adv_image):.4f}")
    print(f"Total queries used: approximately {args.max_iter * args.nb_parallel * 2}")


if __name__ == "__main__":
    main()
