"""
Corrupted Images Blacklist
Identifies and manages images that cause CUDA device-side assert errors.
"""

# List of image paths that are known to cause CUDA errors
CORRUPTED_IMAGES = {
    "visual_puzzle/16.png": {
        "reason": "CUDA device-side assert triggered during tensor operations",
        "affected_attacks": ["all"],
        "error_count": 3,
        "discovered": "2025-07-26"
    },
    "visual_puzzle/17.png": {
        "reason": "CUDA device-side assert triggered during tensor operations", 
        "affected_attacks": ["all"],
        "error_count": 6,
        "discovered": "2025-07-26"
    }
}

# Question IDs that are associated with corrupted images
CORRUPTED_QUESTION_IDS = {
    "6bf97d05-2b26-4d6b-b123-fb0354437b7c": "visual_puzzle/16.png",
    "2da339d3-31d4-46e3-99d2-198ca2c834d8": "visual_puzzle/16.png", 
    "56faabd0-d8ad-4b20-b9ad-0808a0c9828c": "visual_puzzle/16.png",
    "c53f0da9-ea1c-4a57-9434-779708dabe14": "visual_puzzle/17.png",
    "39020f2b-e9d5-4e3e-a9e9-0d9fc4be52f2": "visual_puzzle/17.png",
    "58cc1866-731b-4f77-9a91-360451b2e989": "visual_puzzle/17.png",
    "c9376e54-7a6a-48c9-9a1d-e18976f74c23": "visual_puzzle/17.png",
    "473ac9d9-a8a6-4859-8216-beedace23f39": "visual_puzzle/17.png",
    "30b31a97-5532-4500-9c75-6b1b42a44911": "visual_puzzle/17.png"
}

def is_image_corrupted(image_path):
    """Check if an image is known to be corrupted"""
    return image_path in CORRUPTED_IMAGES

def is_question_corrupted(question_id):
    """Check if a question ID is associated with corrupted image"""
    return question_id in CORRUPTED_QUESTION_IDS

def get_corrupted_image_info(image_path):
    """Get information about a corrupted image"""
    return CORRUPTED_IMAGES.get(image_path, None)

def should_skip_image(image_path, attack_name=None):
    """
    Determine if an image should be skipped due to corruption
    
    Args:
        image_path (str): Path to the image
        attack_name (str): Name of the adversarial attack
        
    Returns:
        bool: True if image should be skipped
    """
    if not is_image_corrupted(image_path):
        return False
        
    info = CORRUPTED_IMAGES[image_path]
    affected_attacks = info.get("affected_attacks", [])
    
    # Skip if all attacks are affected or specific attack is affected
    return "all" in affected_attacks or (attack_name and attack_name in affected_attacks)

def log_corrupted_image_skip(image_path, attack_name, question_id):
    """Log when a corrupted image is skipped"""
    print(f"🚫 Skipping corrupted image: {image_path}")
    print(f"   Attack: {attack_name}")
    print(f"   Question ID: {question_id}")
    print(f"   Reason: Known to cause CUDA device-side assert errors")