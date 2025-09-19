#!/usr/bin/env python3
"""
Dynamic Error Handler for VLM Inference
Replaces hardcoded blacklist with runtime error detection and handling.
"""

import logging
import torch
import gc
from typing import Dict, Set, Optional

logger = logging.getLogger(__name__)

class DynamicErrorHandler:
    def __init__(self):
        self.failed_images: Set[str] = set()
        self.error_counts: Dict[str, int] = {}
        self.max_retries = 1  # Don't retry CUDA errors
    
    def is_cuda_error(self, error_msg: str) -> bool:
        """Check if error is a CUDA-related error"""
        cuda_error_patterns = [
            "CUDA error",
            "device-side assert",
            "CUDA_ERROR_ILLEGAL_ADDRESS",
            "CUDA_ERROR_LAUNCH_FAILED",
            "RuntimeError: CUDA"
        ]
        return any(pattern.lower() in str(error_msg).lower() for pattern in cuda_error_patterns)
    
    def should_skip_image(self, image_path: str) -> bool:
        """Check if image should be skipped based on previous failures"""
        return image_path in self.failed_images
    
    def handle_inference_error(self, image_path: str, question_id: str, error: Exception) -> str:
        """Handle inference errors and decide on retry/skip logic"""
        error_msg = str(error)
        
        # Track error count
        self.error_counts[image_path] = self.error_counts.get(image_path, 0) + 1
        
        if self.is_cuda_error(error_msg):
            # CUDA errors are fatal - add to failed images
            self.failed_images.add(image_path)
            logger.warning(f"CUDA error detected for {image_path}: {error_msg}")
            
            # Clear GPU memory after CUDA error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            return "ERROR: CUDA device-side assert - skipping image"
        
        # For other errors, allow limited retries
        if self.error_counts[image_path] >= self.max_retries:
            self.failed_images.add(image_path)
            return f"ERROR: Max retries exceeded - {error_msg}"
        
        return f"ERROR: {error_msg}"
    
    def log_skip(self, image_path: str, attack_name: str, question_id: str):
        """Log when an image is skipped"""
        error_count = self.error_counts.get(image_path, 0)
        print(f"🚫 Skipping problematic image: {image_path}")
        print(f"   Attack: {attack_name}")
        print(f"   Question ID: {question_id}")
        print(f"   Previous errors: {error_count}")

# Global instance
error_handler = DynamicErrorHandler()
