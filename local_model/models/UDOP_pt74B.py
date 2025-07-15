"""
Implementation of Microsoft UDOP-large model (742M) with optimized settings
for document understanding tasks.
"""

import torch
from transformers import AutoProcessor, UdopForConditionalGeneration
from local_model.base_model import BaseVLModel
from local_model.model_utils import (
    cleanup_memory, 
    get_device, 
    time_inference,
    model_cleanup
)
import time
import psutil
import os
import gc
from PIL import Image
import numpy as np

class UDOPModelWrapper(BaseVLModel):
    """
    Wrapper class for Microsoft UDOP-large model (742M) with optimized settings
    for document understanding tasks.
    """
    
    def __init__(self, model_name):
        super().__init__(model_name)
        self.device = get_device()
        
        # Parse model size from name and set appropriate paths and configurations
        if "udop-large" in model_name.lower():
            self.model_path = "microsoft/udop-large"
            self.model_size = "large"
            self.max_gpu_memory = "2GiB"  # Small model needs less memory
            self.dtype = torch.float32  # UDOP uses F32
        else:
            raise ValueError(f"Unknown UDOP model size in name: {model_name}")
        
        # Aggressive memory cleanup before loading
        cleanup_memory()
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"Loading UDOP-{self.model_size} processor...")
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                apply_ocr=False  # We'll handle OCR separately
            )
            print(f"Processor loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"Error loading processor: {e}")
            raise
        
        # Measure memory before loading
        print("Memory before model loading:")
        self._print_memory_usage()
        
        # Record start time
        start_time = time.time()
        
        try:
            print(f"Loading UDOP-{self.model_size} model...")
            
            # Load model with appropriate settings - avoid using device_map="auto"
            # which is causing issues with this model
            self.model = UdopForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True
            ).to(self.device)
            
            self.model_loaded = True
            
            # Record end time and calculate duration
            end_time = time.time()
            duration = end_time - start_time
            
            # Measure memory after loading
            print("Memory after model loading:")
            self._print_memory_usage()
            print(f"Model loaded in {duration:.2f} seconds")
            
        except Exception as e:
            print(f"Error loading UDOP-{self.model_size} model: {e}")
            import traceback
            traceback.print_exc()
            self.model_loaded = False
    
    def _print_memory_usage(self):
        """Print current memory usage"""
        try:
            # CPU memory
            process = psutil.Process(os.getpid())
            cpu_memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB
            
            # GPU memory if available
            gpu_memory = 0
            gpu_memory_reserved = 0
            gpu_memory_total = 0
            gpu_memory_peak = 0
            
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # Convert to MB
                gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 * 1024)  # Convert to MB
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)  # Convert to MB
                gpu_memory_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)  # Convert to MB
            
            print(f"GPU Memory: {gpu_memory:.2f} MB (Current) / {gpu_memory_peak:.2f} MB (Peak) / {gpu_memory_total:.2f} MB (Total)")
            print(f"GPU Reserved: {gpu_memory_reserved:.2f} MB")
            print(f"CPU Memory: {cpu_memory:.2f} MB")
            
        except Exception as e:
            print(f"Error getting memory usage: {e}")
    
    def _generate_dummy_ocr_data(self, image):
        """
        Generate dummy OCR data for the image.
        UDOP requires words and bounding boxes, so we create a simple grid.
        """
        # Create a grid of dummy words and boxes
        width, height = image.size
        grid_size = 5  # 5x5 grid
        words = []
        boxes = []
        
        # Generate grid cells
        for i in range(grid_size):
            for j in range(grid_size):
                # Create a dummy word for each cell
                word = f"cell_{i}_{j}"
                words.append(word)
                
                # Create a bounding box for each cell
                x1 = j * (width / grid_size)
                y1 = i * (height / grid_size)
                x2 = (j + 1) * (width / grid_size)
                y2 = (i + 1) * (height / grid_size)
                
                # Normalize coordinates to [0, 1]
                box = [x1 / width, y1 / height, x2 / width, y2 / height]
                boxes.append(box)
        
        return words, boxes
    
    @time_inference
    def predict(self, image_path, question):
        """Process a document image and question to generate an answer"""
        if not hasattr(self, 'model_loaded') or not self.model_loaded:
            return "Error: Model failed to load. Cannot perform prediction."
            
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Generate dummy OCR data (words and boxes)
            # UDOP requires this information for processing
            words, boxes = self._generate_dummy_ocr_data(image)
            
            # Measure memory before inference
            print("Memory before inference:")
            self._print_memory_usage()
            
            # Process inputs with memory-efficient settings
            with torch.inference_mode():  # Use inference_mode to save memory
                # Clear cache before heavy operations
                torch.cuda.empty_cache()
                
                # Prepare inputs for UDOP
                # UDOP requires words and boxes for document understanding
                inputs = self.processor(
                    image=image,
                    text=question,
                    words=words,
                    boxes=boxes,
                    return_tensors="pt"
                ).to(self.device)
                
                # Clear cache again before generation
                torch.cuda.empty_cache()
                
                # Generate response
                print(f"Generating response with UDOP-{self.model_size}...")
                
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    num_beams=2,
                    do_sample=False,  # Deterministic to save memory
                )
                
                # Decode the generated tokens
                output_text = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]
            
            # Measure memory after inference
            print("Memory after inference:")
            self._print_memory_usage()
            
            return output_text
            
        except Exception as e:
            print(f"Error in UDOP-{self.model_size} prediction: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    def cleanup(self):
        """Clean up GPU resources"""
        print(f"Cleaning up UDOP-{self.model_size} resources...")
        
        if hasattr(self, 'model') and self.model is not None:
            # Explicitly move model to CPU before deletion
            try:
                self.model.to('cpu')
            except:
                pass
                
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Delete model
            del self.model
            self.model = None
            
            # Force garbage collection
            gc.collect()
            
        print(f"{self.model_name} resources cleaned up")
