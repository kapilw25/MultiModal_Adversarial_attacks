"""
Implementation of vikhyatk/moondream2 model (1.93B) with optimized settings
for chart analysis and visual question answering.
"""

import torch
from transformers import AutoModelForCausalLM
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

class Moondream2ModelWrapper(BaseVLModel):
    """
    Wrapper class for vikhyatk/moondream2 model (1.93B) with optimized settings
    for chart analysis and visual question answering.
    """
    
    def __init__(self, model_name):
        super().__init__(model_name)
        self.device = get_device()
        
        # Parse model size from name and set appropriate paths and configurations
        if "moondream2" in model_name.lower():
            self.model_path = "vikhyatk/moondream2"
            self.model_revision = "2025-06-21"
            self.max_gpu_memory = "4GiB"  # ~2B model needs moderate memory
            self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        else:
            raise ValueError(f"Unknown Moondream model version in name: {model_name}")
        
        # Aggressive memory cleanup before loading
        cleanup_memory()
        gc.collect()
        torch.cuda.empty_cache()
        
        # Measure memory before loading
        print("Memory before model loading:")
        self._print_memory_usage()
        
        # Record start time
        start_time = time.time()
        
        try:
            print(f"Loading {self.model_path} model...")
            
            # Load model with appropriate settings
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                revision=self.model_revision,
                trust_remote_code=True,
                device_map={"": self.device},
                torch_dtype=self.dtype
            )
            
            self.model_loaded = True
            
            # Record end time and calculate duration
            end_time = time.time()
            duration = end_time - start_time
            
            # Measure memory after loading
            print("Memory after model loading:")
            self._print_memory_usage()
            print(f"Model loaded in {duration:.2f} seconds")
            
            # Print model size
            param_count = sum(p.numel() for p in self.model.parameters()) / 1e9
            print(f"Model size: {param_count:.2f} billion parameters")
            
        except Exception as e:
            print(f"Error loading {self.model_path} model: {e}")
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
    
    @time_inference
    def predict(self, image_path, question):
        """Process an image and question to generate an answer"""
        if not hasattr(self, 'model_loaded') or not self.model_loaded:
            return "Error: Model failed to load. Cannot perform prediction."
            
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Measure memory before inference
            print("Memory before inference:")
            self._print_memory_usage()
            
            # Process inputs with memory-efficient settings
            with torch.inference_mode():  # Use inference_mode to save memory
                # Clear cache before heavy operations
                torch.cuda.empty_cache()
                
                # Use the model's query method for visual question answering
                print(f"Generating response with {self.model_path}...")
                
                # Get the answer using the model's query method
                result = self.model.query(image, question)
                answer = result["answer"]
                
                # If the answer is very short, try to get a more detailed response
                # using the caption method
                if len(answer.split()) < 5:
                    print("Answer is very short, trying caption method...")
                    caption = self.model.caption(image, length="normal")["caption"]
                    if len(caption) > len(answer):
                        answer = caption
            
            # Measure memory after inference
            print("Memory after inference:")
            self._print_memory_usage()
            
            return answer
            
        except Exception as e:
            print(f"Error in {self.model_path} prediction: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    def cleanup(self):
        """Clean up GPU resources"""
        print(f"Cleaning up {self.model_path} resources...")
        
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
