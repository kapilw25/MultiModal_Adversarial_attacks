"""
Implementation of SmolVLM2 models (256M, 500M, 2.2B) with 4-bit quantization
for memory-efficient inference on consumer GPUs.
"""

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from local_model.base_model import BaseVLModel
from local_model.model_utils import (
    cleanup_memory, 
    get_device, 
    get_quantization_config,
    time_inference,
    model_cleanup
)
import time
import psutil
import os
import gc
from PIL import Image

class SmolVLM2ModelWrapper(BaseVLModel):
    """
    Wrapper class for SmolVLM2 models (256M, 500M, 2.2B) with optimized settings
    for memory-efficient inference on consumer GPUs.
    """
    
    def __init__(self, model_name):
        super().__init__(model_name)
        self.device = get_device()
        
        # Parse model size from name and set appropriate paths and configurations
        if "256M" in model_name:
            self.model_path = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
            self.model_size = "256M"
            self.max_gpu_memory = "7GiB"
            self.use_4bit = True  # Enable 4-bit quantization for speed
            self.dtype = torch.bfloat16
        elif "500M" in model_name:
            self.model_path = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
            self.model_size = "500M"
            self.max_gpu_memory = "7GiB"
            self.use_4bit = True  # Enable 4-bit quantization for speed
            self.dtype = torch.bfloat16
        elif "2.2B" in model_name:
            self.model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
            self.model_size = "2.2B"
            self.max_gpu_memory = "7GiB"
            self.use_4bit = True  # Enable 4-bit quantization for speed
            self.dtype = torch.bfloat16
        else:
            raise ValueError(f"Unknown SmolVLM2 model size in name: {model_name}")
        
        # Configure 4-bit quantization for all models
        print(f"Setting up 4-bit quantization for {model_name}...")
        self.quantization_config = get_quantization_config(
            load_in_4bit=True,
            compute_dtype=torch.bfloat16,
            use_double_quant=True,
            quant_type="nf4"
        )
        
        # Aggressive memory cleanup before loading
        cleanup_memory()
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"Loading SmolVLM2-{self.model_size} processor...")
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_path)
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
            print(f"Loading SmolVLM2-{self.model_size} model...")
            
            # Load all models with 4-bit quantization
            print(f"Loading SmolVLM2-{self.model_size} with 4-bit quantization...")
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                quantization_config=self.quantization_config,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                device_map="auto",
                max_memory={0: self.max_gpu_memory, "cpu": "16GiB"},
            )
            print(f"Successfully loaded model with 4-bit quantization")
            
            self.model_loaded = True
            
            # Record end time and calculate duration
            end_time = time.time()
            duration = end_time - start_time
            
            # Measure memory after loading
            print("Memory after model loading:")
            self._print_memory_usage()
            print(f"Model loaded in {duration:.2f} seconds")
            
        except Exception as e:
            print(f"Error loading SmolVLM2-{self.model_size} model: {e}")
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
            
            # Prepare messages format according to SmolVLM2's API
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            
            # Measure memory before inference
            print("Memory before inference:")
            self._print_memory_usage()
            
            # Process inputs with memory-efficient settings
            with torch.inference_mode():  # Use inference_mode to save memory
                # Clear cache before heavy operations
                torch.cuda.empty_cache()
                
                # Prepare inputs
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                
                # Move inputs to device with appropriate dtype to match model
                inputs = {k: v.to(self.device, dtype=self.dtype if torch.is_floating_point(v) else None) 
                         for k, v in inputs.items()}
                
                # Clear cache again before generation
                torch.cuda.empty_cache()
                
                # Generate response with memory-efficient settings
                print(f"Generating response with SmolVLM2-{self.model_size}...")
                
                # Adjust generation parameters based on model size
                max_tokens = 64
                if self.model_size == "256M":
                    # More conservative for smallest model
                    max_tokens = 48
                elif self.model_size == "2.2B":
                    # Can generate slightly more with largest model
                    max_tokens = 64
                
                generated_ids = self.model.generate(
                    **inputs,
                    do_sample=False,  # Deterministic to save memory
                    max_new_tokens=max_tokens,
                    num_beams=1,  # No beam search to save memory
                    use_cache=True,
                )
            
            # Process output
            print("Processing output...")
            output_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )
            
            # Measure memory after inference
            print("Memory after inference:")
            self._print_memory_usage()
            
            return output_text[0]
            
        except Exception as e:
            print(f"Error in SmolVLM2-{self.model_size} prediction: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    def cleanup(self):
        """Clean up GPU resources"""
        print(f"Cleaning up SmolVLM2-{self.model_size} resources...")
        
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
