"""
Implementation of Microsoft Florence-2 models (base 0.23B, large 0.77B) with 4-bit quantization
for memory-efficient inference on consumer GPUs.
"""

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
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
import re

class Florence2ModelWrapper(BaseVLModel):
    """
    Wrapper class for Microsoft Florence-2 models (base 0.23B, large 0.77B) with optimized settings
    for memory-efficient inference on consumer GPUs.
    """
    
    def __init__(self, model_name):
        super().__init__(model_name)
        self.device = get_device()
        
        # Parse model size from name and set appropriate paths and configurations
        if "base" in model_name.lower():
            self.model_path = "microsoft/Florence-2-base"
            self.model_size = "base"
            self.max_gpu_memory = "2GiB"  # Smaller model needs less memory
            self.use_4bit = False  # Small enough to run in float16
            self.dtype = torch.float16
        elif "large" in model_name.lower():
            self.model_path = "microsoft/Florence-2-large"
            self.model_size = "large"
            self.max_gpu_memory = "3GiB"
            self.use_4bit = False  # Can still run in float16
            self.dtype = torch.float16
        else:
            raise ValueError(f"Unknown Florence-2 model size in name: {model_name}")
        
        # Aggressive memory cleanup before loading
        cleanup_memory()
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"Loading Florence-2-{self.model_size} processor...")
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
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
            print(f"Loading Florence-2-{self.model_size} model...")
            
            # Load model with appropriate settings
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map="auto",
                max_memory={0: self.max_gpu_memory, "cpu": "16GiB"},
            )
            
            self.model_loaded = True
            
            # Record end time and calculate duration
            end_time = time.time()
            duration = end_time - start_time
            
            # Measure memory after loading
            print("Memory after model loading:")
            self._print_memory_usage()
            print(f"Model loaded in {duration:.2f} seconds")
            
        except Exception as e:
            print(f"Error loading Florence-2-{self.model_size} model: {e}")
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
            
            # For chart analysis, we'll use the DETAILED_CAPTION task
            # This works better for charts than VQA
            task_prompt = "<DETAILED_CAPTION>"
            
            # Measure memory before inference
            print("Memory before inference:")
            self._print_memory_usage()
            
            # Process inputs with memory-efficient settings
            with torch.inference_mode():  # Use inference_mode to save memory
                # Clear cache before heavy operations
                torch.cuda.empty_cache()
                
                # Prepare inputs
                inputs = self.processor(
                    text=task_prompt, 
                    images=image, 
                    return_tensors="pt"
                ).to(self.device, self.dtype)
                
                # Clear cache again before generation
                torch.cuda.empty_cache()
                
                # Generate response
                print(f"Generating response with Florence-2-{self.model_size}...")
                
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=256,
                    num_beams=3,
                    do_sample=False,  # Deterministic to save memory
                )
                
                # Decode and post-process
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                parsed_answer = self.processor.post_process_generation(
                    generated_text, 
                    task=task_prompt, 
                    image_size=(image.width, image.height)
                )
            
            # Measure memory after inference
            print("Memory after inference:")
            self._print_memory_usage()
            
            # Extract the actual answer from the parsed result
            if isinstance(parsed_answer, dict) and task_prompt in parsed_answer:
                result = parsed_answer[task_prompt]
                
                # If the result doesn't seem to answer the question, try a different approach
                if len(result.split()) < 5:  # Very short answer
                    # Try a different task prompt
                    with torch.inference_mode():
                        # Try with VQA task
                        vqa_prompt = f"<VQA> {question}"
                        inputs = self.processor(
                            text=vqa_prompt, 
                            images=image, 
                            return_tensors="pt"
                        ).to(self.device, self.dtype)
                        
                        generated_ids = self.model.generate(
                            input_ids=inputs["input_ids"],
                            pixel_values=inputs["pixel_values"],
                            max_new_tokens=256,
                            num_beams=3,
                        )
                        
                        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                        
                        # Extract answer part (after the prompt)
                        match = re.search(r'<VQA>\s*(.*?)(?:\s*<|$)', generated_text)
                        if match:
                            vqa_result = match.group(1).strip()
                            if len(vqa_result.split()) > 3:  # If we got a reasonable answer
                                return vqa_result
                
                return result
            else:
                # Try to extract meaningful content from the raw output
                if isinstance(parsed_answer, str):
                    # Remove any task prompts from the output
                    cleaned = re.sub(r'<[A-Z_]+>', '', parsed_answer).strip()
                    if cleaned:
                        return cleaned
                
                # If all else fails, return the raw parsed answer
                return str(parsed_answer)
            
        except Exception as e:
            print(f"Error in Florence-2-{self.model_size} prediction: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    def cleanup(self):
        """Clean up GPU resources"""
        print(f"Cleaning up Florence-2-{self.model_size} resources...")
        
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
