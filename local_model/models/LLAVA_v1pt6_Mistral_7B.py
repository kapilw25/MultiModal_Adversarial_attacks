"""
Implementation of LLAVA v1.6 Mistral 7B model with 4-bit quantization
for memory-efficient inference on consumer GPUs.
"""

import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from local_model.base_model import BaseVLModel
from local_model.model_utils import (
    cleanup_memory, 
    get_device, 
    get_quantization_config,
    time_inference,
    model_cleanup,
    measure_memory_usage
)
import time
import gc
from PIL import Image

class LLAVAv16MistralModelWrapper(BaseVLModel):
    """
    Wrapper class for LLAVA v1.6 Mistral 7B model with optimized settings
    for memory-efficient inference on consumer GPUs.
    """
    
    def __init__(self, model_name):
        super().__init__(model_name)
        self.device = get_device()
        
        # Model configuration
        self.model_path = "llava-hf/llava-v1.6-mistral-7b-hf"
        self.model_size = "7B"
        self.max_gpu_memory = "7GiB"
        self.use_4bit = True  # Enable 4-bit quantization for speed
        self.dtype = torch.float16
        
        # Configure 4-bit quantization
        print(f"Setting up 4-bit quantization for {model_name}...")
        self.quantization_config = get_quantization_config(
            load_in_4bit=True,
            compute_dtype=torch.float16,
            use_double_quant=True,
            quant_type="nf4"
        )
        
        # Aggressive memory cleanup before loading
        cleanup_memory()
        
        print(f"Loading LLAVA v1.6 Mistral 7B processor...")
        try:
            # Use LlavaNextProcessor for v1.6 - following working script
            self.processor = LlavaNextProcessor.from_pretrained(self.model_path)
            print(f"Processor loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"Error loading processor: {e}")
            raise
        
        # Measure memory before loading
        print("Memory before model loading:")
        measure_memory_usage()
        
        # Record start time
        start_time = time.time()
        
        try:
            print(f"Loading LLAVA v1.6 Mistral 7B model...")
            
            # Load model with 4-bit quantization - following working script pattern
            print(f"Loading LLAVA v1.6 Mistral 7B with 4-bit quantization...")
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
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
            measure_memory_usage()
            print(f"Model loaded in {duration:.2f} seconds")
            
        except Exception as e:
            print(f"Error loading LLAVA v1.6 Mistral 7B model: {e}")
            import traceback
            traceback.print_exc()
            self.model_loaded = False
    
    @time_inference
    def predict(self, image_path, question):
        """Process an image and question to generate an answer"""
        if not hasattr(self, 'model_loaded') or not self.model_loaded:
            return "Error: Model failed to load. Cannot perform prediction."
            
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Prepare conversation format for LLAVA v1.6 - following working script pattern
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image"},
                    ],
                }
            ]
            
            # Process inputs with memory-efficient settings
            with torch.inference_mode():  # Use inference_mode to save memory
                # Clear cache before heavy operations
                cleanup_memory()
                
                # Prepare prompt using LLAVA Next's chat template - following working script
                prompt = self.processor.apply_chat_template(
                    conversation, 
                    add_generation_prompt=True
                )
                
                # Process inputs - following working script pattern
                inputs = self.processor(
                    images=image, 
                    text=prompt, 
                    return_tensors="pt"
                ).to(self.device)
                
                # Clear cache again before generation
                cleanup_memory()
                
                # Generate response with memory-efficient settings
                print(f"Generating response with LLAVA v1.6 Mistral 7B...")
                
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,  # Deterministic to save memory
                    num_beams=1,  # No beam search to save memory
                )
            
            # Process output - use proper token handling for LLAVA v1.6
            output_text = self.processor.decode(
                generated_ids[0][inputs['input_ids'].shape[1]:],  # Skip input tokens properly
                skip_special_tokens=True
            )
            
            return output_text
            
        except Exception as e:
            print(f"Error in LLAVA v1.6 Mistral 7B prediction: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    def cleanup(self):
        """Clean up GPU resources"""
        print(f"Cleaning up LLAVA v1.6 Mistral 7B resources...")
        
        if hasattr(self, 'model') and self.model is not None:
            # Use robust model cleanup utility
            model_cleanup(self.model)
            self.model = None
            
        print(f"{self.model_name} resources cleaned up")