"""
Implementation of LLAVA 1.5 7B model with 4-bit quantization
for memory-efficient inference on consumer GPUs.
"""

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from local_model.base_model import BaseVLModel
from local_model.model_utils import (
    cleanup_memory,
    get_device,
    get_quantization_config,
    time_inference,
    model_cleanup,
    measure_memory_usage,
    robust_generate_with_cuda_handling,
    safe_move_inputs_to_device
)
import time
import gc
from PIL import Image

class LLAVA15ModelWrapper(BaseVLModel):
    """
    Wrapper class for LLAVA 1.5 7B model with optimized settings
    for memory-efficient inference on consumer GPUs.
    """
    
    def __init__(self, model_name):
        super().__init__(model_name)
        self.device = get_device()
        
        # Model configuration
        self.model_path = "llava-hf/llava-1.5-7b-hf"
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
        
        print(f"Loading LLAVA 1.5 7B processor...")
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_path)
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
            print(f"Loading LLAVA 1.5 7B model...")
            
            # Load model with 4-bit quantization - following working script pattern
            print(f"Loading LLAVA 1.5 7B with 4-bit quantization...")
            self.model = LlavaForConditionalGeneration.from_pretrained(
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
            print(f"Error loading LLAVA 1.5 7B model: {e}")
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
            
            # Prepare conversation format for LLAVA 1.5 - following working script pattern
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
                
                # Prepare prompt using LLAVA's chat template - following working script
                prompt = self.processor.apply_chat_template(
                    conversation, 
                    add_generation_prompt=True
                )
                
                # Process inputs - following working script pattern
                inputs = self.processor(
                    images=image,
                    text=prompt,
                    return_tensors='pt'
                )

                # Safely move inputs to device with centralized error handling
                inputs = safe_move_inputs_to_device(inputs, self.device, self.model_name)
                inputs = {k: v.to(self.dtype) if v.dtype == torch.float32 else v for k, v in inputs.items()}

                # Clear cache again before generation
                cleanup_memory()

                # Generate response with centralized CUDA error handling
                generated_ids = robust_generate_with_cuda_handling(
                    model=self.model,
                    inputs=inputs,
                    processor=self.processor,
                    model_name=self.model_name,
                    max_new_tokens=64,
                    do_sample=False,
                    num_beams=1
                )
            
            # Process output - use proper token handling for LLAVA 1.5
            output_text = self.processor.decode(
                generated_ids[0][inputs['input_ids'].shape[1]:],  # Skip input tokens instead
                skip_special_tokens=True
            )
            
            return output_text
            
        except Exception as e:
            print(f"Error in LLAVA 1.5 7B prediction: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    def cleanup(self):
        """Clean up GPU resources"""
        print(f"Cleaning up LLAVA 1.5 7B resources...")
        
        if hasattr(self, 'model') and self.model is not None:
            # Use robust model cleanup utility
            model_cleanup(self.model)
            self.model = None
            
        print(f"{self.model_name} resources cleaned up")