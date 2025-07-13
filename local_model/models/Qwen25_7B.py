import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
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

class Qwen25VL7BModelWrapper(BaseVLModel):
    """Wrapper class for the Qwen2.5-VL-7B-Instruct model with 4-bit quantization for 8GB GPUs"""
    
    def __init__(self, model_name="Qwen2.5-VL-7B-Instruct-4bit"):
        super().__init__(model_name)
        self.device = get_device()
        
        # Aggressive memory cleanup
        cleanup_memory()
        gc.collect()
        torch.cuda.empty_cache()
        
        # Use standard model path directly
        standard_model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
        
        # Load processor with optimized settings for smaller images
        print(f"Loading Qwen2.5-VL processor...")
        # Set min_pixels and max_pixels for smaller image processing
        min_pixels = 256 * 28 * 28  # Minimum pixel count
        max_pixels = 512 * 28 * 28  # Reduced from 1280*28*28 to save memory
        
        # Load processor directly from standard model
        self.processor = AutoProcessor.from_pretrained(
            standard_model_path, 
            min_pixels=min_pixels, 
            max_pixels=max_pixels
        )
        
        # Measure memory before loading
        print("Memory before model loading:")
        self._print_memory_usage()
        
        # Record start time
        start_time = time.time()
        
        try:
            # Configure 4-bit quantization using bitsandbytes
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            
            # Load the standard model with 4-bit quantization
            print(f"Loading standard model with 4-bit quantization from {standard_model_path}...")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                standard_model_path,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
                max_memory={0: "6GiB", "cpu": "16GiB"},
            )
            print("Successfully loaded standard model with 4-bit quantization")
            self.model_loaded = True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            self.model_loaded = False
        
        # Record end time and calculate duration if model loaded successfully
        if self.model_loaded:
            end_time = time.time()
            duration = end_time - start_time
            
            # Measure memory after loading
            print("Memory after model loading:")
            self._print_memory_usage()
            print(f"Model loaded in {duration:.2f} seconds")
    
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
            # Prepare messages format according to Qwen's API
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                            "resized_height": 224,  # Resize to smaller dimensions to save memory
                            "resized_width": 224,
                        },
                        {
                            "type": "text", 
                            "text": question + " Answer format (do not generate any other content): The answer is <answer>."
                        },
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
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(self.device)
                
                # Clear cache again before generation
                torch.cuda.empty_cache()
                
                # Generate response with extreme memory-efficient settings
                print(f"Generating response with {self.model_name}...")
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=64,  # Reduced from 128 to save memory
                    do_sample=False,    # Disable sampling to save memory
                    num_beams=1,        # Disable beam search to save memory
                    use_cache=True,
                )
            
            # Process output
            print("Processing output...")
            print(f"Input IDs shape: {inputs.input_ids.shape}")
            print(f"Generated IDs shape: {generated_ids.shape}")
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            return output_text[0]
            
        except Exception as e:
            print(f"Error in Qwen2.5-VL-7B prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    def cleanup(self):
        """Clean up GPU resources"""
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
