"""
Implementation of GLM Edge 2B model with bfloat16 precision
for memory-efficient inference on consumer GPUs.
"""

import torch
from transformers import AutoTokenizer, AutoImageProcessor, AutoModelForCausalLM
from local_model.base_model import BaseVLModel
from local_model.model_utils import (
    cleanup_memory, 
    get_device, 
    time_inference
)
import time
import psutil
import os
import gc
from PIL import Image

class GLMEdgeModelWrapper(BaseVLModel):
    """
    Wrapper class for GLM Edge 2B model with optimized settings
    for memory-efficient inference on consumer GPUs.
    """
    
    def __init__(self, model_name):
        super().__init__(model_name)
        self.device = get_device()
        
        # Set model path and configurations
        self.model_path = "THUDM/glm-edge-v-2b"
        self.model_size = "2B"
        self.max_gpu_memory = "4GiB"
        self.dtype = torch.bfloat16
        
        # Aggressive memory cleanup before loading
        cleanup_memory()
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"Memory before __init__:")
        self._print_memory_usage()
        
        print(f"Loading GLM Edge 2B processor...")
        try:
            # Set device_map to avoid device mismatch issues
            self.processor = AutoImageProcessor.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            print(f"Processor and tokenizer loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"Error loading processor or tokenizer: {e}")
            raise
        
        # Measure memory before loading
        print("Memory before model loading:")
        self._print_memory_usage()
        
        # Record start time
        start_time = time.time()
        
        try:
            print(f"Loading GLM Edge 2B model...")
            
            # Use bfloat16 without 4-bit quantization
            print(f"Loading GLM Edge 2B with {self.dtype} dtype...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            print(f"Successfully loaded model with {self.dtype} dtype")
            
            self.model_loaded = True
            
            # Record end time and calculate duration
            end_time = time.time()
            duration = end_time - start_time
            
            # Measure memory after loading
            print("Memory after model loading:")
            self._print_memory_usage()
            print(f"Model loaded in {duration:.2f} seconds")
            
        except Exception as e:
            print(f"Error loading GLM Edge 2B model: {e}")
            import traceback
            traceback.print_exc()
            self.model_loaded = False
        
        print(f"Memory after __init__:")
        self._print_memory_usage()
    
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
            
            # Prepare messages format according to GLM Edge's API
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "image"}, 
                        {"type": "text", "text": question}
                    ]
                }
            ]
            
            # Measure memory before inference
            print("Memory before inference:")
            self._print_memory_usage()
            
            # Process inputs with memory-efficient settings
            with torch.inference_mode():  # Use inference_mode to save memory
                # Clear cache before heavy operations
                torch.cuda.empty_cache()
                
                # Prepare inputs using GLM Edge's chat template
                inputs = self.tokenizer.apply_chat_template(
                    messages, 
                    add_generation_prompt=True, 
                    return_dict=True, 
                    tokenize=True, 
                    return_tensors="pt"
                ).to(next(self.model.parameters()).device)
                
                # Process image with explicit dtype conversion
                processed_image = self.processor(image)
                pixel_values = torch.tensor(
                    processed_image.pixel_values,
                    dtype=self.dtype  # Explicitly convert to bfloat16
                ).to(next(self.model.parameters()).device)
                
                # Extract input_ids and attention_mask
                input_ids = inputs["input_ids"]
                attention_mask = inputs.get("attention_mask", None)
                
                # Clear cache again before generation
                torch.cuda.empty_cache()
                
                # Generate response with memory-efficient settings
                print(f"Generating response with GLM Edge 2B...")
                
                # Set max tokens for generation
                max_tokens = 64
                
                # Use custom token-by-token generation since the standard generate method doesn't work
                
                # Initial forward pass with the image
                outputs = self.model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                
                # Get the logits from the output
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                
                # Initialize the output sequence with the input followed by the first predicted token
                output_sequence = torch.cat([input_ids, next_token], dim=-1)
                
                # Generate remaining tokens
                generated_text = ""
                for _ in range(max_tokens - 1):
                    try:
                        # Forward pass with the current sequence
                        outputs = self.model(
                            input_ids=output_sequence,
                            pixel_values=pixel_values,
                            attention_mask=None,  # Let the model handle attention for the growing sequence
                            return_dict=True,
                        )
                        
                        # Get the next token prediction
                        next_token_logits = outputs.logits[:, -1, :]
                        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                        
                        # Decode the token
                        token_text = self.tokenizer.decode([next_token.item()], skip_special_tokens=True)
                        generated_text += token_text
                        
                        # Append the new token to the sequence
                        output_sequence = torch.cat([output_sequence, next_token], dim=-1)
                        
                        # Check for EOS token
                        if next_token.item() == self.tokenizer.eos_token_id:
                            break
                            
                        # Check if we have a reasonable amount of text
                        if len(generated_text) > 100:
                            break
                            
                    except Exception as e:
                        print(f"Error during token generation: {e}")
                        # If we encounter an error, stop generation and use what we have
                        break
            
            # Process output
            print("Processing output...")
            
            # Measure memory after inference
            print("Memory after inference:")
            self._print_memory_usage()
            
            return generated_text if generated_text else "Could not generate a response for this image."
            
        except Exception as e:
            print(f"Error in GLM Edge 2B prediction: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    def cleanup(self):
        """Clean up GPU resources"""
        print(f"Memory before cleanup:")
        self._print_memory_usage()
        
        print(f"Cleaning up GLM Edge 2B resources...")
        
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
        
        # Also clean up processor and tokenizer
        if hasattr(self, 'processor'):
            del self.processor
        
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
            
        # Force garbage collection again
        gc.collect()
        
        print(f"{self.model_name} resources cleaned up")
        
        print(f"Memory after cleanup:")
        self._print_memory_usage()
