import torch
import os
import gc
import time
import psutil
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from local_model.base_model import BaseVLModel
from local_model.model_utils import (
    cleanup_memory, 
    get_device, 
    get_quantization_config,
    time_inference,
    model_cleanup,
    memory_efficient
)

class Phi35VisionModelWrapper(BaseVLModel):
    """Wrapper class for the Microsoft Phi-3.5-vision-instruct model with 4-bit quantization"""
    
    @memory_efficient
    def __init__(self, model_name="Phi-3.5-vision-instruct-4bit"):
        super().__init__(model_name)
        self.device = get_device()
        
        # Enforce CUDA requirement
        if not torch.cuda.is_available():
            raise RuntimeError("❌ CUDA is not available! Phi-3.5-vision requires GPU.")
        
        # Set environment variable to avoid memory fragmentation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # Initial aggressive memory cleanup
        cleanup_memory()
        
        # Configure 4-bit quantization for Microsoft models (use bfloat16 like Google models)
        print(f"Setting up 4-bit quantization for {model_name}...")
        self.quantization_config = get_quantization_config(
            load_in_4bit=True,
            compute_dtype=torch.bfloat16,  # Better for Microsoft models
            use_double_quant=True,
            quant_type="nf4"
        )
        
        print("   Using 4-bit quantization config:")
        print(f"   - Compute dtype: {self.quantization_config.bnb_4bit_compute_dtype}")
        print(f"   - Quantization type: {self.quantization_config.bnb_4bit_quant_type}")
        print(f"   - Double quantization: {self.quantization_config.bnb_4bit_use_double_quant}")
        
        # Measure memory before loading
        print("Memory before model loading:")
        self._print_memory_usage()
        
        # Record start time
        start_time = time.time()
        
        try:
            # Load model with optimizations
            model_path = "microsoft/Phi-3.5-vision-instruct"
            print(f"Loading model from {model_path}...")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=self.quantization_config,
                device_map="auto",  # Let it decide optimal placement
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,  # Consistent with quantization
                _attn_implementation='eager',  # Explicitly disable flash attention
                low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
                use_cache=False  # Disable cache to avoid DynamicCache issues
            )
            
            print("   ✅ Successfully loaded with 4-bit quantization")
            
            # Load processor with memory optimization
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
                num_crops=4  # Reduced from 16 for memory efficiency
            )
            
            self.model_loaded = True
            
        except Exception as e:
            print(f"Error loading Phi-3.5-vision model: {str(e)}")
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
            
            # Print model statistics
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            print(f"Model Statistics:")
            print(f"   Total parameters: {total_params:,}")
            print(f"   Trainable parameters: {trainable_params:,}")
            print(f"   Model size: ~{total_params * 4 / 1024**3:.2f} GB (FP32 equivalent)")
            print(f"   Actual GPU usage: ~{total_params / 1024**3:.2f} GB (4-bit quantized)")
    
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
            # Load and process the image
            try:
                if os.path.exists(image_path):
                    image = Image.open(image_path)
                    # Resize image to reduce memory usage if too large
                    if image.size[0] > 1024 or image.size[1] > 1024:
                        image = image.resize((1024, 1024), Image.Resampling.LANCZOS)
                    print(f"   ✅ Successfully loaded image: {image.size}")
                else:
                    print(f"   ❌ Image not found at: {image_path}")
                    return f"Error: Image not found at {image_path}"
            except Exception as e:
                print(f"   ⚠️  Failed to load image: {e}")
                return f"Error loading image: {str(e)}"
            
            # Prepare the prompt using Phi-3.5-vision chat format
            messages = [
                {"role": "user", "content": f"<|image_1|>\n{question}"}
            ]
            
            prompt = self.processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            print(f"   Question: {question}")
            print(f"   Complete Prompt:\n{prompt}")
            
            # Measure memory before inference
            print("Memory before inference:")
            self._print_memory_usage()
            
            # Process inputs with reduced memory usage
            inputs = self.processor(prompt, [image], return_tensors="pt")
            
            # Move to GPU efficiently
            inputs = {k: v.to("cuda", non_blocking=True) if hasattr(v, 'to') else v 
                     for k, v in inputs.items()}
            
            generation_args = {
                "max_new_tokens": 500,  # Keep detailed responses
                "do_sample": False,
                "pad_token_id": self.processor.tokenizer.eos_token_id,
                "use_cache": False  # Disable cache to avoid DynamicCache issues
            }
            
            # Generate response with memory management
            print(f"Generating response with {self.model_name}...")
            with torch.no_grad():
                generate_ids = self.model.generate(
                    **inputs,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    **generation_args
                )
            
            # Decode response
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            response = self.processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            # Measure memory after inference
            print("Memory after inference:")
            self._print_memory_usage()
            
            # Cleanup intermediate tensors
            del inputs, generate_ids
            cleanup_memory()
            
            return response
            
        except Exception as e:
            print(f"Error in Phi-3.5-vision prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            cleanup_memory()
            return f"Error: {str(e)}"
    
    @memory_efficient
    def cleanup(self):
        """Clean up GPU resources"""
        if hasattr(self, 'model') and self.model is not None:
            # Explicitly move model to CPU before deletion
            try:
                self.model.to('cpu')
            except:
                pass
                
            # Delete model and processor
            del self.model
            if hasattr(self, 'processor'):
                del self.processor
            
            self.model = None
            self.processor = None
            
        # Force cleanup
        cleanup_memory()
        print(f"{self.model_name} resources cleaned up")
