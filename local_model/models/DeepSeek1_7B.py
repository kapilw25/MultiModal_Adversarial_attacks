import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from deepseek_vl.models import VLChatProcessor
from deepseek_vl.utils.io import load_pil_images
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

class DeepSeekVL7BModelWrapper(BaseVLModel):
    """Wrapper class for the DeepSeek-VL-7B model with extreme 4-bit quantization for 8GB GPUs"""
    
    def __init__(self, model_name="DeepSeek-VL-7B-chat_4bit"):
        super().__init__(model_name)
        self.device = get_device()
        
        # Aggressive memory cleanup
        cleanup_memory()
        gc.collect()
        torch.cuda.empty_cache()
        
        # Configure 4-bit quantization with extreme memory optimization
        print(f"Setting up extreme 4-bit quantization for {model_name}...")
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,  # Use bfloat16 to match the model's expected type
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,  # Enable double quantization for further memory savings
            llm_int8_enable_fp32_cpu_offload=True,  # Enable CPU offloading for parts of the model
            llm_int8_threshold=6.0,  # Aggressive threshold for int8 quantization
            bnb_4bit_quant_storage=torch.uint8  # Use uint8 storage for 4-bit weights
        )
        
        # Model path
        model_path = "deepseek-ai/deepseek-vl-7b-chat"
        
        # Load processor and tokenizer first
        print(f"Loading DeepSeek-VL processor from {model_path}...")
        self.processor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer
        
        # Load model with extreme memory optimization settings
        print(f"Loading DeepSeek-VL-7B model from {model_path} with extreme memory optimization...")
        
        # Measure memory before loading
        print("Memory before model loading:")
        self._print_memory_usage()
        
        # Record start time
        start_time = time.time()
        
        try:
            # Load model with extreme memory optimization flags
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                quantization_config=self.quantization_config,
                torch_dtype=torch.bfloat16,  # Use bfloat16 to match the model's expected type
                low_cpu_mem_usage=True,     # Reduce CPU memory usage during loading
                device_map="auto",          # Let the library optimize device placement
                max_memory={0: "6GiB", "cpu": "16GiB"},  # Limit GPU memory usage to 6GB, offload the rest to CPU
                offload_folder="offload_folder",  # Folder for offloading weights
                offload_state_dict=True,    # Offload state dict to CPU
                trust_remote_code=True
            )
            
            # Record end time and calculate duration
            end_time = time.time()
            duration = end_time - start_time
            
            # Measure memory after loading
            print("Memory after model loading:")
            self._print_memory_usage()
            print(f"Model loaded in {duration:.2f} seconds")
            
            # Store the load_pil_images function for later use
            self.load_pil_images = load_pil_images
            
            self.model_loaded = True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
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
            # Prepare conversation format
            conversation = [
                {
                    "role": "User",
                    "content": f"<image_placeholder>{question} Answer format (do not generate any other content): The answer is <answer>.",
                    "images": [image_path]
                },
                {
                    "role": "Assistant",
                    "content": ""
                }
            ]
            
            # Load and resize image to smaller dimensions to save memory
            image = Image.open(image_path).convert('RGB')
            # Resize to smaller dimensions (224x224 instead of 1024x1024)
            image = image.resize((224, 224))
            pil_images = [image]
            
            # Process inputs with memory-efficient settings
            prepare_inputs = self.processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True
            ).to(self.device)
            
            # Run image encoder to get the image embeddings with memory optimization
            with torch.inference_mode():  # Use inference_mode to save memory
                # Clear cache before heavy operations
                torch.cuda.empty_cache()
                
                # Process in chunks to save memory
                inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
                
                # Clear cache again before generation
                torch.cuda.empty_cache()
                
                # Generate response with extreme memory-efficient settings
                print(f"Generating response with {self.model_name}...")
                outputs = self.model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=self.tokenizer.eos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=32,  # Extremely reduced from 512 to save memory
                    do_sample=False,
                    use_cache=True,
                    num_beams=1,  # Disable beam search to save memory
                )
            
            # Decode output
            answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
            
            return answer
            
        except Exception as e:
            print(f"Error in DeepSeek-VL-7B prediction: {str(e)}")
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
