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

class DeepSeekVL1pt3BModelWrapper(BaseVLModel):
    """Wrapper class for the DeepSeek-VL-1.3B model with 4-bit quantization"""
    
    def __init__(self, model_name="DeepSeek-VL-1.3B-chat_4bit"):
        super().__init__(model_name)
        self.device = get_device()
        
        # Initial memory cleanup
        cleanup_memory()
        
        # Configure 4-bit quantization with more aggressive memory optimization
        print(f"Setting up optimized 4-bit quantization for {model_name}...")
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,  # Use bfloat16 to match the model's expected type
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,  # Enable double quantization for further memory savings
            llm_int8_enable_fp32_cpu_offload=False  # Disable CPU offloading
        )
        
        # Model path
        model_path = "deepseek-ai/deepseek-vl-1.3b-chat"
        
        # Load processor and tokenizer
        print(f"Loading DeepSeek-VL processor from {model_path}...")
        self.processor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer
        
        # Load model with optimized settings
        print(f"Loading DeepSeek-VL model from {model_path} with optimized settings...")
        
        # Measure memory before loading
        print("Memory before model loading:")
        self._print_memory_usage()
        
        # Record start time
        start_time = time.time()
        
        # Load model with memory optimization flags
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            quantization_config=self.quantization_config,
            torch_dtype=torch.bfloat16,  # Use bfloat16 to match the model's expected type
            low_cpu_mem_usage=True,     # Reduce CPU memory usage during loading
            device_map="auto",          # Let the library optimize device placement
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
            
            # Load images - don't resize to avoid potential issues
            pil_images = self.load_pil_images(conversation)
            
            # Process inputs
            prepare_inputs = self.processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True
            ).to(self.device)
            
            # Run image encoder to get the image embeddings
            with torch.inference_mode():  # Use inference_mode to save memory
                inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
                
                # Generate response with memory-efficient settings
                print(f"Generating response with {self.model_name}...")
                outputs = self.model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=self.tokenizer.eos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=64,  # Reduced from 128 to save memory
                    do_sample=False,
                    use_cache=True
                )
            
            # Decode output
            answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
            
            return answer
            
        except Exception as e:
            print(f"Error in DeepSeek-VL-1.3B prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    def cleanup(self):
        """Clean up GPU resources"""
        if hasattr(self, 'model'):
            # Explicitly move model to CPU before deletion
            self.model.to('cpu')
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Delete model
            del self.model
            self.model = None
        print(f"{self.model_name} resources cleaned up")
