"""
Implementation of TinyLLaVA models with SigLIP vision encoders (0.5B, 2.4B, 3B, 3.1B) with 4-bit quantization
for memory-efficient inference on consumer GPUs.
"""

import torch
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModel, AutoProcessor
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
import requests
from io import BytesIO
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

class TinyLLaVASigLIPModelWrapper(BaseVLModel):
    """
    Wrapper class for TinyLLaVA models with SigLIP vision encoders (0.5B, 2.4B, 3B, 3.1B) with optimized settings
    for memory-efficient inference on consumer GPUs.
    """
    
    def __init__(self, model_name):
        super().__init__(model_name)
        self.device = get_device()
        
        # Parse model size from name and set appropriate paths and configurations
        if "0.5B" in model_name or "0.89B" in model_name:
            if "Qwen2" in model_name:
                self.model_path = "Zhang199/TinyLLaVA-Qwen2-0.5B-SigLIP"
                self.model_size = "0.5B"
                self.conv_mode = "phi"  # Using phi mode for Qwen2 models
            else:
                self.model_path = "jiajunlong/TinyLLaVA-0.89B"
                self.model_size = "0.89B"
                self.conv_mode = "llama"
            self.max_gpu_memory = "2GiB"  # Smaller model needs less memory
            self.use_4bit = False  # Small enough to run in float16
            self.dtype = torch.float16
        elif "2.4B" in model_name:
            self.model_path = "tinyllava/TinyLLaVA-Gemma-SigLIP-2.4B"
            self.model_size = "2.4B"
            self.max_gpu_memory = "4GiB"
            self.use_4bit = True  # Use 4-bit quantization for larger models
            self.dtype = torch.bfloat16  # Use bfloat16 for Gemma models
            self.conv_mode = "gemma"
        elif "Qwen2.5-3B" in model_name:
            self.model_path = "Zhang199/TinyLLaVA-Qwen2.5-3B-SigLIP"
            self.model_size = "3B"
            self.max_gpu_memory = "5GiB"
            self.use_4bit = True  # Use 4-bit quantization for larger models
            self.dtype = torch.float16  # Use float16 for Qwen models
            self.conv_mode = "phi"  # Using phi mode for Qwen models
        elif "3.1B" in model_name:
            self.model_path = "tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B"
            self.model_size = "3.1B"
            self.max_gpu_memory = "5GiB"  # Larger model needs more memory
            self.use_4bit = True  # Use 4-bit quantization for larger models
            self.dtype = torch.float16  # Use float16 for Phi models
            self.conv_mode = "phi"
        else:
            raise ValueError(f"Unknown TinyLLaVA model size in name: {model_name}")
        
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
            print(f"Loading TinyLLaVA-SigLIP-{self.model_size} model...")
            
            # Configure model loading based on size
            if self.use_4bit:
                # 4-bit quantization for larger models
                print(f"Using 4-bit quantization for TinyLLaVA-SigLIP-{self.model_size} with {self.dtype} compute dtype")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self.dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                
                try:
                    # First try loading with AutoModelForCausalLM
                    print(f"Attempting to load {self.model_path} with AutoModelForCausalLM...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        quantization_config=quantization_config,
                        torch_dtype=self.dtype,
                        low_cpu_mem_usage=True,
                        device_map="auto",
                        max_memory={0: self.max_gpu_memory, "cpu": "16GiB"},
                        trust_remote_code=True
                    )
                except ValueError as e:
                    if "does not recognize this architecture" in str(e):
                        # If architecture not recognized, try with AutoModel
                        print(f"Standard loading failed. Trying with AutoModel for {self.model_path}...")
                        self.model = AutoModel.from_pretrained(
                            self.model_path,
                            quantization_config=quantization_config,
                            torch_dtype=self.dtype,
                            low_cpu_mem_usage=True,
                            device_map="auto",
                            max_memory={0: self.max_gpu_memory, "cpu": "16GiB"},
                            trust_remote_code=True
                        )
                    else:
                        # If it's some other error, re-raise it
                        raise e
            else:
                # For smaller models, use float16/bfloat16 without 4-bit quantization
                try:
                    print(f"Loading TinyLLaVA-SigLIP-{self.model_size} with {self.dtype} dtype...")
                    try:
                        # First try loading with AutoModelForCausalLM
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_path,
                            torch_dtype=self.dtype,
                            low_cpu_mem_usage=True,
                            device_map="auto",
                            max_memory={0: self.max_gpu_memory, "cpu": "16GiB"},
                            trust_remote_code=True
                        )
                        print(f"Successfully loaded model with {self.dtype} dtype")
                    except ValueError as e:
                        if "does not recognize this architecture" in str(e):
                            # If architecture not recognized, try with AutoModel
                            print(f"Standard loading failed. Trying with AutoModel for {self.model_path}...")
                            self.model = AutoModel.from_pretrained(
                                self.model_path,
                                torch_dtype=self.dtype,
                                low_cpu_mem_usage=True,
                                device_map="auto",
                                max_memory={0: self.max_gpu_memory, "cpu": "16GiB"},
                                trust_remote_code=True
                            )
                        else:
                            # If it's some other error, re-raise it
                            raise e
                except Exception as e:
                    print(f"Error loading model with specified dtype: {e}")
                    print("Trying with default dtype...")
                    try:
                        # Try loading with default dtype
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_path,
                            low_cpu_mem_usage=True,
                            device_map="auto",
                            max_memory={0: self.max_gpu_memory, "cpu": "16GiB"},
                            trust_remote_code=True
                        )
                        print("Successfully loaded model with default dtype")
                    except ValueError as e:
                        if "does not recognize this architecture" in str(e):
                            # If architecture not recognized, try with AutoModel
                            print(f"Standard loading failed. Trying with AutoModel for {self.model_path}...")
                            self.model = AutoModel.from_pretrained(
                                self.model_path,
                                low_cpu_mem_usage=True,
                                device_map="auto",
                                max_memory={0: self.max_gpu_memory, "cpu": "16GiB"},
                                trust_remote_code=True
                            )
                        else:
                            # If it's some other error, re-raise it
                            raise e
            
            # Load tokenizer and processor
            print(f"Loading tokenizer and processor for {self.model_path}...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, 
                    use_fast=False,
                    trust_remote_code=True
                )
                
                # Try to load processor if available
                try:
                    self.processor = AutoProcessor.from_pretrained(
                        self.model_path,
                        trust_remote_code=True
                    )
                except:
                    print("No processor found, will use tokenizer only")
                    self.processor = None
                    
            except Exception as e:
                print(f"Error loading tokenizer directly: {e}")
                print("Using tokenizer from model if available...")
                if hasattr(self.model, 'tokenizer'):
                    self.tokenizer = self.model.tokenizer
                    print("Successfully obtained tokenizer from model")
                else:
                    print("No tokenizer available in model, using default tokenizer")
                    self.tokenizer = None
                self.processor = None
            
            self.model_loaded = True
            
            # Record end time and calculate duration
            end_time = time.time()
            duration = end_time - start_time
            
            # Measure memory after loading
            print("Memory after model loading:")
            self._print_memory_usage()
            print(f"Model loaded in {duration:.2f} seconds")
            
        except Exception as e:
            print(f"Error loading TinyLLaVA-SigLIP-{self.model_size} model: {e}")
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
    
    def _load_image(self, image_path):
        """Load image from path or URL"""
        if image_path.startswith(('http://', 'https://')):
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_path).convert('RGB')
        return image
    
    @time_inference
    def predict(self, image_path, question):
        """Process an image and question to generate an answer"""
        if not hasattr(self, 'model_loaded') or not self.model_loaded:
            return "Error: Model failed to load. Cannot perform prediction."
            
        try:
            # Measure memory before inference
            print("Memory before inference:")
            self._print_memory_usage()
            
            # Load image
            image = self._load_image(image_path)
            
            # Process inputs with memory-efficient settings
            with torch.inference_mode():  # Use inference_mode to save memory
                # Clear cache before heavy operations
                torch.cuda.empty_cache()
                
                # Generate response
                print(f"Generating response with TinyLLaVA-SigLIP-{self.model_size}...")
                
                # Try different methods to generate a response
                try:
                    # Method 1: Try using the model's chat method if available
                    if hasattr(self.model, 'chat'):
                        print("Using model's chat method...")
                        if self.processor:
                            # If processor is available, use it
                            output_text = self.model.chat(
                                prompt=question,
                                image=image,
                                tokenizer=self.tokenizer,
                                processor=self.processor
                            )
                        else:
                            # Otherwise just use tokenizer
                            output_text = self.model.chat(
                                prompt=question,
                                image=image,
                                tokenizer=self.tokenizer
                            )
                        
                        # Handle if output is a tuple (text, time)
                        if isinstance(output_text, tuple) and len(output_text) == 2:
                            output_text, generation_time = output_text
                            print(f"Response generated in {generation_time:.2f} seconds")
                        
                    # Method 2: Try using the processor and generate method
                    elif self.processor:
                        print("Using processor and generate method...")
                        inputs = self.processor(
                            text=question,
                            images=image,
                            return_tensors="pt"
                        ).to(self.device)
                        
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=512,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                        )
                        
                        output_text = self.tokenizer.batch_decode(
                            generated_ids, 
                            skip_special_tokens=True
                        )[0]
                    
                    # Method 3: Try a more generic approach
                    else:
                        print("Using generic approach...")
                        # Try to encode the image if the model has an image encoder
                        if hasattr(self.model, 'encode_image'):
                            # Convert PIL image to tensor
                            from torchvision import transforms
                            transform = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])
                            image_tensor = transform(image).unsqueeze(0).to(self.device)
                            
                            # Encode image
                            image_features = self.model.encode_image(image_tensor)
                            
                            # Tokenize text
                            text_inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
                            
                            # Generate response
                            outputs = self.model.generate(
                                input_ids=text_inputs.input_ids,
                                image_features=image_features,
                                max_new_tokens=512,
                                do_sample=True,
                                temperature=0.7,
                                top_p=0.9,
                            )
                            
                            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                        else:
                            # If all else fails, return an error message
                            output_text = "Error: This model doesn't support the required methods for image-text generation."
                
                except Exception as e:
                    print(f"Error during inference: {e}")
                    import traceback
                    traceback.print_exc()
                    output_text = f"Error during inference: {str(e)}"
            
            # Measure memory after inference
            print("Memory after inference:")
            self._print_memory_usage()
            
            return output_text
            
        except Exception as e:
            print(f"Error in TinyLLaVA-SigLIP-{self.model_size} prediction: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    def cleanup(self):
        """Clean up GPU resources"""
        print(f"Cleaning up TinyLLaVA-SigLIP-{self.model_size} resources...")
        
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
            
            # Delete tokenizer
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
                self.tokenizer = None
                
            # Delete processor
            if hasattr(self, 'processor'):
                del self.processor
                self.processor = None
                
            # Force garbage collection
            gc.collect()
            
        print(f"{self.model_name} resources cleaned up")
