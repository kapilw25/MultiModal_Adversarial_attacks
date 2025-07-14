"""
Implementation of InternVL3 models (1B, 2B) with 4-bit quantization
for memory-efficient inference on consumer GPUs.
"""

import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig, AutoConfig
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

# Constants for image preprocessing
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class InternVL3ModelWrapper(BaseVLModel):
    """
    Wrapper class for InternVL3 models (1B, 2B) with optimized settings
    for memory-efficient inference on consumer GPUs.
    """
    
    def __init__(self, model_name):
        super().__init__(model_name)
        self.device = get_device()
        
        # Parse model size from name and set appropriate paths and configurations
        if "1B" in model_name:
            self.model_path = "OpenGVLab/InternVL3-1B"
            self.model_size = "1B"
            self.max_gpu_memory = "3GiB"
            self.input_size = 448
            self.max_num = 6  # Reduced from 12 for memory efficiency
        elif "2B" in model_name:
            self.model_path = "OpenGVLab/InternVL3-2B"
            self.model_size = "2B"
            self.max_gpu_memory = "5GiB"
            self.input_size = 448
            self.max_num = 4  # Further reduced for larger model
        else:
            raise ValueError(f"Unknown InternVL3 model size in name: {model_name}")
        
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
            print(f"Loading InternVL3-{self.model_size} model and tokenizer...")
            
            # Configure 4-bit quantization for memory efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,  # InternVL uses bfloat16
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            
            # Load tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True, 
                use_fast=False
            )
            
            # Load model with 4-bit quantization
            self.model = AutoModel.from_pretrained(
                self.model_path,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,  # InternVL uses bfloat16
                low_cpu_mem_usage=True,
                device_map="auto",
                max_memory={0: self.max_gpu_memory, "cpu": "16GiB"},
                trust_remote_code=True,
                use_flash_attn=False  # Disable flash attention for compatibility
            ).eval()  # Explicitly set to evaluation mode
            
            self.model_loaded = True
            self.history = None  # Initialize conversation history
            
            # Record end time and calculate duration
            end_time = time.time()
            duration = end_time - start_time
            
            # Measure memory after loading
            print("Memory after model loading:")
            self._print_memory_usage()
            print(f"Model loaded in {duration:.2f} seconds")
            
        except Exception as e:
            print(f"Error loading InternVL3-{self.model_size} model: {e}")
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
    
    def build_transform(self, input_size):
        """Build image transformation pipeline"""
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        return transform
    
    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        """Find the closest aspect ratio for dynamic preprocessing"""
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio
    
    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        """Dynamically preprocess image based on aspect ratio"""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # Calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # Find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # Calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # Resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # Split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images
    
    def load_image(self, image_file):
        """Load and preprocess image for InternVL3 model"""
        image = Image.open(image_file).convert('RGB')
        transform = self.build_transform(input_size=self.input_size)
        images = self.dynamic_preprocess(
            image, 
            image_size=self.input_size, 
            use_thumbnail=True, 
            max_num=self.max_num
        )
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    
    @time_inference
    def predict(self, image_path, question):
        """Process an image and question to generate an answer"""
        if not hasattr(self, 'model_loaded') or not self.model_loaded:
            return "Error: Model failed to load. Cannot perform prediction."
            
        try:
            # Measure memory before inference
            print("Memory before inference:")
            self._print_memory_usage()
            
            # Load and preprocess image
            pixel_values = self.load_image(image_path).to(torch.bfloat16).to(self.device)
            
            # Format question with image placeholder
            formatted_question = "<image>\n" + question
            
            # Process inputs with memory-efficient settings
            with torch.inference_mode():  # Use inference_mode to save memory
                # Clear cache before heavy operations
                torch.cuda.empty_cache()
                
                # Generate response
                print(f"Generating response with InternVL3-{self.model_size}...")
                
                # Configure generation parameters
                generation_config = {
                    "max_new_tokens": 128,
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
                
                # Generate response using the model's chat method
                # Handle conversation history if available
                if hasattr(self, 'history') and self.history:
                    response, new_history = self.model.chat(
                        self.tokenizer,
                        pixel_values,
                        formatted_question,
                        generation_config,
                        history=self.history,
                        return_history=True
                    )
                    self.history = new_history  # Update history for potential future use
                else:
                    # First interaction or no history tracking
                    response, new_history = self.model.chat(
                        self.tokenizer,
                        pixel_values,
                        formatted_question,
                        generation_config,
                        history=None,
                        return_history=True
                    )
                    self.history = new_history  # Store for potential future use
            
            # Measure memory after inference
            print("Memory after inference:")
            self._print_memory_usage()
            
            return response
            
        except Exception as e:
            print(f"Error in InternVL3-{self.model_size} prediction: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    def cleanup(self):
        """Clean up GPU resources"""
        print(f"Cleaning up InternVL3-{self.model_size} resources...")
        
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
            
            # Clear conversation history
            if hasattr(self, 'history'):
                del self.history
                self.history = None
                
            # Force garbage collection
            gc.collect()
            
        print(f"{self.model_name} resources cleaned up")
