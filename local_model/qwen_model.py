import torch
import gc
import time
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from .base_model import BaseVLModel

class QwenVLModelWrapper(BaseVLModel):
    """Wrapper class for the Qwen2.5-VL-3B-Instruct model with 4-bit quantization"""
    
    def __init__(self, model_name="Qwen2.5-VL-3B-Instruct_4bit"):
        super().__init__(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initial memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
        
        # Configure 4-bit quantization
        print(f"Setting up 4-bit quantization for {model_name}...")
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Load model with quantization
        print(f"Loading {model_name}...")
        start_time = time.time()
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            quantization_config=self.quantization_config,
            device_map="auto"
        )
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
        
        # Load processor with recommended pixel settings
        min_pixels = 256*28*28
        max_pixels = 1280*28*28
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct", 
            min_pixels=min_pixels, 
            max_pixels=max_pixels
        )
        print(f"Processor loaded successfully")
    
    def predict(self, image_path, question):
        """Process an image and question to generate an answer"""
        try:
            # Prepare messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {
                            "type": "text", 
                            "text": question + " Answer format (do not generate any other content): The answer is <answer>."
                        },
                    ],
                }
            ]
            
            # Process inputs
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
            
            # Generate response
            print(f"Generating response with {self.model_name}...")
            start_time = time.time()
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
            inference_time = time.time() - start_time
            print(f"Response generated in {inference_time:.2f} seconds")
            
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
            print(f"Error in Qwen prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    def cleanup(self):
        """Clean up GPU resources"""
        del self.model
        torch.cuda.empty_cache()
        gc.collect()
        print(f"{self.model_name} resources cleaned up")
