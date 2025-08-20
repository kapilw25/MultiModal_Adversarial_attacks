import torch
import time
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from local_model.base_model import BaseVLModel
from local_model.model_utils import (
    cleanup_memory, 
    get_device, 
    get_quantization_config, 
    load_model_with_timing,
    time_inference,
    get_processor_with_pixel_settings,
    model_cleanup
)

class QwenVLModelWrapper(BaseVLModel):
    """Wrapper class for the Qwen2.5-VL-3B-Instruct model with 4-bit quantization"""
    
    def __init__(self, model_name="Qwen2.5-VL-3B-Instruct_4bit"):
        super().__init__(model_name)
        self.device = get_device()
        
        # Initial memory cleanup
        cleanup_memory()
        
        # Configure 4-bit quantization
        print(f"Setting up 4-bit quantization for {model_name}...")
        self.quantization_config = get_quantization_config()
        
        # Load model with quantization
        model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
        self.model = load_model_with_timing(
            Qwen2_5_VLForConditionalGeneration,
            model_path,
            self.quantization_config,
            max_memory={0: "7GiB", "cpu": "16GiB"}  # Upgraded to 7GiB for better performance
        )
        
        # Load processor with recommended pixel settings
        self.processor = get_processor_with_pixel_settings(model_path)
    
    @time_inference
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
            
            # Safely move inputs to device with error handling
            try:
                inputs = inputs.to(self.device)
            except RuntimeError as e:
                if "CUDA error: device-side assert triggered" in str(e):
                    print(f"⚠️  CUDA device-side assert error detected. Skipping corrupted input.")
                    print(f"Error details: {e}")
                    return "ERROR: CUDA device-side assert - corrupted input data"
                else:
                    raise e
            
            # Generate response with CUDA error handling
            print(f"Generating response with {self.model_name}...")
            try:
                with torch.inference_mode():
                    generated_ids = self.model.generate(
                        **inputs, 
                        max_new_tokens=128,
                        do_sample=True,
                        temperature=0.3,  # Lower temperature to prevent invalid probabilities
                        top_p=0.95,       # Higher top_p for more stable sampling
                        pad_token_id=self.processor.tokenizer.eos_token_id,  # Explicit pad token
                        eos_token_id=self.processor.tokenizer.eos_token_id,  # Explicit eos token
                    )
            except RuntimeError as e:
                if "CUDA error: device-side assert triggered" in str(e):
                    print(f"⚠️  CUDA device-side assert during generation. Input may be corrupted.")
                    print(f"Error details: {e}")
                    return "ERROR: CUDA device-side assert during generation - corrupted input"
                else:
                    raise e
            
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
        """Clean up GPU resources with enhanced error handling"""
        try:
            print(f"Starting cleanup for {self.model_name}...")
            model_cleanup(self.model)
            
            # Clean up processor if it exists
            if hasattr(self, 'processor'):
                del self.processor
                print("Processor cleaned up")
            
            print(f"{self.model_name} resources cleaned up successfully")
            
        except Exception as e:
            print(f"⚠️  Error during {self.model_name} cleanup: {e}")
            print("Cleanup may be incomplete, but continuing...")
            
            # Force basic cleanup even if enhanced cleanup failed
            try:
                import gc
                gc.collect()
                if hasattr(torch, 'cuda') and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("Basic cleanup forced")
            except Exception as basic_error:
                print(f"❌ Even basic cleanup failed: {basic_error}")
                print("System may need restart to recover GPU resources")
