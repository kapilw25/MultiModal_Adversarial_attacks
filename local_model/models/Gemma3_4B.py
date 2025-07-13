import torch
from transformers import Gemma3ForConditionalGeneration
from local_model.base_model import BaseVLModel
from local_model.model_utils import (
    cleanup_memory, 
    get_device, 
    get_quantization_config,
    load_model_with_timing,
    time_inference,
    get_processor_with_pixel_settings,
    model_cleanup,
    memory_efficient
)

class GemmaVLModelWrapper(BaseVLModel):
    """Wrapper class for the Gemma-3-4b-it model with 4-bit quantization"""
    
    @memory_efficient
    def __init__(self, model_name="Gemma-3-4b-it_4bit"):
        super().__init__(model_name)
        self.device = get_device()
        
        # Initial memory cleanup
        cleanup_memory()
        
        # Configure 4-bit quantization
        print(f"Setting up 4-bit quantization for {model_name}...")
        self.quantization_config = get_quantization_config(
            load_in_4bit=True,
            compute_dtype=torch.bfloat16,  # Use bfloat16 for better compatibility with Gemma
            use_double_quant=True,
            quant_type="nf4"
        )
        
        # Set environment variables for memory optimization
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.9)  # Use only 90% of available GPU memory
        
        # Load model with quantization
        model_path = "google/gemma-3-4b-it"
        self.model = load_model_with_timing(
            Gemma3ForConditionalGeneration,
            model_path,
            self.quantization_config,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for Gemma
            low_cpu_mem_usage=True       # Additional memory optimization
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
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": question + " Answer format (do not generate any other content): The answer is <answer>."}
                    ]
                }
            ]
            
            # Process inputs - use more memory efficient settings
            inputs = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=True,
                return_dict=True, 
                return_tensors="pt"
            ).to(self.device, dtype=torch.bfloat16)  # Use bfloat16 for Gemma
            
            # Generate response
            print(f"Generating response with {self.model_name}...")
            input_len = inputs["input_ids"].shape[-1]
            
            # Clean up memory before generation
            cleanup_memory()
            
            with torch.inference_mode():
                # Use more memory-efficient generation settings
                generation = self.model.generate(
                    **inputs, 
                    max_new_tokens=128,
                    do_sample=False,  # Deterministic generation uses less memory
                    num_beams=1,      # Beam search with 1 beam = greedy search (less memory)
                )
                generation = generation[0][input_len:]
            
            # Process output
            print("Processing output...")
            output_text = self.processor.decode(generation, skip_special_tokens=True)
            
            return output_text
            
        except Exception as e:
            print(f"Error in Gemma prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    @memory_efficient
    def cleanup(self):
        """Clean up GPU resources"""
        model_cleanup(self.model)
        print(f"{self.model_name} resources cleaned up")
