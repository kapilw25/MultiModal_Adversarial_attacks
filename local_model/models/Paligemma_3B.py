import torch
from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
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

class PaliGemmaModelWrapper(BaseVLModel):
    """Wrapper class for the PaliGemma 3B model with 4-bit quantization"""
    
    @memory_efficient
    def __init__(self, model_name="PaliGemma-3B-mix-224_4bit"):
        super().__init__(model_name)
        self.device = get_device()
        
        # Initial memory cleanup
        cleanup_memory()
        
        # Model path
        model_path = "google/paligemma-3b-mix-224"
        
        # Load processor first
        print(f"Loading processor from {model_path}...")
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        # Configure 4-bit quantization
        print(f"Setting up 4-bit quantization for {model_name}...")
        self.quantization_config = get_quantization_config(
            load_in_4bit=True,
            compute_dtype=torch.float16,  # Changed to float16 for better compatibility
            use_double_quant=True,
            quant_type="nf4"
        )
        
        # Set environment variables for memory optimization
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.9)  # Use only 90% of available GPU memory
        
        # Load model directly without using load_model_with_timing to have more control
        print(f"Loading model from {model_path}...")
        try:
            # First try loading with device_map="auto" for automatic placement
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_path,
                quantization_config=self.quantization_config,
                torch_dtype=torch.float16,  # Changed to float16 for better compatibility
                low_cpu_mem_usage=True,
                device_map="auto"  # Let the model decide the best device placement
            )
            
            # Explicitly tie weights to resolve the warning
            if hasattr(self.model, 'tie_weights'):
                self.model.tie_weights()
                
        except Exception as e:
            print(f"Error with auto device mapping: {e}")
            print("Trying alternative loading method...")
            
            # Alternative loading method
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_path,
                quantization_config=self.quantization_config,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            
            # Explicitly move model to device after loading
            self.model = self.model.to(self.device)
            
            # Explicitly tie weights
            if hasattr(self.model, 'tie_weights'):
                self.model.tie_weights()
    
    @time_inference
    def predict(self, image_path, question):
        """Process an image and question to generate an answer"""
        try:
            from PIL import Image
            
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Prepare prompt
            # PaliGemma uses task prefixes like "vqa" for visual question answering
            prompt = f"vqa {question} Answer format (do not generate any other content): The answer is <answer>."
            
            # Process inputs
            model_inputs = self.processor(
                text=prompt, 
                images=image, 
                return_tensors="pt"
            )
            
            # Explicitly move inputs to the same device as the model
            for key in model_inputs:
                if isinstance(model_inputs[key], torch.Tensor):
                    model_inputs[key] = model_inputs[key].to(self.device)
            
            input_len = model_inputs["input_ids"].shape[-1]
            
            # Generate response
            print(f"Generating response with {self.model_name}...")
            
            # Clean up memory before generation
            cleanup_memory()
            
            with torch.inference_mode():
                generation = self.model.generate(
                    **model_inputs, 
                    max_new_tokens=100, 
                    do_sample=False
                )
                generation = generation[0][input_len:]
            
            # Process output
            print("Processing output...")
            decoded = self.processor.decode(generation, skip_special_tokens=True)
            
            return decoded
            
        except Exception as e:
            print(f"Error in PaliGemma prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    @memory_efficient
    def cleanup(self):
        """Clean up GPU resources"""
        model_cleanup(self.model)
        print(f"{self.model_name} resources cleaned up")
