from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationMixin
import torch
import inspect
import requests
from PIL import Image
from io import BytesIO

# Modified script to handle the missing generate method
hf_path = 'jiajunlong/TinyLLaVA-OpenELM-450M-SigLIP-0.89B'

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(hf_path, trust_remote_code=True)
model.cuda()
config = model.config
print("Model loaded successfully")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(hf_path, use_fast=False, 
                                         model_max_length=config.tokenizer_model_max_length, 
                                         padding_side=config.tokenizer_padding_side)
print("Tokenizer loaded successfully")

# Let's try a different approach - use our own implementation to process the image
prompt = "What are these?"
image_url = "http://images.cocodataset.org/test-stuff2017/000000000001.jpg"

# Download and process the image
print(f"Downloading image from {image_url}")
response = requests.get(image_url)
image = Image.open(BytesIO(response.content)).convert('RGB')

# Try to use the model's vision encoder directly
try:
    print("Processing image with model's vision encoder...")
    # Check if the model has a vision encoder
    if hasattr(model, 'vision_tower'):
        print("Using vision_tower for encoding...")
        # Process the image with the vision encoder
        vision_tower = model.vision_tower
        
        # Resize image to expected size
        from torchvision import transforms
        if hasattr(vision_tower, 'image_processor'):
            processor = vision_tower.image_processor
            print(f"Using vision tower's image processor")
            image_tensor = processor(image, return_tensors="pt").pixel_values.to(model.device)
        else:
            # Use a standard transform
            print("Using standard transforms")
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image_tensor = transform(image).unsqueeze(0).to(model.device)
        
        # Get image features
        with torch.no_grad():
            image_features = vision_tower(image_tensor)
            print(f"Image features shape: {image_features.shape}")
        
        # Try to use the model's chat method with processed image features
        print("Attempting to use chat method with processed image features...")
        try:
            # Check if the model has a custom chat method that can handle image features
            if hasattr(model, 'chat') and 'image_features' in inspect.signature(model.chat).parameters:
                output_text, generation_time = model.chat(
                    prompt=prompt, 
                    image_features=image_features, 
                    tokenizer=tokenizer
                )
                print('Model output:', output_text)
                print('Running time:', generation_time)
            else:
                print("Model's chat method doesn't accept image_features parameter")
                # Try with the original image
                print("Trying with original image...")
                output_text, generation_time = model.chat(
                    prompt=prompt, 
                    image=image, 
                    tokenizer=tokenizer
                )
                print('Model output:', output_text)
                print('Running time:', generation_time)
        except Exception as e:
            print(f"Error using chat with processed features: {e}")
            # Try with the original image URL as fallback
            print("Falling back to original image URL...")
            try:
                output_text, generation_time = model.chat(
                    prompt=prompt, 
                    image=image_url, 
                    tokenizer=tokenizer
                )
                print('Model output:', output_text)
                print('Running time:', generation_time)
            except Exception as e:
                print(f"Error using chat with image URL: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("Model doesn't have vision_tower attribute")
        # Try with the original image URL as fallback
        print("Falling back to original image URL...")
        output_text, generation_time = model.chat(
            prompt=prompt, 
            image=image_url, 
            tokenizer=tokenizer
        )
        print('Model output:', output_text)
        print('Running time:', generation_time)
except Exception as e:
    print(f"Error during image processing: {e}")
    import traceback
    traceback.print_exc()
