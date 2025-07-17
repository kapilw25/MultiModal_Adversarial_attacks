from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests
import os
from transformers import BitsAndBytesConfig

# Configure 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

print("Loading model...")
model = InstructBlipForConditionalGeneration.from_pretrained(
    "Salesforce/instructblip-flan-t5-xl",
    quantization_config=quantization_config
)
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")

device = "cuda"
model.to(device)

# Custom post-processing to handle any remaining repetitions
def remove_repetitions(text):
    # Split into sentences
    sentences = text.split('. ')
    unique_sentences = []
    
    # Keep only unique sentences
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and sentence not in unique_sentences:
            unique_sentences.append(sentence)
    
    # Rejoin with proper punctuation
    cleaned_text = '. '.join(unique_sentences)
    if not cleaned_text.endswith('.'):
        cleaned_text += '.'
    
    return cleaned_text

# Function to process an image and generate a response
def process_image_and_question(image, prompt):
    print(f"Processing with prompt: '{prompt}'")
    
    # Process inputs
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    
    # FLAN-T5 compatible generation parameters
    print("Generating response...")
    outputs = model.generate(
        **inputs,
        do_sample=False,
        num_beams=5,
        max_new_tokens=100,  # Use max_new_tokens instead of max_length
        min_length=10,
        repetition_penalty=2.0,  # Increased repetition penalty
        length_penalty=1.0,
        temperature=1.0,
        no_repeat_ngram_size=3,  # Prevent repeating 3-grams
        early_stopping=True  # Stop when all beams reach EOS
    )
    
    # Post-process the output
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    cleaned_text = remove_repetitions(generated_text)
    
    print(f"Original output length: {len(generated_text)}")
    print(f"Cleaned output length: {len(cleaned_text)}")
    
    return cleaned_text

# Test with original image
print("\n=== Testing with original image ===")
url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
original_image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
original_prompt = "What is unusual about this image?"

original_response = process_image_and_question(original_image, original_prompt)
print("\nResponse for original image:")
print(original_response)

# Test with chart image
print("\n=== Testing with chart image ===")
# Hardcoded image path and question
IMAGE_PATH = "data/test_extracted/chart/20231114102825506748.png"
QUESTION = "What is shown in this chart?"

# Get absolute path for the image
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
image_path = os.path.join(project_root, IMAGE_PATH)

# Check if the image exists
if os.path.exists(image_path):
    chart_image = Image.open(image_path).convert("RGB")
    chart_response = process_image_and_question(chart_image, QUESTION)
    print("\nResponse for chart image:")
    print(chart_response)
else:
    print(f"Error: Chart image not found at {image_path}")
