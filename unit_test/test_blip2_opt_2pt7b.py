# pip install accelerate bitsandbytes
import torch
import requests
import os
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map="auto")

# Test with original image
print("=== Testing with original image ===")
img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

question = "Explain the image"
inputs = processor(raw_image, question, return_tensors="pt").to("cuda", torch.float16)

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True).strip())

# Test with chart image
print("\n=== Testing with chart image ===")
# Hardcoded image path and question
IMAGE_PATH = "data/test_extracted/chart/20231114102825506748.png"
QUESTION = "What is shown in this chart?"

# Get absolute path for the image
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
image_path = os.path.join(project_root, IMAGE_PATH)

# Load the image
chart_image = Image.open(image_path).convert('RGB')

chart_inputs = processor(chart_image, QUESTION, return_tensors="pt").to("cuda", torch.float16)

chart_out = model.generate(**chart_inputs)
print(QUESTION)
print(processor.decode(chart_out[0], skip_special_tokens=True).strip())
