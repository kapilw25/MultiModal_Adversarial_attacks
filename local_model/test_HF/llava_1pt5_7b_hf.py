import requests
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "llava-hf/llava-1.5-7b-hf"
# model = LlavaForConditionalGeneration.from_pretrained(
#     model_id, 
#     torch_dtype=torch.float16, 
#     low_cpu_mem_usage=True, 
# ).to(0)

# 4-bit quantization through bitsandbytes library
# First make sure to install bitsandbytes, pip install bitsandbytes and make sure to have access to a CUDA compatible GPU device. Simply change the snippet above with:
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    load_in_4bit=True
)


processor = AutoProcessor.from_pretrained(model_id)

# Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 
conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": "Explain this image"},
          {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
raw_image = Image.open(requests.get(image_file, stream=True).raw)
inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)

output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))

print("\n" + "="*60)
print("MEDICAL X-RAY ANALYSIS TEST:")
print("="*60)

# Test with chest X-ray image - same one that MedGemma failed with
medical_conversation = [
    {
      "role": "user",
      "content": [
          {"type": "text", "text": "You are an expert radiologist. Please analyze this chest X-ray image and provide a detailed medical description of your findings."},
          {"type": "image"},
        ],
    },
]
medical_prompt = processor.apply_chat_template(medical_conversation, add_generation_prompt=True)

# Load chest X-ray
xray_image_url = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
xray_image = Image.open(requests.get(xray_image_url, headers={"User-Agent": "example"}, stream=True).raw)

# Resize for memory efficiency if needed
if xray_image.size[0] > 512 or xray_image.size[1] > 512:
    xray_image = xray_image.resize((512, 512), Image.Resampling.LANCZOS)

print(f"Processing X-ray image of size: {xray_image.size}")

medical_inputs = processor(images=xray_image, text=medical_prompt, return_tensors='pt').to(0, torch.float16)
medical_output = model.generate(**medical_inputs, max_new_tokens=200, do_sample=False)

print("CHEST X-RAY ANALYSIS:")
print(processor.decode(medical_output[0][2:], skip_special_tokens=True))
print("="*60)
