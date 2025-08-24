import torch
import requests
from PIL import Image
from io import BytesIO
from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    AutoModelForCausalLM,
)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from local_model.model_utils import get_quantization_config

# Download image from URL
url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
response = requests.get(url)
image = Image.open(BytesIO(response.content))

messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "describe this image"}]}]

model_dir = "THUDM/glm-edge-v-5b"

processor = AutoImageProcessor.from_pretrained(model_dir, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
quantization_config = get_quantization_config(
    load_in_4bit=True,
    compute_dtype=torch.bfloat16,
    use_double_quant=True,
    quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
)

inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_dict=True, tokenize=True, return_tensors="pt"
).to(next(model.parameters()).device)

generate_kwargs = {
    **inputs,
    "pixel_values": torch.tensor(processor(image).pixel_values).to(next(model.parameters()).device),
}
output = model.generate(**generate_kwargs, max_new_tokens=100)
print(tokenizer.decode(output[0][len(inputs["input_ids"][0]):], skip_special_tokens=True))
