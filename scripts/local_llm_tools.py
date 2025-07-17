"""
Local LLM tools for using vision-language models instead of OpenAI API.
This module provides a drop-in replacement for send_chat_request_azure function
that works with local VLM models.
"""

import os
import sys
import time
import tempfile
import base64

# Add the parent directory to sys.path to import local_model modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import decorators from llm_tools to avoid redundancy
from scripts.llm_tools import retry, timeout_decorator

# Import the model factory function
from local_model.model_classes import create_model

# Global model instances dictionary to avoid reloading models for each request
_model_instances = {}

# Define model mapping once at the module level
# IMPORTANT: These names must exactly match the model names in model_classes.py
MODEL_MAPPING = {
    # Qwen models
    "Qwen25_VL_3B": "Qwen2.5-VL-3B-Instruct_4bit",
    "Qwen25_VL_7B": "Qwen2.5-VL-7B-Instruct-4bit",  # Note the hyphen instead of underscore
    "Qwen2_VL_2B": "Qwen2-VL-2B-Instruct_4bit",
    
    # Google models
    "Gemma3_VL_4B": "Gemma-3-4b-it_4bit",
    "PaliGemma_VL_3B": "PaliGemma-3B-mix-224_4bit",
    
    # DeepSeek models
    "DeepSeek1_VL_1pt3B": "DeepSeek-VL-1.3B-chat_4bit",
    "DeepSeek1_VL_7B": "DeepSeek-VL-7B-chat_4bit",
    
    # SmolVLM2 models
    "SmolVLM2_pt25B": "SmolVLM2-256M-Video-Instruct",
    "SmolVLM2_pt5B": "SmolVLM2-500M-Video-Instruct",
    "SmolVLM2_2pt2B": "SmolVLM2-2.2B-Instruct",
    
    # Microsoft models
    "Phi3pt5_vision_4B": "Phi-3.5-vision-instruct-4bit",
    
    # Florence models
    "Florence2_pt23B": "Florence-2-base",
    "Florence2_pt77B": "Florence-2-large",
    
    # Other models
    "Moondream2_2B": "Moondream2-2B",
    "GLMEdge_2B": "GLM-Edge-V-2B",
    "InternVL3_1B": "InternVL3-1B",
    "InternVL3_2B": "InternVL3-2B",
    "InternVL25_4B": "InternVL2.5-4B"
}

def get_model(engine="Qwen25_VL_3B"):
    """Get or create a model instance based on the engine name."""
    global _model_instances
    
    # Return existing model instance if already loaded
    if engine in _model_instances:
        return _model_instances[engine]
    
    # Check if engine is supported
    if engine not in MODEL_MAPPING:
        raise ValueError(f"Unsupported engine: {engine}. Available engines: {', '.join(MODEL_MAPPING.keys())}")
    
    # Create model instance
    model_name = MODEL_MAPPING[engine]
    _model_instances[engine] = create_model(model_name)
    
    return _model_instances[engine]


@timeout_decorator(180)
@retry(Exception, tries=3, delay=5, backoff=1)
def send_chat_request(
        message_text,
        engine="Qwen25_VL_3B",
        temp=0.2,
        logit_bias: dict = {},
        max_new_token=4096,
        sample_n=1,
):
    """
    Send a chat request to the local vision-language model.
    
    Args:
        message_text: List of message dictionaries with role and content
        engine: Model engine to use (default: "Qwen25_VL_3B")
        temp: Temperature for sampling (default: 0.2)
        logit_bias: Logit bias dictionary (not used for local model)
        max_new_token: Maximum number of new tokens to generate (default: 4096)
        sample_n: Number of samples to generate (default: 1, only 1 is supported)
    
    Returns:
        Tuple of (response_text, [response_text])
    """
    if sample_n != 1:
        print("Warning: sample_n > 1 is not supported for local model. Using sample_n=1.")
    
    # Get the model instance
    model = get_model(engine)
    
    # Extract image URL and text from the message
    image_url = None
    prompt_text = ""
    
    for msg in message_text:
        if msg["role"] == "user":
            for content_item in msg["content"]:
                if content_item["type"] == "text":
                    prompt_text = content_item["text"]
                elif content_item["type"] == "image_url":
                    # Handle data URL format
                    url = content_item["image_url"]["url"]
                    if url.startswith("data:"):
                        # Extract base64 data and save to a temporary file
                        # Parse the data URL
                        header, encoded = url.split(",", 1)
                        data = base64.b64decode(encoded)
                        
                        # Create a temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                            temp_file.write(data)
                            image_url = temp_file.name
                    else:
                        # Assume it's a file path
                        image_url = url
    
    if not image_url or not prompt_text:
        raise ValueError("Both image and text must be provided in the message")
    
    # Run prediction
    response = model.predict(image_url, prompt_text)
    
    # Clean up temporary file if created
    if image_url.startswith("/tmp/"):
        try:
            os.remove(image_url)
        except:
            pass
    
    # Return the response in the same format as send_chat_request_azure
    return response, [response]


# Alias for compatibility with code that uses send_chat_request_azure
send_chat_request_azure = send_chat_request


def list_available_models():
    """
    Returns a list of all available models that can be used with the local_llm_tools module.
    
    Returns:
        list: List of model engine names
    """
    return list(MODEL_MAPPING.keys())
