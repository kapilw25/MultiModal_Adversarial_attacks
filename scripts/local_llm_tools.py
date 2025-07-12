"""
Local LLM tools for using Qwen2.5-VL-3B-Instruct model instead of OpenAI API.
This module provides a drop-in replacement for send_chat_request_azure function
that works with the local Qwen2.5-VL model.
"""

import os
import sys
import time
import tempfile
import base64

# Import decorators from llm_tools to avoid redundancy
from llm_tools import retry, timeout_decorator

# Add the parent directory to sys.path to import local_model modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the model factory function
from local_model.model_classes import create_model

# Global model instance to avoid reloading the model for each request
_model_instance = None

def get_model(engine="Qwen25_VL_3B"):
    """Get or create a model instance based on the engine name."""
    global _model_instance
    
    if _model_instance is None:
        if engine == "Qwen25_VL_3B":
            model_name = "Qwen2.5-VL-3B-Instruct_4bit"
            _model_instance = create_model(model_name)
        elif engine == "Gemma3_VL_4B":
            model_name = "Gemma-3-4b-it_4bit"
            _model_instance = create_model(model_name)
        else:
            raise ValueError(f"Unsupported engine: {engine}")
    
    return _model_instance


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
    Send a chat request to the local Qwen2.5-VL model.
    
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
