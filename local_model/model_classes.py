import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the QwenVLModelWrapper class
from .qwen_model import QwenVLModelWrapper

# Factory function to create model instances
def create_model(model_name):
    """Create a model instance based on model name"""
    if model_name == "Qwen2.5-VL-3B-Instruct_4bit":
        return QwenVLModelWrapper(model_name)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
