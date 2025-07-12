import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the model wrapper classes
from local_model.models.qwen_model import QwenVLModelWrapper
from local_model.models.gemma_model import GemmaVLModelWrapper

# Factory function to create model instances
def create_model(model_name):
    """Create a model instance based on model name"""
    if model_name == "Qwen2.5-VL-3B-Instruct_4bit":
        return QwenVLModelWrapper(model_name)
    elif model_name == "Gemma-3-4b-it_4bit":
        return GemmaVLModelWrapper(model_name)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
