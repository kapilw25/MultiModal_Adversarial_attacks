import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Lazy imports to avoid transformers compatibility issues between environments
# Models are imported only when needed to prevent version conflicts

# Factory function to create model instances
def create_model(model_name):
    """Create a model instance based on model name with lazy imports"""
    if model_name == "Qwen2.5-VL-3B-Instruct_4bit":
        from local_model.models.Qwen25_3B import QwenVLModelWrapper
        return QwenVLModelWrapper(model_name)
    elif model_name == "Qwen2.5-VL-7B-Instruct-4bit":
        from local_model.models.Qwen25_7B import Qwen25VL7BModelWrapper
        return Qwen25VL7BModelWrapper(model_name)
    elif model_name == "Qwen2-VL-2B-Instruct_4bit":
        from local_model.models.Qwen2_2B import Qwen2VL2BModelWrapper
        return Qwen2VL2BModelWrapper(model_name)
    elif model_name == "Gemma-3-4b-it_4bit":
        from local_model.models.Gemma3_4B import GemmaVLModelWrapper
        return GemmaVLModelWrapper(model_name)
    elif model_name == "PaliGemma-3B-mix-224_4bit":
        from local_model.models.Paligemma_3B import PaliGemmaModelWrapper
        return PaliGemmaModelWrapper(model_name)
    # DeepSeek-VL models
    elif model_name == "DeepSeek-VL-1.3B-chat_4bit":
        from local_model.models.DeepSeek1_1pt3B import DeepSeekVL1pt3BModelWrapper
        return DeepSeekVL1pt3BModelWrapper(model_name)
    elif model_name == "DeepSeek-VL-7B-chat_4bit":
        from local_model.models.DeepSeek1_7B import DeepSeekVL7BModelWrapper
        return DeepSeekVL7BModelWrapper(model_name)
    # SmolVLM2 models
    elif "SmolVLM2-256M" in model_name:
        from local_model.models.SmolVLM2_pt25B_pt5B_2pt2B import SmolVLM2ModelWrapper
        return SmolVLM2ModelWrapper(model_name)
    elif "SmolVLM2-500M" in model_name:
        from local_model.models.SmolVLM2_pt25B_pt5B_2pt2B import SmolVLM2ModelWrapper
        return SmolVLM2ModelWrapper(model_name)
    elif "SmolVLM2-2.2B" in model_name:
        from local_model.models.SmolVLM2_pt25B_pt5B_2pt2B import SmolVLM2ModelWrapper
        return SmolVLM2ModelWrapper(model_name)
    # GLM Edge model
    elif model_name == "GLM-Edge-V-2B":
        from local_model.models.GLMEdge_2B import GLMEdgeModelWrapper
        return GLMEdgeModelWrapper(model_name)
    # InternVL3 models
    elif model_name == "InternVL3-1B":
        from local_model.models.InternVL3_1B_2B import InternVL3ModelWrapper
        return InternVL3ModelWrapper(model_name)
    elif model_name == "InternVL3-2B":
        from local_model.models.InternVL3_1B_2B import InternVL3ModelWrapper
        return InternVL3ModelWrapper(model_name)
    # InternVL2.5 model
    elif model_name == "InternVL2.5-4B":
        from local_model.models.InternVL25_4B import InternVL2_5_4BModelWrapper
        return InternVL2_5_4BModelWrapper(model_name)
    # Florence-2 models
    elif model_name == "Florence-2-base":
        from local_model.models.Florence2_pt23B_pt77B import Florence2ModelWrapper
        return Florence2ModelWrapper(model_name)
    elif model_name == "Florence-2-large":
        from local_model.models.Florence2_pt23B_pt77B import Florence2ModelWrapper
        return Florence2ModelWrapper(model_name)
    # Moondream2 model
    elif model_name == "Moondream2-2B":
        from local_model.models.Moondream2_2B import Moondream2ModelWrapper
        return Moondream2ModelWrapper(model_name)
    # Phi-3.5-vision model
    elif model_name == "Phi-3.5-vision-instruct-4bit":
        from local_model.models.Phi3pt5_vision_4B import Phi35VisionModelWrapper
        return Phi35VisionModelWrapper(model_name)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
