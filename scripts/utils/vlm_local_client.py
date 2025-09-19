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
import torch
import gc
import threading
import psutil
from datetime import datetime

# Add the parent directory to sys.path to import local_model modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import decorators from cloud_model_utils to avoid redundancy
from vlm_cloud_client import retry, timeout_decorator

import sys
import os
# Add parent directory to path to access local_model
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from local_model.model_classes import create_model

from text_cleaner import clean_vlm_response

# Global model instances dictionary to avoid reloading models for each request
_model_instances = {}
_model_lock = threading.Lock()  # Thread-safe access to model instances

# Define model mapping once at the module level - Performance-optimized models only
# IMPORTANT: These names must exactly match the model names in model_classes.py
# Matches the exact sequence from local_model/test_vlm.py for consistent performance
MODEL_MAPPING = {
    # Qwen models (1-3)
    "Qwen25_VL_3B": "Qwen2.5-VL-3B-Instruct_4bit",
    "Qwen25_VL_7B": "Qwen2.5-VL-7B-Instruct-4bit",
    "Qwen2_VL_2B": "Qwen2-VL-2B-Instruct_4bit",
    
    # Google models (4-5)
    "Gemma3_VL_4B": "Gemma-3-4b-it_4bit",
    "PaliGemma_VL_3B": "PaliGemma-3B-mix-224_4bit",
    
    # DeepSeek models (6-7)
    "DeepSeek1_VL_1pt3B": "DeepSeek-VL-1.3B-chat_4bit",
    "DeepSeek1_VL_7B": "DeepSeek-VL-7B-chat_4bit",
    
    # SmolVLM2 models (8-10)
    "SmolVLM2_pt25B": "SmolVLM2-256M-Video-Instruct",
    "SmolVLM2_pt5B": "SmolVLM2-500M-Video-Instruct",
    "SmolVLM2_2pt2B": "SmolVLM2-2.2B-Instruct",
    
    # InternVL models (11-13)
    "InternVL3_1B": "InternVL3-1B",
    "InternVL3_2B": "InternVL3-2B",
    "InternVL25_4B": "InternVL2.5-4B",
    
    # Florence models - DISABLED: Not suitable for VQA tasks
    # "Florence2_pt23B": "Florence-2-base",
    # "Florence2_pt77B": "Florence-2-large",
    
    # Other fast models (14-16)
    "Moondream2_2B": "Moondream2-2B",
    "LLAVA_1pt5_7B": "LLAVA-1.5-7B",
    "LLAVA_v1pt6_Mistral_7B": "LLAVA-v1.6-Mistral-7B"
    
    # EXCLUDED - Slow performance models:
    # "Phi3pt5_vision_4B": "Phi-3.5-vision-instruct-4bit",  # 60s inference time
    # "GLMEdge_2B": "GLM-Edge-V-2B",  # 31s inference + quantization issues
}

def cleanup_gpu_memory():
    """Clean up GPU memory to prevent out-of-memory errors"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    print("GPU memory cleaned up")


def unload_all_models():
    """Unload all cached models and free GPU memory"""
    global _model_instances
    
    if not _model_instances:
        return
    
    print(f"Unloading {len(_model_instances)} cached models...")
    
    # Delete all model instances
    for engine_name, model in _model_instances.items():
        try:
            # Try to move model to CPU if it has parameters
            if hasattr(model, 'model') and hasattr(model.model, 'cpu'):
                model.model.cpu()
            elif hasattr(model, 'cpu'):
                model.cpu()
        except Exception as e:
            print(f"Warning: Could not move {engine_name} to CPU: {e}")
        
        # Delete the model reference
        del model
    
    # Clear the global cache
    _model_instances.clear()
    
    # Aggressive GPU cleanup
    cleanup_gpu_memory()
    
    print("All VLM models unloaded from GPU memory")


def unload_model(engine):
    """Unload a specific model from memory"""
    global _model_instances
    
    if engine not in _model_instances:
        return
    
    print(f"Unloading model: {engine}")
    
    try:
        model = _model_instances[engine]
        # Try to move model to CPU if possible
        if hasattr(model, 'model') and hasattr(model.model, 'cpu'):
            model.model.cpu()
        elif hasattr(model, 'cpu'):
            model.cpu()
        
        # Delete the model
        del _model_instances[engine]
        del model
        
        # Clean up GPU memory
        cleanup_gpu_memory()
        
        print(f"Model {engine} unloaded successfully")
        
    except Exception as e:
        print(f"Warning: Error unloading {engine}: {e}")
        # Still try to remove from cache
        if engine in _model_instances:
            del _model_instances[engine]


def get_gpu_memory_info():
    """Get current GPU memory usage information"""
    if not torch.cuda.is_available():
        return "CUDA not available"
    
    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    reserved = torch.cuda.memory_reserved() / 1024**3   # GB
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
    
    return f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total"


def get_detailed_memory_metrics():
    """Get detailed memory metrics for performance tracking"""
    metrics = {}
    
    # GPU Memory
    if torch.cuda.is_available():
        metrics["gpu"] = {
            "allocated_mb": torch.cuda.memory_allocated() / (1024**2),
            "reserved_mb": torch.cuda.memory_reserved() / (1024**2),
            "peak_mb": torch.cuda.max_memory_allocated() / (1024**2),
            "total_mb": torch.cuda.get_device_properties(0).total_memory / (1024**2)
        }
    else:
        metrics["gpu"] = {"allocated_mb": 0, "reserved_mb": 0, "peak_mb": 0, "total_mb": 0}
    
    # CPU Memory
    process = psutil.Process(os.getpid())
    metrics["cpu"] = {
        "memory_mb": process.memory_info().rss / (1024**2)
    }
    
    return metrics


def get_model(engine="Qwen25_VL_3B"):
    """Get or create a model instance based on the engine name (thread-safe)."""
    global _model_instances
    
    # Thread-safe access to model instances
    with _model_lock:
        # Return existing model instance if already loaded
        if engine in _model_instances:
            print(f"Using cached model: {engine} (thread: {threading.current_thread().name})")
            print(get_gpu_memory_info())
            return _model_instances[engine]
        
        # Check if engine is supported
        if engine not in MODEL_MAPPING:
            raise ValueError(f"Unsupported engine: {engine}. Available engines: {', '.join(MODEL_MAPPING.keys())}")
        
        # Show memory before loading
        print(f"Loading new model: {engine} (thread: {threading.current_thread().name})")
        print(f"Before loading - {get_gpu_memory_info()}")
        
        # Create model instance
        model_name = MODEL_MAPPING[engine]
        _model_instances[engine] = create_model(model_name)
        
        # Show memory after loading
        print(f"After loading - {get_gpu_memory_info()}")
        
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
        collect_metrics=False,
):
    """
    Send a chat request to the local vision-language model.
    
    Thread-safe: Model access is protected by locks, but concurrent inference 
    on the same model may still share GPU memory and computational resources.
    
    Args:
        message_text: List of message dictionaries with role and content
        engine: Model engine to use (default: "Qwen25_VL_3B")
        temp: Temperature for sampling (default: 0.2)
        logit_bias: Logit bias dictionary (not used for local model)
        max_new_token: Maximum number of new tokens to generate (default: 4096)
        sample_n: Number of samples to generate (default: 1, only 1 is supported)
        collect_metrics: Whether to collect performance metrics (default: False)
    
    Returns:
        Tuple of (response_text, [response_text]) or (response_text, [response_text], metrics_dict)
    """
    if sample_n != 1:
        print("Warning: sample_n > 1 is not supported for local model. Using sample_n=1.")
    
    # Initialize performance metrics if requested
    performance_metrics = None
    if collect_metrics:
        performance_metrics = {
            "inference_time_seconds": 0.0,
            "gpu_memory": {"before_inference_mb": 0, "after_inference_mb": 0, "peak_mb": 0, "reserved_mb": 0, "total_gpu_mb": 0},
            "cpu_memory": {"before_inference_mb": 0, "after_inference_mb": 0},
            "model_loading": {"was_cached": False, "loading_time_seconds": 0.0},
            "batch_info": {"batch_size": 1, "position_in_batch": 1, "total_batch_questions": 1}
        }
        start_time = time.time()
        memory_before = get_detailed_memory_metrics()
        performance_metrics["gpu_memory"]["before_inference_mb"] = memory_before["gpu"]["allocated_mb"]
        performance_metrics["cpu_memory"]["before_inference_mb"] = memory_before["cpu"]["memory_mb"]
    
    # Check if model is already cached
    model_was_cached = engine in _model_instances
    model_load_start = time.time()
    
    # Get the model instance
    model = get_model(engine)
    
    # Record model loading metrics
    if collect_metrics:
        performance_metrics["model_loading"]["was_cached"] = model_was_cached
        performance_metrics["model_loading"]["loading_time_seconds"] = 0.0 if model_was_cached else (time.time() - model_load_start)
    
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
    
    # Run prediction with error handling
    try:
        response = model.predict(image_url, prompt_text)
        
        # Clean the response to remove verbosity and format artifacts
        cleaned_response = clean_vlm_response(response)
        
        # Check if cleaned response indicates a CUDA error
        if isinstance(cleaned_response, tuple) and len(cleaned_response) >= 1:
            if "ERROR: CUDA device-side assert" in str(cleaned_response[0]):
                print(f"⚠️  CUDA error handled gracefully for engine: {engine}")
                if collect_metrics:
                    return "CUDA_ERROR_HANDLED", ["CUDA_ERROR_HANDLED"], performance_metrics
                return "CUDA_ERROR_HANDLED", ["CUDA_ERROR_HANDLED"]
        elif "ERROR: CUDA device-side assert" in str(cleaned_response):
            print(f"⚠️  CUDA error handled gracefully for engine: {engine}")
            if collect_metrics:
                return "CUDA_ERROR_HANDLED", ["CUDA_ERROR_HANDLED"], performance_metrics
            return "CUDA_ERROR_HANDLED", ["CUDA_ERROR_HANDLED"]
            
    except Exception as e:
        if "CUDA error: device-side assert triggered" in str(e):
            print(f"⚠️  CUDA device-side assert caught at VLM client level for engine: {engine}")
            print(f"Error: {e}")
            if collect_metrics:
                return "CUDA_ERROR_HANDLED", ["CUDA_ERROR_HANDLED"], performance_metrics
            return "CUDA_ERROR_HANDLED", ["CUDA_ERROR_HANDLED"]
        else:
            # Re-raise other exceptions
            raise e
    
    response = cleaned_response
    
    # Collect final performance metrics
    if collect_metrics:
        memory_after = get_detailed_memory_metrics()
        performance_metrics["inference_time_seconds"] = time.time() - start_time
        performance_metrics["gpu_memory"]["after_inference_mb"] = memory_after["gpu"]["allocated_mb"]
        performance_metrics["gpu_memory"]["peak_mb"] = memory_after["gpu"]["peak_mb"]
        performance_metrics["gpu_memory"]["reserved_mb"] = memory_after["gpu"]["reserved_mb"]
        performance_metrics["gpu_memory"]["total_gpu_mb"] = memory_after["gpu"]["total_mb"]
        performance_metrics["cpu_memory"]["after_inference_mb"] = memory_after["cpu"]["memory_mb"]
    
    # Clean up temporary file if created
    if image_url.startswith("/tmp/"):
        try:
            os.remove(image_url)
        except:
            pass
    
    # Return the response in the same format as send_chat_request_azure
    if collect_metrics:
        return response, [response], performance_metrics
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
