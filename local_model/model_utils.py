import torch
import gc
import time
import os
import psutil
from functools import wraps
from transformers import BitsAndBytesConfig

def cleanup_memory():
    """Clean up GPU memory and garbage collection"""
    torch.cuda.empty_cache()
    gc.collect()

def get_device():
    """Get the appropriate device (CUDA or CPU)"""
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_quantization_config(load_in_4bit=True, compute_dtype=torch.float16, use_double_quant=True, quant_type="nf4"):
    """Configure 4-bit quantization settings"""
    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_double_quant,
        bnb_4bit_quant_type=quant_type
    )

def get_8bit_quantization_config():
    """Configure 8-bit quantization settings"""
    return BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0
    )

def measure_memory_usage():
    """Measure current memory usage"""
    # CPU memory
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / (1024 * 1024)  # MB
    
    # GPU memory if available
    gpu_mem = 0
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        gpu_max = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
        gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)  # MB
        gpu_reserved = torch.cuda.memory_reserved() / (1024 * 1024)  # MB
        
        print(f"GPU Memory: {gpu_mem:.2f} MB (Current) / {gpu_max:.2f} MB (Peak) / {gpu_total:.2f} MB (Total)")
        print(f"GPU Reserved: {gpu_reserved:.2f} MB")
    
    print(f"CPU Memory: {cpu_mem:.2f} MB")
    return cpu_mem, gpu_mem

def memory_efficient(func):
    """Decorator to make functions more memory efficient"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Measure memory before
        print(f"Memory before {func.__name__}:")
        measure_memory_usage()
        
        # Run garbage collection before function
        cleanup_memory()
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Run garbage collection after function
        cleanup_memory()
        
        # Measure memory after
        print(f"Memory after {func.__name__}:")
        measure_memory_usage()
        
        return result
    return wrapper

def load_model_with_timing(model_class, model_path, quantization_config=None, **kwargs):
    """Load a model with timing information"""
    print(f"Loading model from {model_path}...")
    start_time = time.time()
    
    # Measure memory before loading
    print("Memory before model loading:")
    measure_memory_usage()
    
    # Prepare loading arguments
    load_args = {"device_map": "auto"}
    if quantization_config:
        load_args["quantization_config"] = quantization_config
    load_args.update(kwargs)
    
    # Load the model
    model = model_class.from_pretrained(model_path, **load_args)
    
    # Measure memory after loading
    print("Memory after model loading:")
    measure_memory_usage()
    
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    return model

def time_inference(func):
    """Decorator to time inference operations"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        # Measure memory before inference
        print(f"Memory before inference:")
        measure_memory_usage()
        
        result = func(*args, **kwargs)
        
        # Measure memory after inference
        print(f"Memory after inference:")
        measure_memory_usage()
        
        inference_time = time.time() - start_time
        print(f"Inference completed in {inference_time:.2f} seconds")
        return result
    return wrapper

def get_processor_with_pixel_settings(processor_path, min_size=256, max_size=1280):
    """Load processor with recommended pixel settings"""
    from transformers import AutoProcessor
    
    min_pixels = min_size * 28 * 28
    max_pixels = max_size * 28 * 28
    
    processor = AutoProcessor.from_pretrained(
        processor_path,
        min_pixels=min_pixels,
        max_pixels=max_pixels
    )
    print(f"Processor loaded successfully from {processor_path}")
    return processor

def model_cleanup(model):
    """Clean up model resources"""
    del model
    cleanup_memory()
    print("Model resources cleaned up")
