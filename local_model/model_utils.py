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

def clear_model_memory_if_needed(engine, model_instance=None):
    """Clear model memory for engines prone to context bleeding/repetition
    
    Args:
        engine (str): Model engine name (e.g., 'LLAVA_v1pt6_Mistral_7B')
        model_instance: Optional model instance to clear specific attributes
        
    Returns:
        bool: True if memory clearing was applied, False otherwise
    """
    # List of models that need memory clearing between questions
    memory_sensitive_models = [
        'LLAVA_1pt5_7B',
        'LLAVA_v1pt6_Mistral_7B'
        # Add more models here as needed: 'DeepSeek1_1pt3B', etc.
    ]
    
    if engine in memory_sensitive_models:
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        # Clear model-specific memory states
        if model_instance:
            # Clear past key values that cause context bleeding
            if hasattr(model_instance, 'past_key_values'):
                model_instance.past_key_values = None
            if hasattr(model_instance, 'model') and hasattr(model_instance.model, 'past_key_values'):
                model_instance.model.past_key_values = None
            # Clear any cached generation states
            if hasattr(model_instance, '_generation_cache'):
                model_instance._generation_cache = None
        
        # Light garbage collection for problematic models
        gc.collect()
        return True
    
    return False


def robust_generate_with_cuda_handling(model, inputs, processor, model_name="Unknown", **generation_config):
    """
    Centralized robust generation with CUDA error handling for all VLM models.

    This function provides retry logic with progressively more conservative parameters
    to handle CUDA device-side assertion errors caused by probability tensor corruption.

    Args:
        model: The model instance with generate() method
        inputs: Preprocessed model inputs (tokenized, moved to device)
        processor: Model processor for tokenization
        model_name: Human-readable model name for logging
        **generation_config: Generation parameters (max_new_tokens, temperature, etc.)

    Returns:
        torch.Tensor: Generated token IDs

    Raises:
        RuntimeError: If all retry attempts fail or non-CUDA errors occur
    """
    print(f"Generating response with {model_name}...")

    generation_attempts = 0
    max_attempts = 3

    # Default generation config
    default_config = {
        "max_new_tokens": 128,
        "do_sample": True,
        "temperature": 0.3,
        "top_p": 0.95,
        "pad_token_id": processor.tokenizer.eos_token_id if hasattr(processor, 'tokenizer') else None,
        "eos_token_id": processor.tokenizer.eos_token_id if hasattr(processor, 'tokenizer') else None,
        "repetition_penalty": 1.1,
        "num_beams": 1,
        "early_stopping": True,
        "use_cache": True
    }

    # Merge with provided config
    final_config = {**default_config, **generation_config}

    while generation_attempts < max_attempts:
        try:
            with torch.inference_mode():
                # Adjust parameters based on retry attempt
                if generation_attempts > 0:
                    # Make parameters more conservative on retries
                    final_config["temperature"] = max(0.1, 0.3 - generation_attempts * 0.1)
                    final_config["top_p"] = min(0.99, 0.95 + generation_attempts * 0.02)

                # On final attempt, use deterministic generation to avoid probability issues
                if generation_attempts == max_attempts - 1:
                    print("⚠️  Using deterministic generation as fallback")
                    final_config.update({
                        "do_sample": False,
                        "temperature": None,
                        "top_p": None,
                        "num_beams": 1
                    })

                # Clean None values from config
                clean_config = {k: v for k, v in final_config.items() if v is not None}

                generated_ids = model.generate(**inputs, **clean_config)
                return generated_ids  # Success, return result

        except RuntimeError as e:
            error_msg = str(e)
            is_cuda_probability_error = (
                "probability tensor contains either `inf`, `nan` or element < 0" in error_msg or
                "CUDA error: device-side assert triggered" in error_msg
            )

            if is_cuda_probability_error:
                generation_attempts += 1
                print(f"⚠️  CUDA device-side assert detected (attempt {generation_attempts}/{max_attempts})")
                print(f"Error details: {e}")

                if generation_attempts >= max_attempts:
                    print(f"⚠️  All generation attempts failed. Returning error response.")
                    raise RuntimeError("ERROR: CUDA device-side assert - probability tensor corruption")

                # Clear CUDA cache before retry
                torch.cuda.empty_cache()
                print(f"🔄 Retrying with more conservative generation parameters...")

            else:
                # Re-raise non-CUDA errors immediately
                raise e

    # This should never be reached due to the raise above, but added for completeness
    raise RuntimeError("ERROR: Maximum retry attempts exceeded")


def robust_chat_with_cuda_handling(model, tokenizer, pixel_values, question, generation_config, model_name="Unknown", **chat_kwargs):
    """
    Centralized robust chat method with CUDA error handling for InternVL-style models.

    This function provides retry logic with progressively more conservative parameters
    to handle CUDA device-side assertion errors during the chat method.

    Args:
        model: The model instance with chat() method
        tokenizer: Model tokenizer
        pixel_values: Preprocessed image tensors
        question: Text question/prompt
        generation_config: Generation parameters dict
        model_name: Human-readable model name for logging
        **chat_kwargs: Additional chat method parameters (history, return_history, etc.)

    Returns:
        Tuple: (response, history) or just response depending on return_history parameter

    Raises:
        RuntimeError: If all retry attempts fail or non-CUDA errors occur
    """
    print(f"Generating response with {model_name} using chat method...")

    generation_attempts = 0
    max_attempts = 3

    # Default generation config for chat methods
    default_config = {
        "max_new_tokens": 128,
        "do_sample": True,
        "temperature": 0.3,
        "top_p": 0.95
    }

    # Merge with provided config
    final_config = {**default_config, **generation_config}

    while generation_attempts < max_attempts:
        try:
            with torch.inference_mode():
                # Adjust parameters based on retry attempt
                if generation_attempts > 0:
                    # Make parameters more conservative on retries
                    final_config["temperature"] = max(0.1, 0.3 - generation_attempts * 0.1)
                    final_config["top_p"] = min(0.99, 0.95 + generation_attempts * 0.02)

                # On final attempt, use deterministic generation to avoid probability issues
                if generation_attempts == max_attempts - 1:
                    print("⚠️  Using deterministic generation as fallback")
                    final_config.update({
                        "do_sample": False,
                        "temperature": 0.1,  # Use minimal temperature instead of None for chat
                        "top_p": 0.99
                    })

                chat_result = model.chat(
                    tokenizer,
                    pixel_values,
                    question,
                    final_config,
                    **chat_kwargs
                )
                return chat_result  # Success, return result

        except RuntimeError as e:
            error_msg = str(e)
            is_cuda_probability_error = (
                "probability tensor contains either `inf`, `nan` or element < 0" in error_msg or
                "CUDA error: device-side assert triggered" in error_msg
            )

            if is_cuda_probability_error:
                generation_attempts += 1
                print(f"⚠️  CUDA device-side assert detected in chat method (attempt {generation_attempts}/{max_attempts})")
                print(f"Error details: {e}")

                if generation_attempts >= max_attempts:
                    print(f"⚠️  All chat attempts failed. Returning error response.")
                    raise RuntimeError("ERROR: CUDA device-side assert - probability tensor corruption in chat")

                # Clear CUDA cache before retry
                torch.cuda.empty_cache()
                print(f"🔄 Retrying chat with more conservative generation parameters...")

            else:
                # Re-raise non-CUDA errors immediately
                raise e

    # This should never be reached due to the raise above, but added for completeness
    raise RuntimeError("ERROR: Maximum chat retry attempts exceeded")


def robust_query_with_cuda_handling(model, image, question, model_name="Unknown", **query_kwargs):
    """
    Centralized robust query method with CUDA error handling for Moondream-style models.

    This function provides retry logic to handle CUDA device-side assertion errors
    during the query method.

    Args:
        model: The model instance with query() method
        image: PIL Image object
        question: Text question/prompt
        model_name: Human-readable model name for logging
        **query_kwargs: Additional query method parameters

    Returns:
        Dict: Query result (typically contains 'answer' key)

    Raises:
        RuntimeError: If all retry attempts fail or non-CUDA errors occur
    """
    print(f"Generating response with {model_name} using query method...")

    generation_attempts = 0
    max_attempts = 3

    while generation_attempts < max_attempts:
        try:
            with torch.inference_mode():
                query_result = model.query(image, question, **query_kwargs)
                return query_result  # Success, return result

        except RuntimeError as e:
            error_msg = str(e)
            is_cuda_probability_error = (
                "probability tensor contains either `inf`, `nan` or element < 0" in error_msg or
                "CUDA error: device-side assert triggered" in error_msg
            )

            if is_cuda_probability_error:
                generation_attempts += 1
                print(f"⚠️  CUDA device-side assert detected in query method (attempt {generation_attempts}/{max_attempts})")
                print(f"Error details: {e}")

                if generation_attempts >= max_attempts:
                    print(f"⚠️  All query attempts failed. Returning error response.")
                    raise RuntimeError("ERROR: CUDA device-side assert - probability tensor corruption in query")

                # Clear CUDA cache before retry
                torch.cuda.empty_cache()
                print(f"🔄 Retrying query with CUDA recovery...")

            else:
                # Re-raise non-CUDA errors immediately
                raise e

    # This should never be reached due to the raise above, but added for completeness
    raise RuntimeError("ERROR: Maximum query retry attempts exceeded")


def safe_move_inputs_to_device(inputs, device, model_name="Unknown"):
    """
    Safely move model inputs to device with CUDA error handling.

    Args:
        inputs: Model inputs to move to device
        device: Target device (usually 'cuda')
        model_name: Human-readable model name for logging

    Returns:
        Moved inputs or raises error if CUDA corruption detected

    Raises:
        RuntimeError: If CUDA device-side assertion occurs during input movement
    """
    try:
        return inputs.to(device)
    except RuntimeError as e:
        if "CUDA error: device-side assert triggered" in str(e):
            print(f"⚠️  CUDA device-side assert error detected while moving inputs for {model_name}")
            print(f"Error details: {e}")
            raise RuntimeError("ERROR: CUDA device-side assert - corrupted input data")
        else:
            raise e

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
    """Clean up model resources with robust error handling for CUDA context corruption"""
    try:
        print("Starting model cleanup...")
        
        # Try to move model to CPU first to free GPU memory
        try:
            if hasattr(model, 'cpu'):
                model.cpu()
                print("Model moved to CPU")
        except Exception as e:
            print(f"Warning: Could not move model to CPU: {e}")
        
        # Delete model reference
        del model
        print("Model reference deleted")
        
        # Cleanup memory
        cleanup_memory()
        
        # Additional GPU context reset for CUDA errors
        if torch.cuda.is_available():
            try:
                # Force synchronization to catch any pending CUDA errors
                torch.cuda.synchronize()
                
                # Reset max memory stats
                torch.cuda.reset_max_memory_allocated()
                torch.cuda.reset_max_memory_cached()
                
                print("GPU context reset successfully")
            except RuntimeError as cuda_error:
                if "CUDA error" in str(cuda_error):
                    print(f"⚠️  CUDA context corruption detected during cleanup: {cuda_error}")
                    print("Attempting GPU context recovery...")
                    
                    # Try to reset CUDA context more aggressively
                    try:
                        torch.cuda.empty_cache()
                        # Reset accumulated CUDA errors
                        torch.cuda.synchronize()
                        print("GPU context recovery attempted")
                    except Exception as recovery_error:
                        print(f"❌ GPU context recovery failed: {recovery_error}")
                        print("System may require process restart to recover GPU context")
                else:
                    raise cuda_error
        
        print("Model resources cleaned up successfully")
        
    except Exception as cleanup_error:
        print(f"❌ Error during model cleanup: {cleanup_error}")
        print("Forcing aggressive cleanup...")
        
        # Force garbage collection even if cleanup failed
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        raise cleanup_error
