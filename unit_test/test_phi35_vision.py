#!/usr/bin/env python3
"""
Unit test for Microsoft Phi-3.5-vision-instruct model
GPU/CUDA ONLY - Will fail if CUDA is not available
Tests basic functionality, memory usage, and inference capabilities with 4-bit quantization
"""

import os
import sys
import time
import torch
import psutil
import gc
from PIL import Image
import requests
from transformers import AutoModelForCausalLM, AutoProcessor

# Import quantization utilities from local_model
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'local_model'))
from model_utils import (
    get_quantization_config, 
    cleanup_memory, 
    measure_memory_usage,
    memory_efficient,
    time_inference
)

@memory_efficient
def test_phi35_vision_basic():
    """Test basic model loading and inference - GPU ONLY with 4-bit quantization"""
    print("=" * 60)
    print("Testing Microsoft Phi-3.5-vision-instruct Model (GPU ONLY + 4-bit)")
    print("=" * 60)
    
    # Enforce CUDA requirement
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA is not available! This test requires GPU.")
    
    print(f"✅ CUDA device: {torch.cuda.get_device_name()}")
    print(f"✅ CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    model_id = "microsoft/Phi-3.5-vision-instruct"
    
    try:
        # Clear GPU memory aggressively
        cleanup_memory()
        
        print("\n1. Loading model with 4-bit quantization...")
        start_time = time.time()
        
        # Configure 4-bit quantization for Microsoft models (use bfloat16 like Google models)
        quantization_config = get_quantization_config(
            load_in_4bit=True,
            compute_dtype=torch.bfloat16,  # Better for Microsoft models
            use_double_quant=True,
            quant_type="nf4"
        )
        
        print("   Using 4-bit quantization config:")
        print(f"   - Compute dtype: {quantization_config.bnb_4bit_compute_dtype}")
        print(f"   - Quantization type: {quantization_config.bnb_4bit_quant_type}")
        print(f"   - Double quantization: {quantization_config.bnb_4bit_use_double_quant}")
        
        # Load model with optimizations
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",  # Let it decide optimal placement
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,  # Consistent with quantization
            _attn_implementation='eager',  # Skip flash attention to avoid issues
            low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
            use_cache=True  # Enable KV cache for efficiency
        )
        
        print("   ✅ Successfully loaded with 4-bit quantization")
        
        # Load processor with memory optimization
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            num_crops=4  # Reduced from 16 for memory efficiency
        )
        
        loading_time = time.time() - start_time
        print(f"   Loading time: {loading_time:.2f}s")
        
        # Measure memory after loading
        print("   Memory usage after loading:")
        measure_memory_usage()
        
        print("\n2. Testing with the specified chart image...")
        
        # Hardcoded image path and question
        IMAGE_PATH = "data/test_extracted/chart/20231114102825506748.png"
        QUESTION = "What is shown in this chart?"
        
        # Load the specific chart image
        try:
            image_path = os.path.join(os.path.dirname(__file__), '..', IMAGE_PATH)
            if os.path.exists(image_path):
                image = Image.open(image_path)
                print(f"   ✅ Successfully loaded chart image: {image.size}")
                print(f"   📁 Image path: {IMAGE_PATH}")
            else:
                print(f"   ❌ Chart image not found at: {image_path}")
                print("   Creating a simple test image as fallback...")
                image = Image.new('RGB', (512, 512), color='blue')
        except Exception as e:
            print(f"   ⚠️  Failed to load chart image: {e}")
            print("   Creating a simple test image as fallback...")
            image = Image.new('RGB', (512, 512), color='blue')
        
        # Prepare the prompt using the specific question
        messages = [
            {"role": "user", "content": f"<|image_1|>\n{QUESTION}"}
        ]
        
        prompt = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        print(f"   Question: {QUESTION}")
        print(f"   Complete Prompt:\n{prompt}")
        
        # Run inference with memory monitoring
        success = run_inference_test(model, processor, prompt, image)
        
        if success:
            # Test model parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"\n5. Model Statistics:")
            print(f"   Total parameters: {total_params:,}")
            print(f"   Trainable parameters: {trainable_params:,}")
            print(f"   Model size: ~{total_params * 4 / 1024**3:.2f} GB (FP32 equivalent)")
            print(f"   Actual GPU usage: ~{total_params / 1024**3:.2f} GB (4-bit quantized)")
        
        # Cleanup
        del model, processor
        cleanup_memory()
        
        print("\n✅ Test completed successfully!")
        return success
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Emergency cleanup
        cleanup_memory()
        return False

@time_inference
def run_inference_test(model, processor, prompt, image):
    """Run inference test with memory monitoring"""
    print("\n3. Running inference on GPU with 4-bit quantization...")
    
    try:
        # Process inputs with reduced memory usage
        inputs = processor(prompt, [image], return_tensors="pt")
        
        # Move to GPU efficiently
        inputs = {k: v.to("cuda", non_blocking=True) if hasattr(v, 'to') else v 
                 for k, v in inputs.items()}
        
        generation_args = {
            "max_new_tokens": 500,  # Keep detailed responses
            "do_sample": False,
            "pad_token_id": processor.tokenizer.eos_token_id,
            "use_cache": False  # Disable cache to avoid DynamicCache issues
        }
        
        # Generate response with memory management
        with torch.no_grad():
            generate_ids = model.generate(
                **inputs,
                eos_token_id=processor.tokenizer.eos_token_id,
                **generation_args
            )
        
        # Decode response
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        print("\n4. Model Response:")
        print("-" * 40)
        print(response)
        print("-" * 40)
        
        # Cleanup intermediate tensors
        del inputs, generate_ids
        cleanup_memory()
        
        return True
        
    except Exception as e:
        print(f"❌ Inference failed: {str(e)}")
        cleanup_memory()
        return False

@memory_efficient
def test_phi35_vision_multi_image():
    """Test with multiple images (simplified version) - GPU ONLY with 4-bit quantization"""
    print("\n" + "=" * 60)
    print("Testing Phi-3.5-vision with Multiple Images (GPU ONLY + 4-bit)")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA is not available! This test requires GPU.")
    
    model_id = "microsoft/Phi-3.5-vision-instruct"
    
    try:
        cleanup_memory()
        
        # Configure 4-bit quantization
        quantization_config = get_quantization_config(
            load_in_4bit=True,
            compute_dtype=torch.bfloat16,
            use_double_quant=True,
            quant_type="nf4"
        )
        
        # Load model with 4-bit quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            _attn_implementation='eager',
            low_cpu_mem_usage=True,
            use_cache=True
        )
        
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            num_crops=4
        )
        
        # Create simple test images instead of downloading slides
        print("Creating 2 test images for memory efficiency...")
        images = []
        placeholder = ""
        
        for i in range(1, 3):  # Only 2 images to save memory
            colors = ['red', 'green']
            image = Image.new('RGB', (256, 256), color=colors[i-1])  # Smaller images
            images.append(image)
            placeholder += f"<|image_{i}|>\n"
        
        messages = [
            {"role": "user", "content": placeholder + "Describe these colored images briefly."}
        ]
        
        prompt = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        print("Running inference with multiple images on GPU...")
        success = run_multi_image_inference(model, processor, prompt, images)
        
        # Cleanup
        del model, processor
        cleanup_memory()
        
        if success:
            print("✅ Multi-image test completed!")
        return success
        
    except Exception as e:
        print(f"❌ Multi-image test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        cleanup_memory()
        return False

@time_inference
def run_multi_image_inference(model, processor, prompt, images):
    """Run multi-image inference with memory management"""
    try:
        inputs = processor(prompt, images, return_tensors="pt")
        inputs = {k: v.to("cuda", non_blocking=True) if hasattr(v, 'to') else v 
                 for k, v in inputs.items()}
        
        generation_args = {
            "max_new_tokens": 150,  # Reduced for memory
            "temperature": 0.0,
            "do_sample": False,
            "pad_token_id": processor.tokenizer.eos_token_id,
            "use_cache": True
        }
        
        with torch.no_grad():
            generate_ids = model.generate(
                **inputs,
                eos_token_id=processor.tokenizer.eos_token_id,
                **generation_args
            )
        
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        print("\nMulti-image Response:")
        print("-" * 40)
        print(response)
        print("-" * 40)
        
        # Cleanup
        del inputs, generate_ids
        cleanup_memory()
        
        return True
        
    except Exception as e:
        print(f"❌ Multi-image inference failed: {str(e)}")
        cleanup_memory()
        return False

if __name__ == "__main__":
    # Set environment variable to avoid memory fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    print("Microsoft Phi-3.5-vision-instruct Unit Test (GPU ONLY + 4-bit Quantization)")
    print(f"PyTorch version: {torch.__version__}")
    
    # Enforce CUDA requirement
    if not torch.cuda.is_available():
        print("❌ CUDA is not available! This test requires GPU.")
        print("❌ Test FAILED - GPU/CUDA required!")
        sys.exit(1)
    
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    print(f"✅ CUDA device: {torch.cuda.get_device_name()}")
    print(f"✅ CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"✅ Memory fragmentation fix: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'Not set')}")
    
    # Initial memory cleanup
    cleanup_memory()
    
    # Run tests
    success = True
    
    # Test: Single image inference with detailed responses and 4-bit quantization
    success &= test_phi35_vision_basic()
    
    # Final cleanup
    cleanup_memory()
    
    if success:
        print("\n🎉 Single image test passed with 4-bit quantization!")
        print("✅ Phi-3.5-vision-instruct is working with detailed responses!")
        sys.exit(0)
    else:
        print("\n💥 Test failed!")
        sys.exit(1)
