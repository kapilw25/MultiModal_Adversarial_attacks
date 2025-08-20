# Requirements
# The code of Qwen2.5-VL has been in the latest Hugging face transformers and we advise you to build from source with command:
# ```
# pip install git+https://github.com/huggingface/transformers accelerate
# ```

# or you might encounter the following error:
# ```
# KeyError: 'qwen2_5_vl'
# ```
# Quickstart
# Below, we provide simple examples to show how to use Qwen2.5-VL with 🤖 ModelScope and 🤗 Transformers.

# The code of Qwen2.5-VL has been in the latest Hugging face transformers and we advise you to build from source with command:
# ```
# pip install git+https://github.com/huggingface/transformers accelerate
# ```
# or you might encounter the following error:
# ```
# KeyError: 'qwen2_5_vl'
# ```
# We offer a toolkit to help you handle various types of visual input more conveniently, as if you were using an API. This includes base64, URLs, and interleaved images and videos. You can install it using the following command:

# # It's highly recommanded to use `[decord]` feature for faster video loading.
# ```
# pip install qwen-vl-utils[decord]==0.0.8
# ```

# If you are not using Linux, you might not be able to install decord from PyPI. In that case, you can use pip install qwen-vl-utils which will fall back to using torchvision for video processing. However, you can still install decord from source to get decord used when loading video.

# Using 🤗 Transformers to Chat
# Here we show a code snippet to show you how to use the chat model with transformers and qwen_vl_utils:

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import sys
import os
import time
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_utils import get_quantization_config, measure_memory_usage, cleanup_memory

def run_pytorch_baseline():
    """Run PyTorch baseline without SDPA"""
    print(f"\n{'='*60}")
    print("PYTORCH BASELINE (No SDPA)")
    print(f"{'='*60}")
    
    cleanup_memory()
    torch.cuda.reset_peak_memory_stats()
    
    print("Loading PyTorch model...")
    start_load_time = time.time()
    
    # Load model without SDPA - use default attention
    quantization_config = get_quantization_config(
        load_in_4bit=True,
        compute_dtype=torch.bfloat16,
        use_double_quant=True,
        quant_type="nf4"
    )
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
        # No attn_implementation specified - uses default
    )
    
    load_time = time.time() - start_load_time
    
    # Processor
    min_pixels = 128*28*28
    max_pixels = 256*28*28
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
    
    # Test messages
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    
    # Preparation for inference
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")
    
    # Memory before inference
    memory_before = torch.cuda.memory_allocated() / (1024**3)  # GB
    
    # Inference timing
    print("Running PyTorch inference...")
    start_inference_time = time.time()
    
    try:
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        inference_time = time.time() - start_inference_time
        
        # Memory after inference
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        memory_after = torch.cuda.memory_allocated() / (1024**3)  # GB
        
        # Generate output
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        success = True
        error_msg = None
        
    except Exception as e:
        inference_time = time.time() - start_inference_time
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        memory_after = torch.cuda.memory_allocated() / (1024**3)  # GB
        output_text = None
        success = False
        error_msg = str(e)
    
    # Cleanup
    del model
    cleanup_memory()
    
    return {
        'method': 'PyTorch Baseline',
        'success': success,
        'error': error_msg,
        'load_time': load_time,
        'inference_time': inference_time,
        'memory_before_gb': memory_before,
        'memory_after_gb': memory_after,
        'peak_memory_gb': peak_memory,
        'output_text': output_text[0] if output_text else None
    }

def run_onnx_cuda_optimization():
    """Run ONNX Runtime with CUDAExecutionProvider"""
    print(f"\n{'='*60}")
    print("ONNX RUNTIME + CUDA EXECUTION PROVIDER")
    print(f"{'='*60}")
    
    try:
        from optimum.onnxruntime import ORTModelForVision2Seq
        from optimum.pipelines import pipeline
        
        cleanup_memory()
        torch.cuda.reset_peak_memory_stats()
        
        print("Loading ONNX model with CUDA provider...")
        start_load_time = time.time()
        
        # Export and load model with ONNX Runtime CUDA provider
        ort_model = ORTModelForVision2Seq.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            export=True,
            provider="CUDAExecutionProvider",
            use_io_binding=True  # Enable IOBinding for better performance
        )
        
        load_time = time.time() - start_load_time
        
        # Processor
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        
        # Memory before inference
        memory_before = torch.cuda.memory_allocated() / (1024**3)  # GB
        
        # Create pipeline with GPU device
        print("Running ONNX CUDA inference...")
        start_inference_time = time.time()
        
        pipe = pipeline(
            task="image-to-text",
            model=ort_model,
            tokenizer=processor.tokenizer,
            device="cuda:0"
        )
        
        # Run inference
        result = pipe("https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg")
        
        inference_time = time.time() - start_inference_time
        
        # Memory after inference
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        memory_after = torch.cuda.memory_allocated() / (1024**3)  # GB
        
        success = True
        error_msg = None
        output_text = result[0]['generated_text'] if result else None
        
    except Exception as e:
        load_time = 0
        inference_time = 0
        peak_memory = 0
        memory_before = 0
        memory_after = 0
        output_text = None
        success = False
        error_msg = str(e)
    
    # Cleanup
    cleanup_memory()
    
    return {
        'method': 'ONNX CUDA + IOBinding',
        'success': success,
        'error': error_msg,
        'load_time': load_time,
        'inference_time': inference_time,
        'memory_before_gb': memory_before,
        'memory_after_gb': memory_after,
        'peak_memory_gb': peak_memory,
        'output_text': output_text
    }

def run_onnx_tensorrt_optimization():
    """Run ONNX Runtime with TensorRT ExecutionProvider"""
    print(f"\n{'='*60}")
    print("ONNX RUNTIME + TENSORRT EXECUTION PROVIDER")
    print(f"{'='*60}")
    
    try:
        from optimum.onnxruntime import ORTModelForVision2Seq
        from optimum.pipelines import pipeline
        
        cleanup_memory()
        torch.cuda.reset_peak_memory_stats()
        
        print("Loading ONNX model with TensorRT provider...")
        start_load_time = time.time()
        
        # Export and load model with ONNX Runtime TensorRT provider
        ort_model = ORTModelForVision2Seq.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            export=True,
            provider="TensorrtExecutionProvider",
            use_io_binding=True  # Enable IOBinding for better performance
        )
        
        load_time = time.time() - start_load_time
        
        # Processor
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        
        # Memory before inference
        memory_before = torch.cuda.memory_allocated() / (1024**3)  # GB
        
        # Create pipeline with GPU device
        print("Running ONNX TensorRT inference...")
        start_inference_time = time.time()
        
        pipe = pipeline(
            task="image-to-text",
            model=ort_model,
            tokenizer=processor.tokenizer,
            device="cuda:0"
        )
        
        # Run inference
        result = pipe("https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg")
        
        inference_time = time.time() - start_inference_time
        
        # Memory after inference
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        memory_after = torch.cuda.memory_allocated() / (1024**3)  # GB
        
        success = True
        error_msg = None
        output_text = result[0]['generated_text'] if result else None
        
    except Exception as e:
        load_time = 0
        inference_time = 0
        peak_memory = 0
        memory_before = 0
        memory_after = 0
        output_text = None
        success = False
        error_msg = str(e)
    
    # Cleanup
    cleanup_memory()
    
    return {
        'method': 'ONNX TensorRT + IOBinding',
        'success': success,
        'error': error_msg,
        'load_time': load_time,
        'inference_time': inference_time,
        'memory_before_gb': memory_before,
        'memory_after_gb': memory_after,
        'peak_memory_gb': peak_memory,
        'output_text': output_text
    }

def print_optimization_comparison(results):
    """Print comprehensive comparison of all optimization methods"""
    print(f"\n{'='*100}")
    print("INFERENCE OPTIMIZATION COMPARISON")
    print(f"{'='*100}")
    
    # Header
    print(f"{'Method':<30} {'Success':<10} {'Load(s)':<10} {'Inference(s)':<12} {'Peak Mem(GB)':<12} {'Speedup':<10}")
    print(f"{'-'*100}")
    
    baseline_inference = None
    
    for result in results:
        success_str = "✅" if result['success'] else "❌"
        
        if result['success']:
            load_str = f"{result['load_time']:.2f}"
            inf_str = f"{result['inference_time']:.2f}"
            mem_str = f"{result['peak_memory_gb']:.2f}"
            
            # Calculate speedup vs baseline
            if result['method'] == 'PyTorch Baseline':
                baseline_inference = result['inference_time']
                speedup_str = "1.00x"
            elif baseline_inference:
                speedup = baseline_inference / result['inference_time']
                speedup_str = f"{speedup:.2f}x"
            else:
                speedup_str = "N/A"
        else:
            load_str = inf_str = mem_str = speedup_str = "FAILED"
        
        print(f"{result['method']:<30} {success_str:<10} {load_str:<10} {inf_str:<12} {mem_str:<12} {speedup_str:<10}")
    
    # Detailed results
    print(f"\n{'='*100}")
    print("DETAILED RESULTS")
    print(f"{'='*100}")
    
    for result in results:
        print(f"\n{result['method']}:")
        if result['success']:
            print(f"  ✅ Success")
            print(f"  📊 Load Time: {result['load_time']:.2f}s")
            print(f"  ⚡ Inference Time: {result['inference_time']:.2f}s")
            print(f"  💾 Peak Memory: {result['peak_memory_gb']:.2f}GB")
            print(f"  📝 Output Length: {len(result['output_text']) if result['output_text'] else 0} chars")
            if result['output_text']:
                print(f"  📄 Output: {result['output_text'][:100]}...")
        else:
            print(f"  ❌ Failed: {result['error']}")

def install_onnx_dependencies():
    """Install required ONNX Runtime dependencies"""
    print("Installing ONNX Runtime GPU dependencies...")
    try:
        import subprocess
        subprocess.run(["pip", "install", "optimum[onnxruntime-gpu]"], check=True)
        print("✅ ONNX Runtime GPU dependencies installed successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

if __name__ == "__main__":
    print("🚀 INFERENCE OPTIMIZATION BENCHMARK")
    print("Testing PyTorch vs ONNX Runtime optimizations")
    
    # Install dependencies
    if not install_onnx_dependencies():
        print("⚠️  Continuing with available optimizations...")
    
    results = []
    
    # Test 1: PyTorch Baseline (no SDPA)
    print("\n🔥 Running PyTorch Baseline...")
    results.append(run_pytorch_baseline())
    
    # Test 2: ONNX Runtime with CUDA
    print("\n🔥 Running ONNX CUDA Optimization...")
    results.append(run_onnx_cuda_optimization())
    
    # Test 3: ONNX Runtime with TensorRT
    print("\n🔥 Running ONNX TensorRT Optimization...")
    results.append(run_onnx_tensorrt_optimization())
    
    # Print comprehensive comparison
    print_optimization_comparison(results)
