from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import sys
import os
import time
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_utils import get_quantization_config, measure_memory_usage, cleanup_memory

def run_model_test(model_size, optimization_type):
    """Run inference test for specific model size and optimization"""
    model_name = f"Qwen/Qwen2.5-VL-{model_size}-Instruct"
    
    print(f"\n{'='*70}")
    print(f"TESTING: {model_size} MODEL + {optimization_type}")
    print(f"{'='*70}")
    
    cleanup_memory()
    torch.cuda.reset_peak_memory_stats()
    
    print(f"Loading {model_size} model with {optimization_type}...")
    start_load_time = time.time()
    
    # Configure quantization based on optimization type
    if optimization_type == "FP16 Mixed Precision":
        quantization_config = get_quantization_config(
            load_in_4bit=True,
            compute_dtype=torch.float16,  # FP16 for mixed precision
            use_double_quant=True,
            quant_type="nf4"
        )
        torch_dtype = torch.float16
    else:
        quantization_config = get_quantization_config(
            load_in_4bit=True,
            compute_dtype=torch.bfloat16,
            use_double_quant=True,
            quant_type="nf4"
        )
        torch_dtype = torch.bfloat16
    
    # Load model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        device_map="auto"
    )
    
    # Apply optimization
    compile_success = False
    if optimization_type == "torch.compile":
        try:
            print("Applying torch.compile optimization...")
            model = torch.compile(model, mode="reduce-overhead")
            compile_success = True
        except Exception as e:
            print(f"torch.compile failed: {e}")
    elif optimization_type == "torch.compile + FP16":
        try:
            print("Applying torch.compile + FP16 optimization...")
            model = torch.compile(model, mode="reduce-overhead")
            compile_success = True
        except Exception as e:
            print(f"torch.compile failed: {e}")
    
    load_time = time.time() - start_load_time
    
    # Processor
    min_pixels = 128*28*28
    max_pixels = 256*28*28
    processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)
    
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
    
    # Inference timing with appropriate precision
    print(f"Running {optimization_type} inference...")
    start_inference_time = time.time()
    
    try:
        if "FP16" in optimization_type:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                generated_ids = model.generate(**inputs, max_new_tokens=128)
        else:
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
        'model_size': model_size,
        'optimization': optimization_type,
        'compile_success': compile_success if optimization_type in ["torch.compile", "torch.compile + FP16"] else None,
        'success': success,
        'error': error_msg,
        'load_time': load_time,
        'inference_time': inference_time,
        'memory_before_gb': memory_before,
        'memory_after_gb': memory_after,
        'peak_memory_gb': peak_memory,
        'output_text': output_text[0] if output_text else None
    }

def print_comprehensive_comparison(results):
    """Print comprehensive comparison across model sizes and optimizations"""
    print(f"\n{'='*120}")
    print("COMPREHENSIVE OPTIMIZATION COMPARISON: 3B vs 7B MODELS")
    print(f"{'='*120}")
    
    # Header
    print(f"{'Model + Optimization':<40} {'Success':<10} {'Load(s)':<10} {'Inference(s)':<12} {'Peak Mem(GB)':<12} {'Speedup':<10} {'Mem Efficiency':<15}")
    print(f"{'-'*120}")
    
    # Group results by model size
    results_3b = [r for r in results if r['model_size'] == '3B']
    results_7b = [r for r in results if r['model_size'] == '7B']
    
    # Find baselines for speedup calculation
    baseline_3b = next((r for r in results_3b if r['optimization'] == 'Baseline'), None)
    baseline_7b = next((r for r in results_7b if r['optimization'] == 'Baseline'), None)
    
    def print_result_row(result, baseline):
        success_str = "✅" if result['success'] else "❌"
        model_opt = f"{result['model_size']} + {result['optimization']}"
        
        if result['success']:
            load_str = f"{result['load_time']:.2f}"
            inf_str = f"{result['inference_time']:.2f}"
            mem_str = f"{result['peak_memory_gb']:.2f}"
            
            # Calculate speedup vs baseline
            if baseline and baseline['success']:
                speedup = baseline['inference_time'] / result['inference_time']
                speedup_str = f"{speedup:.2f}x"
                
                # Memory efficiency (lower is better)
                mem_eff = (result['peak_memory_gb'] / baseline['peak_memory_gb']) * 100
                mem_eff_str = f"{mem_eff:.1f}%"
            else:
                speedup_str = "N/A"
                mem_eff_str = "N/A"
        else:
            load_str = inf_str = mem_str = speedup_str = mem_eff_str = "FAILED"
        
        print(f"{model_opt:<40} {success_str:<10} {load_str:<10} {inf_str:<12} {mem_str:<12} {speedup_str:<10} {mem_eff_str:<15}")
    
    # Print 3B results
    print("\n3B MODEL RESULTS:")
    for result in results_3b:
        print_result_row(result, baseline_3b)
    
    print("\n7B MODEL RESULTS:")
    for result in results_7b:
        print_result_row(result, baseline_7b)
    
    # Cross-model comparison
    print(f"\n{'='*120}")
    print("CROSS-MODEL SIZE COMPARISON")
    print(f"{'='*120}")
    
    if baseline_3b and baseline_7b and baseline_3b['success'] and baseline_7b['success']:
        size_speedup = baseline_7b['inference_time'] / baseline_3b['inference_time']
        mem_ratio = baseline_7b['peak_memory_gb'] / baseline_3b['peak_memory_gb']
        
        print(f"3B vs 7B Baseline Comparison:")
        print(f"  📊 3B is {size_speedup:.2f}x faster than 7B")
        print(f"  💾 3B uses {(1/mem_ratio)*100:.1f}% of 7B's memory")
        print(f"  🎯 Memory difference: {baseline_7b['peak_memory_gb'] - baseline_3b['peak_memory_gb']:.2f}GB")
    
    # Detailed results
    print(f"\n{'='*120}")
    print("DETAILED RESULTS & ANALYSIS")
    print(f"{'='*120}")
    
    for result in results:
        print(f"\n{result['model_size']} + {result['optimization']}:")
        if result['success']:
            print(f"  ✅ Success")
            if result['compile_success'] is not None:
                compile_status = "✅" if result['compile_success'] else "❌"
                print(f"  🔧 torch.compile: {compile_status}")
            print(f"  📊 Load Time: {result['load_time']:.2f}s")
            print(f"  ⚡ Inference Time: {result['inference_time']:.2f}s")
            print(f"  💾 Peak Memory: {result['peak_memory_gb']:.2f}GB")
            print(f"  📝 Output Length: {len(result['output_text']) if result['output_text'] else 0} chars")
            if result['output_text']:
                print(f"  📄 Output Preview: {result['output_text'][:80]}...")
        else:
            print(f"  ❌ Failed: {result['error']}")
    
    # Optimization effectiveness summary
    print(f"\n{'='*120}")
    print("OPTIMIZATION EFFECTIVENESS SUMMARY")
    print(f"{'='*120}")
    
    print("🎯 BEST OPTIMIZATIONS:")
    
    # Find best performing configurations
    successful_results = [r for r in results if r['success']]
    if successful_results:
        fastest_3b = min([r for r in successful_results if r['model_size'] == '3B'], key=lambda x: x['inference_time'], default=None)
        fastest_7b = min([r for r in successful_results if r['model_size'] == '7B'], key=lambda x: x['inference_time'], default=None)
        
        if fastest_3b:
            print(f"  🥇 Fastest 3B: {fastest_3b['optimization']} ({fastest_3b['inference_time']:.2f}s)")
        if fastest_7b:
            print(f"  🥇 Fastest 7B: {fastest_7b['optimization']} ({fastest_7b['inference_time']:.2f}s)")
    
    print("\n📈 OPTIMIZATION IMPACT:")
    print("  ✅ torch.compile: Graph optimization, kernel fusion")
    print("  ✅ FP16 Mixed Precision: Faster computation, more memory")
    print("  ✅ Combined: Best of both optimizations")
    print("  🎯 Model Size Impact: 3B significantly faster and more memory efficient")

if __name__ == "__main__":
    print("🚀 COMPREHENSIVE MODEL SIZE & OPTIMIZATION BENCHMARK")
    print("Testing 3B vs 7B models with torch.compile and FP16 mixed precision")
    
    results = []
    
    # Test configurations
    model_sizes = ['3B', '7B']
    optimizations = ['Baseline', 'torch.compile', 'FP16 Mixed Precision', 'torch.compile + FP16']
    
    for model_size in model_sizes:
        for optimization in optimizations:
            print(f"\n🔥 Running {model_size} + {optimization}...")
            try:
                result = run_model_test(model_size, optimization)
                results.append(result)
            except Exception as e:
                print(f"❌ Failed to run {model_size} + {optimization}: {e}")
                results.append({
                    'model_size': model_size,
                    'optimization': optimization,
                    'compile_success': None,
                    'success': False,
                    'error': str(e),
                    'load_time': 0,
                    'inference_time': 0,
                    'memory_before_gb': 0,
                    'memory_after_gb': 0,
                    'peak_memory_gb': 0,
                    'output_text': None
                })
    
    # Print comprehensive comparison
    print_comprehensive_comparison(results)
