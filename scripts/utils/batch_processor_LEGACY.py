"""
Dynamic Batch Processing Module for VLM Inference

This module handles intelligent batching of images based on:
- Real-time GPU memory detection
- Dynamic model memory usage calculation  
- Configurable safety margins
- Automatic batch size optimization
"""

import torch
import gc
import os
import psutil
import time
import threading
# import concurrent.futures  # Removed - using sequential processing to avoid GPU OOM
from typing import Dict, List, Tuple, Optional, Callable, Any
import json
from tqdm import tqdm
from text_cleaner import clean_vlm_response
from datetime import datetime, timedelta
# Import memory clearing utility
try:
    from local_model.model_utils import clear_model_memory_if_needed
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'local_model'))
    from model_utils import clear_model_memory_if_needed
from dynamic_error_handler import error_handler


def get_prompt_suffix(engine):
    """Get model-specific prompt suffix based on model capabilities"""
    # REVERT: Enhanced prompting made LLAVA worse - going back to standard prompts
    # Memory clearing will handle repetition, prompts should remain consistent
    return " Answer format (do not generate any other content): The answer is <answer>."


class MemoryMonitor:
    """Monitor and manage GPU/system memory"""
    
    def __init__(self, safety_margin_percent: float = 20.0):
        """
        Initialize memory monitor
        
        Args:
            safety_margin_percent: Percentage of memory to reserve as safety buffer
        """
        self.safety_margin_percent = safety_margin_percent
        self.baseline_memory = None
        
    def get_total_gpu_memory_gb(self) -> float:
        """Get total GPU memory in GB"""
        if not torch.cuda.is_available():
            return 0.0
        
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    def get_available_gpu_memory_gb(self) -> float:
        """Get currently available GPU memory in GB"""
        if not torch.cuda.is_available():
            return 0.0
        
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated()
        cached_memory = torch.cuda.memory_reserved()
        
        # Available = Total - max(Allocated, Cached)
        used_memory = max(allocated_memory, cached_memory)
        available = (total_memory - used_memory) / (1024**3)
        
        return max(0.0, available)
    
    def get_current_memory_usage_gb(self) -> Dict[str, float]:
        """Get detailed current memory usage"""
        if not torch.cuda.is_available():
            return {"total": 0.0, "allocated": 0.0, "cached": 0.0, "available": 0.0}
        
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated() / (1024**3)
        cached = torch.cuda.memory_reserved() / (1024**3)
        available = self.get_available_gpu_memory_gb()
        
        return {
            "total": total,
            "allocated": allocated, 
            "cached": cached,
            "available": available
        }
    
    def set_baseline_memory(self):
        """Set baseline memory usage (call after model loading)"""
        if torch.cuda.is_available():
            self.baseline_memory = torch.cuda.memory_allocated() / (1024**3)
            print(f"📊 Baseline memory set: {self.baseline_memory:.2f}GB")
    
    def get_model_memory_usage_gb(self) -> float:
        """Get actual model memory usage based on baseline"""
        if not torch.cuda.is_available() or self.baseline_memory is None:
            return 0.0
        
        current = torch.cuda.memory_allocated() / (1024**3)
        model_memory = current - (self.baseline_memory or 0)
        return max(0.0, model_memory)
    
    def cleanup_memory(self):
        """Clean up GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


class DynamicBatchCalculator:
    """Calculate optimal batch sizes based on real-time memory conditions"""
    
    def __init__(self, memory_monitor: MemoryMonitor):
        self.memory_monitor = memory_monitor
        self.image_memory_mb = 25  # Average image memory usage
        
    def estimate_image_memory_mb(self, image_count: int = 1) -> float:
        """Estimate memory usage for a given number of images"""
        # Base estimate, could be enhanced with actual image size analysis
        return image_count * self.image_memory_mb
    
    def calculate_optimal_parallel_batch_size(self, engine: str) -> int:
        """
        Calculate optimal parallel batch size for simultaneous question processing
        
        Args:
            engine: Model engine name
            
        Returns:
            Optimal parallel batch size for GPU utilization
        """
        # Get current memory status
        memory_info = self.memory_monitor.get_current_memory_usage_gb()
        available_memory_gb = memory_info['available']
        
        # Estimate memory per question (more conservative for parallel processing)
        if engine == 'gpt4o':
            return 10  # Cloud model
        
        # Estimate based on available memory for parallel processing
        # Further optimized memory estimates for 6-7GB target utilization
        if "Gemma" in engine:
            memory_per_question_gb = 1.2  # Gemma models need more memory due to vision tower
        elif "3B" in engine:
            memory_per_question_gb = 0.4  # More aggressive for 3B models (was 0.7)
        elif "7B" in engine:
            memory_per_question_gb = 0.8  # Reduced for 7B models (was 1.2)
        else:
            memory_per_question_gb = 0.6  # More aggressive default (was 0.9)
        
        # Calculate how many questions can fit in available memory
        # Ultra-aggressive safety margin for maximum utilization (targeting 6-7GB usage)
        aggressive_safety_margin = min(self.memory_monitor.safety_margin_percent, 10.0)  # Max 10% safety
        usable_memory_gb = available_memory_gb * (1 - aggressive_safety_margin / 100)
        max_parallel_questions = int(usable_memory_gb / memory_per_question_gb)
        
        # Focus on image batching instead of question parallelism for 8GB GPU optimization
        # Set all models to sequential processing to prevent OOM errors
        parallel_batch_size = 1  # Sequential processing for all VLMs to avoid memory issues
        
        # Simple validation - ensure we don't exceed available memory
        if available_memory_gb < 2.0:
            parallel_batch_size = 1  # Sequential processing for low memory
        
        # Calculate expected memory usage for sequential processing
        expected_memory_usage_gb = parallel_batch_size * memory_per_question_gb + 3.0  # +3.0GB for model base (conservative)
        expected_memory_percent = (expected_memory_usage_gb / 8.2) * 100  # Total GPU memory
        
        print(f"🚀 SEQUENTIAL PROCESSING - Memory calculation for {engine}:")
        print(f"   Available memory: {available_memory_gb:.1f}GB")
        print(f"   Usable memory: {usable_memory_gb:.1f}GB (safety margin: {aggressive_safety_margin:.1f}%)")
        print(f"   Memory per question: {memory_per_question_gb:.1f}GB")
        print(f"   Max parallel questions by memory: {max_parallel_questions}")
        print(f"   🎯 SEQUENTIAL batch size: {parallel_batch_size} question at a time")
        print(f"   🔥 Expected memory usage: {expected_memory_usage_gb:.1f}GB ({expected_memory_percent:.0f}%)")
        print(f"   🚀 Target: Safe memory utilization to prevent OOM")
        
        return parallel_batch_size

    def calculate_optimal_batch_size(self, engine: str, max_batch_size: int = 50) -> int:
        """
        Calculate optimal batch size based on available memory
        
        Args:
            engine: Model engine name
            max_batch_size: Maximum allowed batch size
            
        Returns:
            Optimal batch size (minimum 1)
        """
        # Handle cloud models
        if engine == 'gpt4o':
            return min(max_batch_size, 50)
        
        # Get current memory status
        memory_info = self.memory_monitor.get_current_memory_usage_gb()
        available_memory_gb = memory_info["available"]
        
        # Apply safety margin
        safety_factor = (100 - self.memory_monitor.safety_margin_percent) / 100
        usable_memory_gb = available_memory_gb * safety_factor
        
        # Calculate how many images can fit
        usable_memory_mb = usable_memory_gb * 1024
        max_images_by_memory = int(usable_memory_mb / self.image_memory_mb)
        
        # Maximize image batching for single model runs (8GB GPU optimization)
        if usable_memory_gb >= 5.0:
            suggested_batch = min(50, max_images_by_memory)  # Aggressive batching for 8GB GPU
        elif usable_memory_gb >= 3.0:
            suggested_batch = min(30, max_images_by_memory)  # High batching
        elif usable_memory_gb >= 2.0:
            suggested_batch = min(20, max_images_by_memory)  # Medium batching
        elif usable_memory_gb >= 1.0:
            suggested_batch = min(12, max_images_by_memory)  # Conservative batching
        else:
            suggested_batch = min(6, max_images_by_memory)   # Minimal batching
        
        # Ensure minimum batch size and respect maximum
        batch_size = max(1, min(suggested_batch, max_batch_size))
        
        print(f"🧮 Dynamic batch calculation for {engine}:")
        print(f"   Available memory: {available_memory_gb:.1f}GB")
        print(f"   Usable (with {self.memory_monitor.safety_margin_percent}% margin): {usable_memory_gb:.1f}GB")
        print(f"   Optimal batch size: {batch_size} images")
        
        return batch_size
    
    def validate_batch_size(self, batch_size: int, engine: str) -> bool:
        """
        Validate if a batch size is safe to use
        
        Args:
            batch_size: Proposed batch size
            engine: Model engine name
            
        Returns:
            True if batch size is safe, False otherwise
        """
        if engine == 'gpt4o':
            return True  # Cloud model, no memory constraints
        
        estimated_memory_gb = self.estimate_image_memory_mb(batch_size) / 1024
        available_memory_gb = self.memory_monitor.get_available_gpu_memory_gb()
        
        # Check if batch fits with safety margin
        safety_factor = (100 - self.memory_monitor.safety_margin_percent) / 100
        return estimated_memory_gb <= (available_memory_gb * safety_factor)


class TimeEstimator:
    """Track timing and estimate remaining time for batch processing"""
    
    def __init__(self):
        self.start_time = None
        self.task_start_times = {}
        self.question_times = []
        self.attack_times = {}
        self.current_attack = None
        
    def start_timing(self, task_name: str = "overall"):
        """Start timing for a task"""
        current_time = time.time()
        if task_name == "overall":
            self.start_time = current_time
        self.task_start_times[task_name] = current_time
        
    def record_question_time(self, processing_time: float):
        """Record time taken for a single question"""
        self.question_times.append(processing_time)
        # Keep only last 50 times for rolling average
        if len(self.question_times) > 50:
            self.question_times = self.question_times[-50:]
    
    def get_avg_question_time(self) -> float:
        """Get average time per question"""
        if not self.question_times:
            return 0.0
        return sum(self.question_times) / len(self.question_times)
    
    def estimate_remaining_time(self, completed: int, total: int) -> str:
        """Estimate remaining time based on progress"""
        if completed == 0 or not self.question_times:
            return "Calculating..."
        
        avg_time = self.get_avg_question_time()
        remaining_questions = total - completed
        remaining_seconds = remaining_questions * avg_time
        
        if remaining_seconds < 60:
            return f"{remaining_seconds:.0f}s"
        elif remaining_seconds < 3600:
            return f"{remaining_seconds/60:.1f}m"
        else:
            hours = remaining_seconds // 3600
            minutes = (remaining_seconds % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"
    
    def get_elapsed_time(self, task_name: str = "overall") -> str:
        """Get elapsed time for a task"""
        if task_name not in self.task_start_times:
            return "0s"
        
        elapsed_seconds = time.time() - self.task_start_times[task_name]
        
        if elapsed_seconds < 60:
            return f"{elapsed_seconds:.0f}s"
        elif elapsed_seconds < 3600:
            return f"{elapsed_seconds/60:.1f}m"
        else:
            hours = elapsed_seconds // 3600
            minutes = (elapsed_seconds % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"
    
    def start_attack(self, attack_name: str):
        """Start timing for a specific attack"""
        self.current_attack = attack_name
        self.start_timing(f"attack_{attack_name}")
    
    def finish_attack(self, attack_name: str, total_questions: int):
        """Finish timing for an attack and record statistics"""
        if f"attack_{attack_name}" in self.task_start_times:
            elapsed = time.time() - self.task_start_times[f"attack_{attack_name}"]
            self.attack_times[attack_name] = {
                'elapsed': elapsed,
                'questions': total_questions,
                'questions_per_second': total_questions / elapsed if elapsed > 0 else 0
            }
    
    def get_attack_summary(self) -> str:
        """Get summary of attack processing times"""
        if not self.attack_times:
            return "No attacks completed yet"
        
        lines = ["📊 Attack Processing Summary:"]
        for attack, stats in self.attack_times.items():
            elapsed_str = f"{stats['elapsed']/60:.1f}m" if stats['elapsed'] >= 60 else f"{stats['elapsed']:.0f}s"
            lines.append(f"   {attack}: {stats['questions']} questions in {elapsed_str} ({stats['questions_per_second']:.1f} q/s)")
        
        return "\n".join(lines)


class VLMBatchOrchestrator:
    """Orchestrate VLM inference with intelligent batching and cross-attack optimization"""
    
    def __init__(self, safety_margin_percent: float = 20.0):
        self.memory_monitor = MemoryMonitor(safety_margin_percent)
        self.batch_calculator = DynamicBatchCalculator(self.memory_monitor)
        self.time_estimator = TimeEstimator()
        self._thread_local = threading.local()  # For thread-safe operations
    
    def _process_single_question(self, 
                                question_data: Dict,
                                image_url: str,
                                vlm_client_func: Callable,
                                engine: str,
                                metadata: Dict,
                                batch_info: Dict = None) -> Dict:
        """
        Process a single question (thread-safe)
        
        Args:
            question_data: Dictionary containing question information
            image_url: Data URL of the image
            vlm_client_func: VLM client function
            engine: Model engine name
            metadata: Metadata for the result
            
        Returns:
            Processed result dictionary
        """
        data = question_data
        
        # Check if this image should be skipped due to previous errors
        img_path = question_data.get('image_path', '')
        if error_handler.should_skip_image(img_path):
            error_handler.log_skip(img_path, engine, data['question_id'])
            
            # Create error result for skipped image
            markers = data.get('markers', [])
            return {
                "question_id": data['question_id'],
                "prompt": data['text'],
                "text": "ERROR: Image skipped - known to cause CUDA device-side assert errors",
                "truth": data['answer'],
                "type": data['type'],
                "answer_id": "",
                "markers": markers,
                "model_id": engine,
                "metadata": metadata
            }
        
        # Prepare message format
        msgs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": data['text'] + get_prompt_suffix(engine)
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ]
        
        # Clear memory for models prone to context bleeding/repetition
        memory_cleared = clear_model_memory_if_needed(engine)
        if memory_cleared:
            print(f"🧹 Memory cleared for {engine} to prevent context bleeding")
        
        # Call VLM client with error handling, timing, and performance metrics
        question_start_time = time.time()
        performance_metrics = None
        try:
            # Call VLM client with metrics collection enabled
            vlm_result = vlm_client_func(message_text=msgs, engine=engine, sample_n=1, collect_metrics=True)
            
            # Handle different return formats (with or without metrics)
            if len(vlm_result) == 3:
                res, _, performance_metrics = vlm_result
            else:
                res, _ = vlm_result
            
            # Update batch information in performance metrics
            if performance_metrics and batch_info:
                performance_metrics["batch_info"].update(batch_info)
            
            # Handle CUDA errors gracefully
            if res == "CUDA_ERROR_HANDLED":
                print(f"⚠️  CUDA error handled for question {data['question_id']}")
                res = error_handler.handle_inference_error(img_path, data['question_id'], Exception("CUDA device-side assert"))
        except Exception as e:
            print(f"⚠️  VLM client error for question {data['question_id']}: {e}")
            res = error_handler.handle_inference_error(img_path, data['question_id'], e)
        finally:
            # Record processing time
            processing_time = time.time() - question_start_time
            # Note: We'll record this in the batch orchestrator level to avoid thread issues
            
            # Aggressive cleanup after each question for DeepSeek1_VL_7B
            if engine == "DeepSeek1_VL_7B":
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        # Prepare result
        markers = data.get('markers', [])
        result = {
            "question_id": data['question_id'],
            "prompt": data['text'],
            "text": clean_vlm_response(res),
            "truth": data['answer'],
            "type": data['type'],
            "answer_id": "",
            "markers": markers,
            "model_id": engine,
            "metadata": metadata
        }
        
        # Add performance metrics if available
        if performance_metrics:
            result["performance_metrics"] = performance_metrics
            
        return result
    
    def _process_parallel_sub_batch(self,
                                   questions_data: List[Dict],
                                   image_url: str,
                                   vlm_client_func: Callable,
                                   engine: str,
                                   metadata: Dict,
                                   max_workers: int = None) -> List[Dict]:
        """
        Process multiple questions SEQUENTIALLY for the same image to avoid GPU OOM
        
        Args:
            questions_data: List of question dictionaries
            image_url: Data URL of the image
            vlm_client_func: VLM client function
            engine: Model engine name
            metadata: Default metadata (used as fallback)
            max_workers: Ignored - kept for compatibility (sequential processing)
            
        Returns:
            List of processed results
        """
        if not questions_data:
            return []
        
        # Use SEQUENTIAL processing to avoid GPU memory contention
        results = []
        
        # Process questions one by one to prevent GPU OOM
        for i, question_data in enumerate(questions_data):
            try:
                # Create batch info for performance metrics
                batch_info = {
                    "batch_size": len(questions_data),
                    "position_in_batch": i + 1,
                    "total_batch_questions": len(questions_data)
                }
                
                # Use individual question metadata if available, otherwise use default
                question_metadata = question_data.get('metadata', metadata)
                
                result = self._process_single_question(
                    question_data,
                    image_url,
                    vlm_client_func,
                    engine,
                    question_metadata,
                    batch_info
                )
                results.append(result)
                
                # Brief pause between questions to allow GPU memory cleanup
                import time
                time.sleep(0.1)  # 100ms pause for memory cleanup
                
            except Exception as e:
                print(f"⚠️  Sequential processing error for question {question_data['question_id']}: {e}")
                # Create error result
                markers = question_data.get('markers', [])
                error_result = {
                    "question_id": question_data['question_id'],
                    "prompt": question_data['text'],
                    "text": f"ERROR: Sequential processing exception - {str(e)}",
                    "truth": question_data['answer'],
                    "type": question_data['type'],
                    "answer_id": "",
                    "markers": markers,
                    "model_id": engine,
                    "metadata": question_data.get('metadata', metadata)
                }
                results.append(error_result)
        
        return results
    
    def process_cross_attack_batches(self,
                                   attack_configs: List[Tuple[str, str, str]],
                                   task: str,
                                   engine: str,
                                   vlm_client_func: Callable,
                                   local_image_to_data_url_func: Callable,
                                   eval_data: List[Dict],
                                   processed_images: List[str]) -> Dict[str, List[Dict]]:
        """
        Process multiple attacks efficiently using attack-specific batching
        
        This approach processes attacks sequentially but uses optimal batch sizes
        to maximize GPU utilization while avoiding CUDA tensor compatibility issues.
        
        Args:
            attack_configs: List of (output_file, img_dir, attack_name) tuples
            task: Task name
            engine: Model engine name
            vlm_client_func: VLM client function
            local_image_to_data_url_func: Function to convert image paths to data URLs
            eval_data: Evaluation data for the task
            processed_images: List of processed image paths
            
        Returns:
            Dictionary mapping attack names to their results
        """
        # Calculate optimal batch size based on available memory
        optimal_batch_size = self.batch_calculator.calculate_optimal_batch_size(engine)
        
        attack_results = {}
        
        # Initialize results for each attack
        for _, _, attack_name in attack_configs:
            attack_results[attack_name] = []
        
        print(f"\n🚀 Cross-Attack Efficient Processing:")
        print(f"   Optimal batch size: {optimal_batch_size} images")
        print(f"   Processing {len(attack_configs)} attacks × {len(processed_images)} images = {len(attack_configs) * len(processed_images)} total images")
        
        # Clean GPU memory before starting to prevent CUDA assertion errors
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("🧹 Cleaned GPU memory cache before cross-attack processing")
        except Exception as e:
            print(f"⚠️  Warning: Could not clean GPU memory: {e}")
        
        # Process each attack with optimal batching
        for attack_idx, (output_file, img_dir, attack_name) in enumerate(attack_configs):
            print(f"\n🎯 Processing attack {attack_idx + 1}/{len(attack_configs)}: {attack_name}")
            
            # Load SSIM mapping from attack_parameters.json
            from pathlib import Path
            import json
            import os
            attack_params_file = Path("results/attack_parameters.json")
            ssim_mapping = {}
            if attack_params_file.exists():
                try:
                    with open(attack_params_file, 'r') as f:
                        data = json.load(f)
                    
                    if 'attacks' in data:
                        for attack_type, attack_data in data['attacks'].items():
                            if 'executions' in attack_data:
                                for execution in attack_data['executions']:
                                    adversarial_path = execution.get('adversarial_image_path', '')
                                    ssim_value = execution.get('parameters', {}).get('ssim', 0.0)
                                    ssim_target = execution.get('parameters', {}).get('ssim_target', 0.0)
                                    
                                    # Use adversarial path as key, store both ssim and ssim_target
                                    if adversarial_path:
                                        ssim_mapping[adversarial_path] = {
                                            'ssim': ssim_value,
                                            'ssim_target': ssim_target
                                        }
                except Exception as e:
                    print(f"⚠️ Warning: Could not load SSIM data: {e}")
            
            # Group data by image for this attack
            image_to_data = {}
            for data in eval_data:
                img_path = data.get('image', '')
                if img_path in processed_images:
                    if img_path not in image_to_data:
                        image_to_data[img_path] = []
                    
                    # Get SSIM value for this image and attack
                    ssim_value = 1.0  # Default for clean images
                    ssim_target = 1.0  # Default for clean images
                    metadata_image_path = img_path
                    
                    if attack_name != "Original (No Attack)":
                        # Construct expected adversarial image path
                        img_filename = os.path.basename(img_path)
                        expected_adversarial_path = f"{img_dir.rstrip('/')}/chart/{img_filename}"

                        # Construct database composite key (original_path, adversarial_path)
                        original_path = f"data/clean/chart/{img_filename}"
                        composite_key = (original_path, expected_adversarial_path)

                        # Look up SSIM using database format
                        if composite_key in ssim_mapping:
                            ssim_value = ssim_mapping[composite_key]['ssim']
                            ssim_target = ssim_mapping[composite_key]['ssim_target']
                            metadata_image_path = expected_adversarial_path
                        else:
                            print(f"⚠️ Warning: No SSIM data found for key: {composite_key}")
                    
                    # Add attack metadata with SSIM
                    data_with_attack = data.copy()
                    data_with_attack['metadata'] = {
                        **data.get('metadata', {}),
                        "adversarial": attack_name != "Original (No Attack)",
                        "task": task,
                        "attack_type": attack_name.lower().replace(" ", "_").replace("(", "").replace(")", ""),
                        "ssim": ssim_value,
                        "ssim_target": ssim_target,
                        "image_path": metadata_image_path,
                        "original_image_path": img_path,
                        "timestamp": __import__('datetime').datetime.now().isoformat() + "Z",
                        "attack_name": attack_name,
                        "attack_output_file": output_file
                    }
                    image_to_data[img_path].append(data_with_attack)
            
            # Create batched data for this attack
            attack_image_data = [(img_path, data_items) for img_path, data_items in image_to_data.items()]
            
            # Process this attack using standard batch processing
            attack_results_list = self.process_vlm_requests_in_batches(
                messages_data=attack_image_data,
                vlm_client_func=vlm_client_func,
                engine=engine,
                local_image_to_data_url_func=local_image_to_data_url_func,
                img_dir=img_dir
            )
            
            # Store results for this attack
            attack_results[attack_name] = attack_results_list
            
            # 💾 SAVE RESULTS IMMEDIATELY after processing each attack
            import json
            import os
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Save results to file immediately
            with open(output_file, 'w') as fout:
                for result in attack_results_list:
                    fout.write(json.dumps(result) + '\n')
            
            print(f"   ✅ {attack_name}: {len(attack_results_list)} results processed")
            print(f"   💾 Results saved to {output_file}")
            
            # Memory cleanup between attacks
            self.memory_monitor.cleanup_memory()
        
        # Print simple completion message
        total_results = sum(len(results) for results in attack_results.values())
        print(f"\n✅ All attacks completed! Total: {total_results} results")
        
        return attack_results

    def process_vlm_requests_in_batches(self,
                                      messages_data: List[Tuple[Any, Dict]],
                                      vlm_client_func: Callable,
                                      engine: str,
                                      local_image_to_data_url_func: Callable,
                                      img_dir: str) -> List[Dict]:
        """
        Process VLM requests in optimized batches
        
        Args:
            messages_data: List of (image_path, data_items) tuples
            vlm_client_func: VLM client function (e.g., send_chat_request_azure)
            engine: Model engine name
            local_image_to_data_url_func: Function to convert image paths to data URLs
            img_dir: Base image directory
            
        Returns:
            List of processed results
        """
        if not messages_data:
            return []
        
        # Calculate optimal batch size
        batch_size = self.batch_calculator.calculate_optimal_batch_size(engine)
        
        # Split into batches
        batches = []
        for i in range(0, len(messages_data), batch_size):
            batch = messages_data[i:i + batch_size]
            batches.append(batch)
        
        print(f"📦 Created {len(batches)} batches from {len(messages_data)} images")
        print(f"   Batch sizes: {[len(batch) for batch in batches]}")
        
        all_results = []
        
        # Process each batch
        for batch_idx, batch in enumerate(batches):
            batch_results = self._process_single_batch(
                batch, vlm_client_func, engine, local_image_to_data_url_func, 
                img_dir, batch_idx, len(batches)
            )
            all_results.extend(batch_results)
        
        return all_results
    
    def _process_single_batch(self,
                            batch: List[Tuple[str, List]], 
                            vlm_client_func: Callable,
                            engine: str,
                            local_image_to_data_url_func: Callable,
                            img_dir: str,
                            batch_index: int,
                            total_batches: int) -> List[Dict]:
        """
        Process a single batch with memory monitoring and image pre-loading
        
        Args:
            batch: Batch of (image_path, data_items) to process
            vlm_client_func: VLM client function
            engine: Model engine name
            local_image_to_data_url_func: Function to convert images
            img_dir: Base image directory
            batch_index: Current batch index
            total_batches: Total number of batches
            
        Returns:
            List of processing results for this batch
        """
        print(f"\n🔄 Processing batch {batch_index + 1}/{total_batches} ({len(batch)} images)")
        
        # Monitor memory before batch
        memory_before = self.memory_monitor.get_current_memory_usage_gb()
        print(f"   💾 Memory before: {memory_before['allocated']:.1f}GB allocated, "
              f"{memory_before['available']:.1f}GB available")
        
        batch_results = []
        batch_urls = {}
        
        try:
            # Phase 1: Pre-load all images in the batch into memory
            print(f"   🖼️  Phase 1: Loading {len(batch)} images into memory...")
            for img_path, _ in batch:
                full_img_path = img_dir + img_path
                
                if not os.path.exists(full_img_path):
                    print(f"   ⚠️  Warning: Image not found: {full_img_path}")
                    continue
                
                batch_urls[img_path] = local_image_to_data_url_func(full_img_path)
            
            print(f"   📸 Successfully loaded {len(batch_urls)} images into memory")
            
            # Phase 2: Process all questions for all images in this batch with sequential processing
            total_questions = sum(len(data_items) for _, data_items in batch)
            print(f"   🚀 Phase 2: Processing {total_questions} questions with sequential optimization...")
            
            # Calculate optimal sequential batch size for sub-batching
            optimal_sequential_size = self.batch_calculator.calculate_optimal_parallel_batch_size(engine)
            
            # Simple batch progress without detailed timing
            with tqdm(total=total_questions, desc=f"   Batch {batch_index + 1}/{total_batches}", leave=False) as pbar:
                for img_path, data_items in batch:
                    if img_path not in batch_urls:
                        pbar.update(len(data_items))
                        continue
                    
                    url = batch_urls[img_path]
                    
                    # Filter out previously failed images
                    valid_data_items = []
                    for data in data_items:
                        if error_handler.should_skip_image(img_path):
                            error_handler.log_skip(img_path, engine, data['question_id'])
                            
                            # Create error result for skipped image
                            markers = data.get('markers', [])
                            result = {
                                "question_id": data['question_id'],
                                "prompt": data['text'],
                                "text": "ERROR: Image skipped - known to cause CUDA device-side assert errors",
                                "truth": data['answer'],
                                "type": data['type'],
                                "answer_id": "",
                                "markers": markers,
                                "model_id": engine,
                                "metadata": data.get('metadata', {})
                            }
                            batch_results.append(result)
                            pbar.update(1)
                        else:
                            valid_data_items.append(data)
                    
                    # Process valid questions in parallel sub-batches
                    if valid_data_items:
                        # Split questions into sequential sub-batches
                        for i in range(0, len(valid_data_items), optimal_sequential_size):
                            sub_batch = valid_data_items[i:i + optimal_sequential_size]
                            
                            print(f"      🔄 Processing {len(sub_batch)} questions sequentially for {img_path}")
                            
                            # Time the sub-batch processing
                            sub_batch_start = time.time()
                            
                            # Process this sub-batch sequentially to avoid GPU OOM
                            sub_batch_results = self._process_parallel_sub_batch(
                                questions_data=sub_batch,
                                image_url=url,
                                vlm_client_func=vlm_client_func,
                                engine=engine,
                                metadata={},
                                max_workers=1  # Sequential processing
                            )
                            
                            batch_results.extend(sub_batch_results)
                            pbar.update(len(sub_batch))
                            
                            # Brief pause between sequential sub-batches for GPU memory cleanup
                            if i + optimal_sequential_size < len(valid_data_items):
                                time.sleep(0.2)
            
            # Monitor memory after batch processing
            memory_after = self.memory_monitor.get_current_memory_usage_gb()
            print(f"   💾 Memory after processing: {memory_after['allocated']:.1f}GB allocated, "
                  f"{memory_after['available']:.1f}GB available")
            
        except Exception as e:
            print(f"   ❌ Error processing batch {batch_index + 1}: {e}")
            raise
        
        finally:
            # Phase 3: Cleanup batch URLs from memory
            batch_urls.clear()
            self.memory_monitor.cleanup_memory()
            
            memory_final = self.memory_monitor.get_current_memory_usage_gb()
            print(f"   🧹 Memory after cleanup: {memory_final['allocated']:.1f}GB allocated, "
                  f"{memory_final['available']:.1f}GB available")
        
        print(f"   ✅ Batch {batch_index + 1} completed: {len(batch_results)} results")
        return batch_results


class ImageBatchProcessor:
    """Process images in intelligent batches (Legacy class for backward compatibility)"""
    
    def __init__(self, safety_margin_percent: float = 20.0):
        self.memory_monitor = MemoryMonitor(safety_margin_percent)
        self.batch_calculator = DynamicBatchCalculator(self.memory_monitor)
        self.vlm_orchestrator = VLMBatchOrchestrator(safety_margin_percent)
        self.time_estimator = self.vlm_orchestrator.time_estimator  # Share time estimator
        
    def prepare_batches(self, 
                       image_data: List[Tuple[str, List]], 
                       engine: str,
                       max_batch_size: Optional[int] = None) -> List[List[Tuple[str, List]]]:
        """
        Prepare image batches based on optimal batch size
        
        Args:
            image_data: List of (image_path, data_items) tuples
            engine: Model engine name
            max_batch_size: Optional maximum batch size override
            
        Returns:
            List of batches, where each batch is a list of (image_path, data_items)
        """
        if not image_data:
            return []
        
        # Calculate optimal batch size
        if max_batch_size is None:
            batch_size = self.batch_calculator.calculate_optimal_batch_size(engine)
        else:
            batch_size = min(
                max_batch_size,
                self.batch_calculator.calculate_optimal_batch_size(engine, max_batch_size)
            )
        
        # Split into batches
        batches = []
        for i in range(0, len(image_data), batch_size):
            batch = image_data[i:i + batch_size]
            batches.append(batch)
        
        print(f"📦 Created {len(batches)} batches from {len(image_data)} images")
        print(f"   Batch sizes: {[len(batch) for batch in batches]}")
        
        return batches
    
    def process_batch_with_monitoring(self, 
                                    batch: List[Tuple[str, List]], 
                                    process_func,
                                    batch_index: int,
                                    total_batches: int) -> List:
        """
        Process a batch with memory monitoring
        
        Args:
            batch: Batch of (image_path, data_items) to process
            process_func: Function to process each item in batch
            batch_index: Current batch index (0-based)
            total_batches: Total number of batches
            
        Returns:
            List of processing results
        """
        print(f"\n🔄 Processing batch {batch_index + 1}/{total_batches} ({len(batch)} images)")
        
        # Monitor memory before batch
        memory_before = self.memory_monitor.get_current_memory_usage_gb()
        print(f"   💾 Memory before: {memory_before['allocated']:.1f}GB allocated, "
              f"{memory_before['available']:.1f}GB available")
        
        try:
            # Process the batch
            results = process_func(batch)
            
            # Monitor memory after batch
            memory_after = self.memory_monitor.get_current_memory_usage_gb()
            print(f"   💾 Memory after: {memory_after['allocated']:.1f}GB allocated, "
                  f"{memory_after['available']:.1f}GB available")
            
            # Cleanup memory
            self.memory_monitor.cleanup_memory()
            
            memory_final = self.memory_monitor.get_current_memory_usage_gb()
            print(f"   🧹 Memory after cleanup: {memory_final['allocated']:.1f}GB allocated, "
                  f"{memory_final['available']:.1f}GB available")
            
            return results
            
        except Exception as e:
            print(f"   ❌ Error processing batch {batch_index + 1}: {e}")
            # Emergency cleanup
            self.memory_monitor.cleanup_memory()
            raise
    
    def get_memory_summary(self) -> Dict[str, float]:
        """Get current memory summary"""
        return self.memory_monitor.get_current_memory_usage_gb()
    
    def process_vlm_requests_in_batches(self, *args, **kwargs):
        """Delegate to VLM orchestrator"""
        return self.vlm_orchestrator.process_vlm_requests_in_batches(*args, **kwargs)
    
    def process_cross_attack_batches(self, *args, **kwargs):
        """Delegate cross-attack processing to VLM orchestrator"""
        return self.vlm_orchestrator.process_cross_attack_batches(*args, **kwargs)


def create_batch_processor(safety_margin_percent: float = 20.0) -> ImageBatchProcessor:
    """
    Factory function to create a batch processor
    
    Args:
        safety_margin_percent: Memory safety margin (default 20%)
        
    Returns:
        Configured ImageBatchProcessor instance
    """
    return ImageBatchProcessor(safety_margin_percent)


# Example usage and testing
if __name__ == "__main__":
    # Test the batch processor
    processor = create_batch_processor(safety_margin_percent=25.0)
    
    # Get memory info
    memory_info = processor.get_memory_summary()
    print("Memory Summary:")
    for key, value in memory_info.items():
        print(f"  {key}: {value:.2f}GB")
    
    # Test batch size calculation
    test_engines = ['gpt4o', 'DeepSeek1_VL_1pt3B', 'Qwen25_VL_7B']
    for engine in test_engines:
        batch_size = processor.batch_calculator.calculate_optimal_batch_size(engine)
        print(f"\nOptimal batch size for {engine}: {batch_size}")