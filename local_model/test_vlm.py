"""
Test script for VLM model variants - can be used to test any integrated VLM
"""

import os
import sys
import time
import threading
import psutil
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from local_model.model_classes import create_model

# Hardcoded image path and question
IMAGE_PATH = "data/test_extracted/chart/20231114102825506748.png"
QUESTION = "What is shown in this chart?"

# Global flag to control the monitoring thread
monitoring_active = True

def monitor_system_activity():
    """Monitor and display system activity metrics every second"""
    print("\n--- Starting System Activity Monitor ---")
    
    # Get the current process
    process = psutil.Process(os.getpid())
    
    # Initialize counters
    last_disk_read = psutil.disk_io_counters().read_bytes if hasattr(psutil, 'disk_io_counters') and psutil.disk_io_counters() else 0
    last_disk_write = psutil.disk_io_counters().write_bytes if hasattr(psutil, 'disk_io_counters') and psutil.disk_io_counters() else 0
    last_net_sent = psutil.net_io_counters().bytes_sent if hasattr(psutil, 'net_io_counters') and psutil.net_io_counters() else 0
    last_net_recv = psutil.net_io_counters().bytes_recv if hasattr(psutil, 'net_io_counters') and psutil.net_io_counters() else 0
    
    try:
        while monitoring_active:
            # CPU usage
            cpu_percent = process.cpu_percent(interval=0.1)
            
            # Memory usage
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            # Disk activity
            if hasattr(psutil, 'disk_io_counters') and psutil.disk_io_counters():
                current_disk_read = psutil.disk_io_counters().read_bytes
                current_disk_write = psutil.disk_io_counters().write_bytes
                disk_read_speed = (current_disk_read - last_disk_read) / 1024  # KB/s
                disk_write_speed = (current_disk_write - last_disk_write) / 1024  # KB/s
                last_disk_read = current_disk_read
                last_disk_write = current_disk_write
            else:
                disk_read_speed = 0
                disk_write_speed = 0
            
            # Network activity
            if hasattr(psutil, 'net_io_counters') and psutil.net_io_counters():
                current_net_sent = psutil.net_io_counters().bytes_sent
                current_net_recv = psutil.net_io_counters().bytes_recv
                net_send_speed = (current_net_sent - last_net_sent) / 1024  # KB/s
                net_recv_speed = (current_net_recv - last_net_recv) / 1024  # KB/s
                last_net_sent = current_net_sent
                last_net_recv = current_net_recv
            else:
                net_send_speed = 0
                net_recv_speed = 0
            
            # GPU info if available
            gpu_info = ""
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                    gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 * 1024)  # MB
                    gpu_info = f" | GPU: {gpu_memory:.1f}MB/{gpu_memory_reserved:.1f}MB"
            except:
                pass
            
            # Terminal width for progress bar
            term_width = shutil.get_terminal_size().columns
            
            # Clear previous line and print new stats
            sys.stdout.write("\r" + " " * term_width)
            # sys.stdout.write(f"\rCPU: {cpu_percent:.1f}% | RAM: {memory_info.rss/(1024*1024):.1f}MB ({memory_percent:.1f}%) | " +
            #                 f"Disk: R:{disk_read_speed:.1f}KB/s W:{disk_write_speed:.1f}KB/s | " +
            #                 f"Net: ↓{net_recv_speed:.1f}KB/s ↑{net_send_speed:.1f}KB/s{gpu_info}")
            sys.stdout.flush()
            
            # Sleep for a second
            time.sleep(1)
    except Exception as e:
        print(f"\nMonitoring error: {e}")
    finally:
        print("\n--- System Activity Monitor Stopped ---")

def select_model():
    """Interactive function to select the model to test"""
    print("\nSelect the VLM model to test:")
    print("  [1] Qwen2.5-VL-3B-Instruct_4bit")
    print("  [2] Qwen2.5-VL-7B-Instruct-4bit")
    print("  [3] Qwen2-VL-2B-Instruct_4bit")
    print("  [4] Gemma-3-4b-it_4bit")
    print("  [5] PaliGemma-3B-mix-224_4bit")
    print("  [6] DeepSeek-VL-1.3B-chat_4bit")
    print("  [7] DeepSeek-VL-7B-chat_4bit")
    print("  [8] SmolVLM2-256M-Video-Instruct")
    print("  [9] SmolVLM2-500M-Video-Instruct")
    print("  [10] SmolVLM2-2.2B-Instruct")
    print("  [11] GLM-Edge-V-2B")
    print("  [12] InternVL3-1B")
    print("  [13] InternVL3-2B")
    print("  [14] InternVL2.5-4B")
    print("  [15] Florence-2-base")
    print("  [16] Florence-2-large")
    print("  [17] UDOP-large")
    print("  [18] Moondream2-2B")
    print("  [19] Phi-3.5-vision-instruct-4bit")
    print("  [20] ALL")
    
    model_map = {
        '1': "Qwen2.5-VL-3B-Instruct_4bit",
        '2': "Qwen2.5-VL-7B-Instruct-4bit",
        '3': "Qwen2-VL-2B-Instruct_4bit",
        '4': "Gemma-3-4b-it_4bit",
        '5': "PaliGemma-3B-mix-224_4bit",
        '6': "DeepSeek-VL-1.3B-chat_4bit",
        '7': "DeepSeek-VL-7B-chat_4bit",
        '8': "SmolVLM2-256M-Video-Instruct",
        '9': "SmolVLM2-500M-Video-Instruct",
        '10': "SmolVLM2-2.2B-Instruct",
        '11': "GLM-Edge-V-2B",
        '12': "InternVL3-1B",
        '13': "InternVL3-2B",
        '14': "InternVL2.5-4B",
        '15': "Florence-2-base",
        '16': "Florence-2-large",
        '17': "UDOP-large",
        '18': "Moondream2-2B",
        '19': "Phi-3.5-vision-instruct-4bit",
    }
    
    while True:
        choice = input("\nEnter your choice (1-20): ")
        if choice in model_map:
            model_name = model_map[choice]
            print(f"Selected: {model_name}")
            return [model_name]
        elif choice == '20':
            print("Selected: ALL models")
            return list(model_map.values())
        else:
            print("Invalid choice. Please enter a number between 1 and 20.")

def test_model(model_name):
    """Test a specific model with the hardcoded image and question"""
    global monitoring_active
    
    # Check if image exists
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image not found at {IMAGE_PATH}")
        return False
    
    # Create model
    print(f"\nCreating {model_name} model...")
    try:
        model = create_model(model_name)
    except ValueError as e:
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        # Run prediction
        print(f"Running prediction with question: '{QUESTION}'")
        answer = model.predict(IMAGE_PATH, QUESTION)
        print(f"\nModel response:\n{answer}")
        return True
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        model.cleanup()

def main():
    global monitoring_active
    
    # Get absolute path for the image
    global IMAGE_PATH
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    IMAGE_PATH = os.path.join(project_root, IMAGE_PATH)
    
    print(f"Using image: {IMAGE_PATH}")
    print(f"Using question: '{QUESTION}'")
    
    # Start the monitoring thread
    monitoring_active = True
    monitor_thread = threading.Thread(target=monitor_system_activity)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    try:
        # Select model(s) to test
        models = select_model()
        
        # Test each selected model
        results = {}
        for model_name in models:
            print(f"\n{'='*50}")
            print(f"Testing {model_name}...")
            success = test_model(model_name)
            results[model_name] = "Success" if success else "Failed"
        
        # Print summary
        if len(models) > 1:
            print("\n\n" + "="*50)
            print("TESTING SUMMARY")
            print("="*50)
            for model_name, result in results.items():
                print(f"{model_name}: {result}")
    finally:
        # Stop the monitoring thread
        monitoring_active = False
        monitor_thread.join(timeout=2)

if __name__ == "__main__":
    main()
