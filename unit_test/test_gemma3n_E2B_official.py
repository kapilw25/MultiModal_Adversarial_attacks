from transformers import AutoProcessor, Gemma3nForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import requests
import torch
import gc
import os

# Set environment variable for memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Clear GPU memory
torch.cuda.empty_cache()
gc.collect()

model_id = "google/gemma-3n-e2b-it"
model = None

print("🚀 Starting Gemma 3n E2B loading with sequential fallback approaches...")
print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved() / 1024**3:.2f} GB reserved")

# APPROACH 1: 4-bit quantization (most memory efficient)
print("\n📦 APPROACH 1: Trying 4-bit quantization...")
try:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model = Gemma3nForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        max_memory={0: "6GB", "cpu": "16GB"}
    ).eval()
    
    print("✅ SUCCESS: Model loaded with 4-bit quantization")
    print(f"GPU memory after loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated")
    
except Exception as e:
    print(f"❌ FAILED: 4-bit quantization failed: {e}")
    model = None
    torch.cuda.empty_cache()
    gc.collect()

# APPROACH 2: 8-bit quantization (fallback)
if model is None:
    print("\n📦 APPROACH 2: Trying 8-bit quantization...")
    try:
        model = Gemma3nForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            load_in_8bit=True,
            device_map="auto",
            low_cpu_mem_usage=True,
            max_memory={0: "6GB", "cpu": "16GB"}
        ).eval()
        
        print("✅ SUCCESS: Model loaded with 8-bit quantization")
        print(f"GPU memory after loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated")
        
    except Exception as e:
        print(f"❌ FAILED: 8-bit quantization failed: {e}")
        model = None
        torch.cuda.empty_cache()
        gc.collect()

# APPROACH 3: CPU PLE with specific device mapping
if model is None:
    print("\n🖥️ APPROACH 3: Trying CPU PLE with custom device mapping...")
    try:
        device_map = {
            # Keep embedding layers on CPU to save GPU memory
            'model.embed_tokens': 'cpu',
            'model.embed_tokens_per_layer': 'cpu',
            'lm_head': 'cpu',
            
            # Keep vision encoder on CPU too
            'model.vision_tower': 'cpu',
            'model.multi_modal_projector': 'cpu',
            
            # Only keep the main language model layers on GPU
            'model.language_model': 0,
        }
        
        model = Gemma3nForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            low_cpu_mem_usage=True,
            max_memory={0: "5GB", "cpu": "32GB"}
        ).eval()
        
        print("✅ SUCCESS: Model loaded with CPU PLE")
        print(f"GPU memory after loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated")
        
    except Exception as e:
        print(f"❌ FAILED: CPU PLE failed: {e}")
        model = None
        torch.cuda.empty_cache()
        gc.collect()

# APPROACH 4: Full CPU loading then selective GPU movement
if model is None:
    print("\n🔄 APPROACH 4: Trying full CPU loading then selective GPU movement...")
    try:
        model = Gemma3nForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True
        ).eval()
        
        print("✅ Model loaded on CPU, attempting to move core components to GPU...")
        
        # Try to move only the language model to GPU
        try:
            model.model.language_model = model.model.language_model.to("cuda")
            print("✅ SUCCESS: Core language model moved to GPU")
            print(f"GPU memory after moving: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated")
        except Exception as gpu_move_error:
            print(f"⚠️ WARNING: Could not move to GPU: {gpu_move_error}")
            print("✅ Continuing with CPU-only inference")
        
    except Exception as e:
        print(f"❌ FAILED: Full CPU loading failed: {e}")
        model = None

# APPROACH 5: Last resort - CPU only
if model is None:
    print("\n💻 APPROACH 5: Last resort - CPU only...")
    try:
        model = Gemma3nForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map="cpu",
            low_cpu_mem_usage=True
        ).eval()
        
        print("✅ SUCCESS: Model loaded on CPU only")
        
    except Exception as e:
        print(f"❌ FAILED: Even CPU-only loading failed: {e}")
        raise RuntimeError("All loading approaches failed!")

if model is None:
    raise RuntimeError("Failed to load model with any approach!")

# Load processor
print("\n🔧 Loading processor...")
processor = AutoProcessor.from_pretrained(model_id)

# Test inference
print("\n🧪 Testing inference...")
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
            {"type": "text", "text": "Describe this image in detail."}
        ]
    }
]

try:
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    
    # Handle device placement carefully
    model_device = next(model.parameters()).device
    print(f"Model device: {model_device}")
    
    # Move inputs to appropriate device
    inputs_on_device = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs_on_device[k] = v.to(model_device)
        else:
            inputs_on_device[k] = v
    
    input_len = inputs_on_device["input_ids"].shape[-1]
    
    print("🎯 Running inference...")
    with torch.inference_mode():
        generation = model.generate(
            **inputs_on_device, 
            max_new_tokens=50,
            do_sample=False,
            use_cache=True,
            pad_token_id=processor.tokenizer.eos_token_id
        )
        generation = generation[0][input_len:]
    
    decoded = processor.decode(generation, skip_special_tokens=True)
    print(f"\n🎉 SUCCESS! Model output:\n{decoded}")
    
    print(f"\nFinal GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved() / 1024**3:.2f} GB reserved")
    
except Exception as e:
    print(f"❌ INFERENCE FAILED: {e}")
    print("Model loaded successfully but inference failed. This might be due to:")
    print("- Input processing issues")
    print("- Memory allocation during generation")
    print("- Device placement problems")
    raise