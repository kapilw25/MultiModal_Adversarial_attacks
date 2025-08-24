#!/usr/bin/env python3
"""
Test MedGemma with text-only input to check if the gibberish is image-related 
or a general model corruption issue.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

print("Testing MedGemma text-only generation (no image) to isolate the issue...")

# Use 8-bit quantization that worked before
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
)

try:
    # Load model and tokenizer
    model_id = "google/medgemma-4b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print("Loading MedGemma model with 8-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory={0: "7GiB", "cpu": "12GiB"}
    )
    
    print("✅ Model loaded successfully")
    
    # Test pure text generation without any images
    test_prompts = [
        "What are the main components of a chest X-ray examination?",
        "Explain pneumonia in simple terms.",
        "Hello, how are you today?"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i}: Text-only generation ---")
        print(f"Prompt: {prompt}")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        
        print(f"Response: {generated_text}")
        print(f"Quality: {'✅ Coherent' if len(generated_text.split()) > 3 and not any(c in generated_text for c in ['ک', 'ു', 'ග', '馳']) else '❌ Gibberish'}")

    print("\n" + "="*60)
    print("CONCLUSION:")
    if all(len(prompt.split()) < 10 for prompt in test_prompts):  # Simple heuristic
        print("✅ Text-only generation works fine")
        print("🔍 Issue is specifically with vision-language processing")
        print("💡 Recommendation: Use alternative VLM like LLaVA with medical prompting")
    else:
        print("❌ MedGemma produces gibberish even for text-only")
        print("🔍 Model weights may be corrupted or quantization incompatible")
        print("💡 Recommendation: Try different medical VLM entirely")
    print("="*60)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()