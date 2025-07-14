from transformers import AutoTokenizer, AutoModel

# Modified script to work around the missing generate method
hf_path = 'jiajunlong/TinyLLaVA-OpenELM-450M-SigLIP-0.89B'

# Try loading with AutoModel instead of AutoModelForCausalLM
print("Loading model with AutoModel...")
model = AutoModel.from_pretrained(hf_path, trust_remote_code=True)
model.cuda()
config = model.config
print("Model loaded successfully")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(hf_path, use_fast=False)
print("Tokenizer loaded successfully")

# Test if the model has the chat method directly
if hasattr(model, 'chat'):
    print("Model has chat method, using it directly...")
    prompt = "What are these?"
    image_url = "http://images.cocodataset.org/test-stuff2017/000000000001.jpg"
    
    try:
        print("Calling chat method...")
        output_text, generation_time = model.chat(prompt=prompt, image=image_url, tokenizer=tokenizer)
        print('Model output:', output_text)
        print('Running time:', generation_time)
    except Exception as e:
        print(f"Error during chat: {e}")
        import traceback
        traceback.print_exc()
else:
    print("Model does not have chat method")
