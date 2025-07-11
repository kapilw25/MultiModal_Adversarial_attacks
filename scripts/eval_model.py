import time
import random, json
from tqdm import tqdm
import os
import base64
from mimetypes import guess_type
import sys
from select_attack import select_attack


# Function to encode a local image into data URL
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


def select_engine():
    """Interactive function to select the engine to use"""
    print("\nSelect the engine to use:")
    print("  [1] OpenAI GPT-4o")
    print("  [2] Qwen25_VL_3B")
    print("  [3] Gemma3_VL_4B")
    print("  [4] ALL")
    
    while True:
        choice = input("\nEnter your choice (1, 2, 3, or 4): ")
        if choice == '1':
            print("Selected: OpenAI GPT-4o")
            # Import for OpenAI GPT-4o
            from llm_tools import send_chat_request_azure
            return [('gpt4o', send_chat_request_azure)]
        elif choice == '2':
            print("Selected: Qwen25_VL_3B")
            # Import for Qwen25_VL_3B
            from local_llm_tools import send_chat_request_azure
            return [('Qwen25_VL_3B', send_chat_request_azure)]
        elif choice == '3':
            print("Selected: Gemma3_VL_4B")
            # Import for Gemma3_VL_4B
            from local_llm_tools import send_chat_request_azure
            return [('Gemma3_VL_4B', send_chat_request_azure)]
        elif choice == '4':
            print("Selected: ALL engines")
            # Import both modules
            from llm_tools import send_chat_request_azure as gpt4o_send_chat
            from local_llm_tools import send_chat_request_azure as local_send_chat
            return [('gpt4o', gpt4o_send_chat), 
                    ('Qwen25_VL_3B', local_send_chat),
                    ('Gemma3_VL_4B', local_send_chat)]
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


def run_evaluation(engine, send_chat_request_azure, task, random_count, output_file, img_dir, attack_name):
    """Run evaluation for a specific attack type"""
    print(f"\nRunning evaluation for {attack_name}")
    print(f"Output file: {output_file}")
    print(f"Image directory: {img_dir}")
    
    # Define input file path
    file_path = f'results/{engine}/eval_{task}.json'
    
    try:
        with open(file_path) as f:
            eval_data = []
            for line in f:
                eval_data.append(json.loads(line))

            human_select = eval_data[:random_count]
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            res_list = []
            try:
                # Open file in write mode initially to clear any existing content
                with open(output_file, 'w') as fout:
                    pass  # Just create/clear the file
                
                for data in tqdm(human_select):
                    img_path = img_dir + data['image']
                    
                    # Check if image exists
                    if not os.path.exists(img_path):
                        print(f"Warning: Image not found: {img_path}")
                        continue
                        
                    url = local_image_to_data_url(img_path)

                    msgs = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": data['text'] + " Answer format (do not generate any other content): The answer is <answer>."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": url
                                    }
                                }
                            ]
                        }
                    ]

                    res, _ = send_chat_request_azure(message_text=msgs, engine=engine, sample_n=1)

                    if 'markers' in data.keys():
                        markers = data['markers']
                    else:
                        # Initialize markers to avoid NoneType error
                        markers = []

                    res = {
                        "question_id": data['question_id'],
                        "prompt": data['text'],
                        "text": res,
                        "truth": data['answer'],
                        "type": data['type'],
                        "answer_id": "",
                        "markers": markers,
                        "model_id": engine,
                        "metadata": {"adversarial": attack_name != "Original (No Attack)"}
                    }

                    res_list.append(res)

                    time.sleep(0.1)
                    
                    with open(output_file, 'a') as fout:
                        fout.write(json.dumps(res) + '\n')
                
                print(f"\nInference completed! Results saved to {output_file}")
                
            except Exception as e:
                print(f"Error during evaluation: {e}")
                # Save partial results
                with open(output_file, 'w') as fout:
                    for res in res_list:
                        fout.write(json.dumps(res) + '\n')
                print(f"Partial results saved to {output_file}")
                
    except FileNotFoundError:
        print(f"Error: Input file {file_path} not found.")
        print(f"Make sure the file exists at {os.path.abspath(file_path)}")
        print("Directory structure should be:")
        print(f"  results/{engine}/eval_{task}.json")


if __name__ == '__main__':
    # Select engine(s) and get the appropriate send_chat_request_azure function(s)
    engine_configs = select_engine()
    
    # Fixed task
    task = 'chart'
    
    # Fixed random count
    random_count = 17
    
    # Process each engine
    for engine, send_chat_request_azure in engine_configs:
        print(f"\n{'='*20} Evaluating {engine} {'='*20}")
        
        # Select attack type(s)
        attack_configs = select_attack(engine, task, random_count)
        
        if not attack_configs:
            print(f"No attacks selected or all selected attacks already have output files for {engine}. Skipping.")
            continue
        
        # Run evaluation for each selected attack
        for output_file, img_dir, attack_name in attack_configs:
            run_evaluation(engine, send_chat_request_azure, task, random_count, output_file, img_dir, attack_name)
