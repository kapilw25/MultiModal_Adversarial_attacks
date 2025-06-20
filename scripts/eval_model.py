import time

# For OpenAI GPT-4o model
# from llm_tools import send_chat_request_azure
# For local Qwen2.5-VL model
from local_llm_tools import send_chat_request_azure

import random, json
from tqdm import tqdm
import os

import base64
from mimetypes import guess_type


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


if __name__ == '__main__':

    eval_data = []
    
    # For OpenAI GPT-4o model
    # engine = 'gpt4o'
    # For local Qwen2.5-VL model
    engine = 'Qwen25_VL_3B'
    
    task = 'chart'
    
    # Define input and output file paths
    file_path = f'results/{engine}/eval_{task}.json'
    output_file = f'results/{engine}/eval_{engine}_{task}_output.json'
    
    with open(file_path) as f:
        for line in f:
            eval_data.append(json.loads(line))

        # random_count = len(eval_data)
        random_count = 17

        human_select = eval_data[:random_count]

        res_list = []
        try:
            for data in tqdm(human_select):
                img_path = 'data/test_extracted/' + data['image']
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
                    "metadata": {}
                }

                res_list.append(res)

                time.sleep(0.1)

                # Ensure output directory exists
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                
                with open(output_file, 'a') as fout:
                    fout.write(json.dumps(res) + '\n')
        except Exception as e:
            print(e)
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w') as fout:
                for res in res_list:
                    fout.write(json.dumps(res) + '\n')
