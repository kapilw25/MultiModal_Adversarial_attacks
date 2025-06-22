#!/usr/bin/env python3
"""
Test script to verify that local_llm_tools.py works correctly.
"""

import os
import sys
import base64
from mimetypes import guess_type

# Import the send_chat_request function from local_llm_tools
from local_llm_tools import send_chat_request

# Function to encode a local image into data URL (copied from eval_model.py)
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

def test_local_llm_tools():
    """Test function to verify local_llm_tools functionality"""
    
    # Test data
    image_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             "data/test_extracted/chart/20231114102825506748.png")
    question = "How many different coloured lines are there?"
    
    # Check if the image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
    
    # Convert image to data URL
    url = local_image_to_data_url(image_path)
    
    # Create message in the format expected by send_chat_request
    msgs = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": question + " Answer format (do not generate any other content): The answer is <answer>."
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
    
    # Call send_chat_request
    print("Sending request to local Qwen2.5-VL model...")
    response, all_responses = send_chat_request(message_text=msgs, engine="Qwen25_VL_3B")
    
    # Print the response
    print(f"\nModel response: {response}")
    print(f"All responses: {all_responses}")
    
    print("Test completed successfully")

if __name__ == "__main__":
    test_local_llm_tools()
