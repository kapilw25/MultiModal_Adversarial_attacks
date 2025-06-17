import os
import json
import requests
import base64
from abc import ABC, abstractmethod
import torch
import gc
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class BaseVLModel(ABC):
    """Base class for vision-language models"""
    
    def __init__(self, model_name):
        self.model_name = model_name
    
    @abstractmethod
    def predict(self, image_path, question):
        """Process an image and question to generate an answer"""
        pass
    
    def cleanup(self):
        """Clean up resources"""
        pass

class OpenAIVLModel(BaseVLModel):
    """Class for OpenAI vision-language models like GPT-4o"""
    
    def __init__(self, model_name, model_id="gpt-4o"):
        super().__init__(model_name)
        self.model_id = model_id
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.api_url = "https://api.openai.com/v1/chat/completions"
    
    def _encode_image(self, image_path):
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def predict(self, image_path, question):
        """Process an image and question using OpenAI API"""
        base64_image = self._encode_image(image_path)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model_id,
            "messages": [
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
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }
        
        print(f"Sending request to OpenAI API ({self.model_id})...")
        response = requests.post(self.api_url, headers=headers, json=payload)
        
        if response.status_code != 200:
            print(f"API Error: {response.status_code}")
            print(f"Response: {response.text}")
            raise Exception(f"API request failed with status code {response.status_code}")
            
        result = response.json()
        answer = result["choices"][0]["message"]["content"]
        
        return answer

# Import the QwenVLModelWrapper class
from .qwen_model import QwenVLModelWrapper

class QwenVLModel(BaseVLModel):
    """Class for Qwen vision-language models using the wrapper"""
    
    def __init__(self, model_name):
        super().__init__(model_name)
        self.model = QwenVLModelWrapper(model_name)
    
    def predict(self, image_path, question):
        """Process an image and question using Qwen model wrapper"""
        return self.model.predict(image_path, question)
    
    def cleanup(self):
        """Clean up GPU resources"""
        self.model.cleanup()


# Factory function to create model instances
def create_model(model_name):
    """Create a model instance based on model name"""
    if model_name == "GPT-4o":
        return OpenAIVLModel(model_name)
    elif model_name == "Qwen2.5-VL-3B-Instruct_4bit":
        return QwenVLModel(model_name)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


# Factory function to create model instances
def create_model(model_name):
    """Create a model instance based on model name"""
    if model_name == "GPT-4o":
        return OpenAIVLModel(model_name)
    elif model_name == "Qwen2.5-VL-3B-Instruct_4bit":
        return QwenVLModel(model_name)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
