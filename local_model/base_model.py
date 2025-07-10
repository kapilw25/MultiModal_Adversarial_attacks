from abc import ABC, abstractmethod

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
