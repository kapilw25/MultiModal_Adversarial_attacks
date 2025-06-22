#!/usr/bin/env python3
"""
Simple test script to verify if QwenVLModel class is working correctly.
This is for testing purposes only, not for production use.
"""

import os
import sys
import traceback

# Add the parent directory to sys.path to import local_model modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the model factory function
from local_model.model_classes import create_model

def test_qwen_model():
    """Test function to verify QwenVLModel functionality"""
    
    # Test data
    data = {
        'image': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             "data/test_extracted/chart/20231114102825506748.png"),
        'text': "How many different coloured lines are there?"
    }
    
    # Check if the image file exists
    if not os.path.exists(data['image']):
        print(f"Error: Image file not found at {data['image']}")
        print("Please make sure the path is correct or adjust it in the script.")
        return
    
    try:
        # Create the model
        model_name = "Qwen2.5-VL-3B-Instruct_4bit"
        print(f"Creating model: {model_name}")
        model = create_model(model_name)
        
        # Run prediction
        print(f"Running prediction on image: {data['image']}")
        print(f"Question: {data['text']}")
        response = model.predict(data['image'], data['text'])
        
        # Print the response
        print(f"\nModel response: {response}")
        
        # Clean up resources
        model.cleanup()
        print("Test completed successfully")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        print("\nDetailed traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    test_qwen_model()
