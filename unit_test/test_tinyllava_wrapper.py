import sys
import os

# Add the parent directory to the path so we can import from local_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our custom implementation
from local_model.models.TinyLLaVA_SigLIP_pt5B_2pt4B_3B_3pt1B import TinyLLaVASigLIPModelWrapper

def main():
    # Create an instance of our custom wrapper
    model_name = "TinyLLaVA-SigLIP-0.89B"  # Use the smallest model for testing
    print(f"Creating {model_name} wrapper...")
    model = TinyLLaVASigLIPModelWrapper(model_name)
    
    # Test with a sample image and question
    image_url = "http://images.cocodataset.org/test-stuff2017/000000000001.jpg"
    question = "What are these?"
    
    print(f"Running inference with question: '{question}'")
    response = model.predict(image_url, question)
    
    print("\nModel response:")
    print(response)
    
    # Clean up
    model.cleanup()

if __name__ == "__main__":
    main()
