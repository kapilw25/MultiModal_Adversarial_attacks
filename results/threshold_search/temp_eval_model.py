
import os
import sys
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the original eval_model.py
from scripts.local_llm_tools import create_model, generate_response

def load_data(task, count=None):
    """Load data from the specified task."""
    data_file = f"data/test_extracted/{task}/data.json"
    with open(data_file, "r") as f:
        data_list = json.load(f)
    
    if count is not None:
        data_list = data_list[:count]
    
    return data_list

def process_response(response):
    """Process the response to extract the answer."""
    return response.strip()

# Constants
ENGINE = "Qwen25_VL_3B"
TASK = "chart"
SAMPLE_COUNT = 1  # Only process one image for efficiency
ADV_IMAGE_PATH = "data/test_BB_square/chart/20231114102825506748.png"

# Load data
data_list = load_data(TASK, SAMPLE_COUNT)

# Create model
model, processor = create_model(ENGINE)

# Generate responses
results = []
for data in data_list:
    # Use the adversarial image
    img_path = ADV_IMAGE_PATH
    
    # Generate response
    response = generate_response(model, processor, ENGINE, img_path, data["question"])
    
    # Process response
    processed_response = process_response(response)
    
    # Add to results
    result = {
        "id": data["id"],
        "image": data["image"],
        "question": data["question"],
        "answer": data["answer"],
        "response": processed_response
    }
    results.append(result)

# Save results
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
output_file = f"results/{ENGINE}/eval_{ENGINE}_{TASK}_search_{timestamp}.json"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {output_file}")
print(output_file)  # Print the output file path for the parent script to capture
