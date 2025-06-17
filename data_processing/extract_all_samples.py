#!/usr/bin/env python
"""
Multi-modal Self-instruct Dataset Extractor (All Examples)

This script extracts all examples from the Multi-modal Self-instruct test dataset,
saves the corresponding images, and creates evaluation JSON files without categorizing by task type.

INPUT:
- Multi-modal Self-instruct dataset loaded from disk (./data)

OUTPUT:
- Images saved to their respective directories based on image_path
- Evaluation JSON file saved to ./benchmark/eval_all_test.json
"""

import os
import json
from datasets import load_from_disk
from tqdm import tqdm

def extract_all_data(dataset_path="./data", output_dir="./benchmark", output_file="eval_all_test.json"):
    """
    Extract all examples from the test dataset, save images, and create evaluation JSON.
    
    Args:
        dataset_path (str): Path to the dataset
        output_dir (str): Directory to save the evaluation JSON file
        output_file (str): Name of the output JSON file
    
    Returns:
        int: Number of examples extracted
    """
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all test examples
    test_examples = dataset['test']
    total_count = len(test_examples)
    
    print(f"Found {total_count} examples in the test dataset")
    
    # Save to a JSON file for evaluation
    print(f"Saving {total_count} examples to evaluation JSON...")
    with open(f"{output_dir}/{output_file}", 'w') as f:
        for example in tqdm(test_examples):
            # Convert the example to a format compatible with eval_model.py
            formatted_example = {
                'question_id': example['question_id'],
                'image': example['image_path'],
                'text': example['question'],
                'answer': example['answer'],
                'type': example['image_path'].split('/')[0] if '/' in example['image_path'] else 'unknown',
                'markers': []
            }
            f.write(json.dumps(formatted_example) + '\n')
    
    # Save images
    print(f"Saving images...")
    for example in tqdm(test_examples):
        # Get the image path
        image_path = example['image_path']
        
        # Create the full path for saving
        full_path = os.path.join(os.path.dirname("."), image_path)
        
        # Create subdirectories if needed
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        # Save the image
        example['image'].save(full_path)
    
    print(f"\nExtraction complete:")
    print(f"- {total_count} examples extracted")
    print(f"- Evaluation JSON saved to {output_dir}/{output_file}")
    print(f"- Images saved to their respective directories based on image_path")
    
    return total_count

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract all examples from Multi-modal Self-instruct test dataset")
    parser.add_argument("--dataset", type=str, default="./data", help="Path to the dataset")
    parser.add_argument("--output", type=str, default="./benchmark", help="Directory to save the evaluation JSON file")
    parser.add_argument("--filename", type=str, default="eval_all_test.json", help="Name of the output JSON file")
    
    args = parser.parse_args()
    
    num_examples = extract_all_data(args.dataset, args.output, args.filename)
