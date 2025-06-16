#!/usr/bin/env python
"""
Multi-modal Self-instruct Dataset Extractor (Version 2)

This script extracts all examples from the Multi-modal Self-instruct dataset,
saves the corresponding images to their respective directories, and creates
evaluation JSON files in the specified output directory structure.

INPUT:
- Multi-modal Self-instruct dataset loaded from disk (./data)

OUTPUT:
- Images saved to their respective directories in data/test_extracted_2
- Evaluation JSON file saved to data/test_extracted_2/benchmark/eval_all_test.json
"""

import os
import json
from datasets import load_from_disk
from tqdm import tqdm

# Define task types and their corresponding image path prefixes
TASK_TYPES = {
    "chart": "chart/",
    "table": "table/",
    "dashboard": "dashboard/",
    "flowchart": "flowchart/",
    "relation_graph": "relation_graph/",
    "road_map": "road_map/",
    "visual_puzzle": "visual_puzzle/",
    "planar_layout": "planar_layout/"
}

def extract_all_data(dataset_path="./data", output_dir="./data/test_extracted_2"):
    """
    Extract all examples from the dataset, save images, and create evaluation JSON.
    
    Args:
        dataset_path (str): Path to the dataset
        output_dir (str): Base directory for output
    
    Returns:
        dict: Number of examples and images extracted for each task type
    """
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    
    # Create output directories
    benchmark_dir = os.path.join(output_dir, "benchmark")
    os.makedirs(benchmark_dir, exist_ok=True)
    
    for task_type in TASK_TYPES.keys():
        task_dir = os.path.join(output_dir, task_type)
        os.makedirs(task_dir, exist_ok=True)
    
    # Dictionary to store counts
    counts = {
        "examples": {},
        "images": {}
    }
    
    # Get all test examples
    test_examples = dataset['test']
    total_count = len(test_examples)
    
    print(f"Found {total_count} examples in the test dataset")
    
    # Process and categorize examples by task type
    task_examples = {}
    for task_type in TASK_TYPES.keys():
        task_examples[task_type] = []
    
    # First pass: categorize examples by task type based on image_path
    print("Categorizing examples by task type...")
    for example in tqdm(test_examples):
        image_path = example['image_path']
        assigned = False
        
        for task_type, prefix in TASK_TYPES.items():
            if image_path.startswith(prefix):
                task_examples[task_type].append(example)
                assigned = True
                break
        
        if not assigned:
            # Try to infer task type from the image path
            if "chart" in image_path.lower():
                task_examples["chart"].append(example)
            elif "table" in image_path.lower():
                task_examples["table"].append(example)
            elif "dashboard" in image_path.lower():
                task_examples["dashboard"].append(example)
            elif "flow" in image_path.lower():
                task_examples["flowchart"].append(example)
            elif "map" in image_path.lower():
                task_examples["road_map"].append(example)
            elif "puzzle" in image_path.lower():
                task_examples["visual_puzzle"].append(example)
            elif "layout" in image_path.lower() or "plan" in image_path.lower():
                task_examples["planar_layout"].append(example)
            elif "graph" in image_path.lower() or "relation" in image_path.lower():
                task_examples["relation_graph"].append(example)
            else:
                # Default to chart if we can't determine the type
                task_examples["chart"].append(example)
    
    # Count examples by task type
    for task_type, examples in task_examples.items():
        counts["examples"][task_type] = len(examples)
    
    # Save all examples to a combined JSON file
    print(f"\nSaving all {total_count} examples to evaluation JSON...")
    
    with open(f"{benchmark_dir}/eval_all_test.json", 'w') as f:
        for task_type, examples in task_examples.items():
            for example in examples:
                # Convert the example to a format compatible with eval_model.py
                formatted_example = {
                    'question_id': example['question_id'],
                    'image': f"{task_type}/{os.path.basename(example['image_path'])}",
                    'text': example['question'],
                    'answer': example['answer'],
                    'type': task_type,
                    'markers': []
                }
                f.write(json.dumps(formatted_example) + '\n')
    
    # Save images
    print(f"Saving images to their respective directories...")
    image_counts = {task_type: 0 for task_type in TASK_TYPES.keys()}
    saved_images = {task_type: set() for task_type in TASK_TYPES.keys()}
    
    for task_type, examples in task_examples.items():
        print(f"\nProcessing {len(examples)} {task_type} examples...")
        task_dir = os.path.join(output_dir, task_type)
        
        for example in tqdm(examples):
            image_path = example['image_path']
            image_filename = os.path.basename(image_path)
            
            # Skip if we've already saved this image
            if image_filename in saved_images[task_type]:
                continue
            
            # Save the image
            output_path = os.path.join(task_dir, image_filename)
            
            # Save the image using PIL
            example['image'].save(output_path)
            saved_images[task_type].add(image_filename)
            image_counts[task_type] += 1
    
    # Store image counts
    for task_type, count in image_counts.items():
        counts["images"][task_type] = count
    
    # Print summary
    print("\nExtraction complete:")
    print("\nExample counts by task type:")
    for task_type, count in counts["examples"].items():
        print(f"- {task_type}: {count} examples")
    print(f"- Total: {total_count} examples")
    
    print("\nImage counts by task type:")
    for task_type, count in counts["images"].items():
        print(f"- {task_type}: {count} images")
    print(f"- Total: {sum(counts['images'].values())} images")
    
    print(f"\nEvaluation JSON saved to {benchmark_dir}/eval_all_test.json")
    print(f"Images saved to their respective directories in {output_dir}")
    
    return counts

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract all examples from Multi-modal Self-instruct dataset")
    parser.add_argument("--dataset", type=str, default="./data", help="Path to the dataset")
    parser.add_argument("--output", type=str, default="./data/test_extracted_2", help="Base directory for output")
    
    args = parser.parse_args()
    
    counts = extract_all_data(args.dataset, args.output)
