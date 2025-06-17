#!/usr/bin/env python
"""
Split JSON by Task Type

This script reads the eval_all_test.json file and splits it into separate JSON files
for each task type, storing them in the same location.

Input: data/test_extracted/benchmark/eval_all_test.json
Output: data/test_extracted/benchmark/eval_<task>.json for each task type
"""

import json
import os

def split_json_by_task(input_file, output_dir):
    """
    Split JSON file by task type.
    
    Args:
        input_file (str): Path to the input JSON file
        output_dir (str): Directory to save the output JSON files
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store examples by task type
    task_examples = {}
    
    # Read the input file
    print(f"Reading input file: {input_file}")
    with open(input_file, 'r') as f:
        for line in f:
            example = json.loads(line.strip())
            task_type = example['type']
            
            if task_type not in task_examples:
                task_examples[task_type] = []
            
            task_examples[task_type].append(example)
    
    # Write examples to separate files by task type
    for task_type, examples in task_examples.items():
        output_file = os.path.join(output_dir, f"eval_{task_type}.json")
        print(f"Writing {len(examples)} examples to {output_file}")
        
        with open(output_file, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
    
    # Create a combined file with all examples
    output_file = os.path.join(output_dir, "eval_all.json")
    print(f"Writing all examples to {output_file}")
    
    with open(output_file, 'w') as f:
        for examples in task_examples.values():
            for example in examples:
                f.write(json.dumps(example) + '\n')
    
    print("\nSummary:")
    for task_type, examples in task_examples.items():
        print(f"- {task_type}: {len(examples)} examples")
    
    print(f"Total: {sum(len(examples) for examples in task_examples.values())} examples")

if __name__ == "__main__":
    input_file = "./data/test_extracted/benchmark/eval_all_test.json"
    output_dir = "./data/test_extracted/benchmark"
    
    split_json_by_task(input_file, output_dir)
