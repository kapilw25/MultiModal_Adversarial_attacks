import json
import os
from collections import defaultdict

# Load the JSON file
with open('../data/test_extracted/benchmark/eval_all_test.json', 'r') as f:
    data = [json.loads(line) for line in f]

# List of task types to look for
task_types = [
    'chart', 
    'table', 
    'road_map', 
    'dashboard', 
    'flowchart', 
    'relation_graph', 
    'planar_layout', 
    'visual_puzzle'
]

# Create a dictionary to store image counts for each task type
image_counts = {}
for task_type in task_types:
    image_counts[task_type] = defaultdict(set)

# Process each entry in the JSON
for entry in data:
    task_type = entry.get('type')
    if task_type in task_types:
        image_path = entry.get('image')
        question_id = entry.get('question_id')
        if image_path and question_id:
            image_counts[task_type][image_path].add(question_id)

# Find the top 3 images with the most questions for each task
results = {}
for task_type, images in image_counts.items():
    if images:
        # Sort images by question count in descending order
        sorted_images = sorted(images.items(), key=lambda x: len(x[1]), reverse=True)
        # Take top 3 or all if less than 3
        top_images = sorted_images[:3]
        
        results[task_type] = [
            {
                'image': img[0],
                'question_count': len(img[1]),
                'question_ids': list(img[1])
            }
            for img in top_images
        ]

# Print results
print("Top 3 images with maximum unique question_ids for each task:")
print("=" * 70)
for task_type, top_images in results.items():
    print(f"Task: {task_type}")
    for i, info in enumerate(top_images, 1):
        print(f"  {i}. Image: {info['image']}")
        print(f"     Number of unique questions: {info['question_count']}")
    print("-" * 70)
