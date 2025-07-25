import json
import os
from collections import defaultdict, Counter

# Load the JSON file
with open('data/test_extracted/benchmark/eval_all_test.json', 'r') as f:
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

# Create dictionaries to store image counts
unique_question_counts = {}  # For unique question IDs
total_question_counts = {}   # For total questions

for task_type in task_types:
    unique_question_counts[task_type] = defaultdict(set)
    total_question_counts[task_type] = Counter()

# Process each entry in the JSON
for entry in data:
    task_type = entry.get('type')
    if task_type in task_types:
        image_path = entry.get('image')
        question_id = entry.get('question_id')
        if image_path and question_id:
            unique_question_counts[task_type][image_path].add(question_id)
            total_question_counts[task_type][image_path] += 1

# Find the top 3 images with the most questions for each task
results = {}
for task_type in task_types:
    if unique_question_counts[task_type]:
        # Sort images by total question count in descending order
        sorted_images = sorted(
            [(img, len(unique_question_counts[task_type][img]), total_question_counts[task_type][img]) 
             for img in unique_question_counts[task_type]],
            key=lambda x: (x[1], x[2]), reverse=True
        )
        # Take top 3 or all if less than 3
        top_images = sorted_images[:3]
        
        results[task_type] = [
            {
                'image': img[0],
                'unique_questions': img[1],
                'total_questions': img[2]
            }
            for img in top_images
        ]

# Print results
print("Top 3 images with maximum questions for each task:")
print("=" * 70)
for task_type, top_images in results.items():
    print(f"Task: {task_type}")
    for i, info in enumerate(top_images, 1):
        print(f"  {i}. Image: {info['image']}")
        print(f"     Unique question IDs: {info['unique_questions']}")
        print(f"     Total questions: {info['total_questions']}")
    print("-" * 70)
