import sqlite3
import json
import os
import re

# Set engine name
engine = 'gpt4o'

# Create database file with engine-specific path
db_path = f'../{engine}/eval_chart.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create table with auto-increment ID as primary key to keep all rows
cursor.execute('''
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id TEXT,
    image TEXT,
    "Question Prompt" TEXT,
    "Answer Ground Truth" TEXT,
    type TEXT,
    markers TEXT,
    "Prediction" TEXT NULL,
    answer_id TEXT NULL,
    model_id TEXT NULL,
    metadata TEXT NULL,
    result TEXT NULL
)
''')

# Read input data (eval_chart.json)
input_file = f'../{engine}/eval_chart.json'
with open(input_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        
        # Convert markers list to string
        markers_str = json.dumps(data.get('markers', []))
        
        # Insert data with merged columns
        cursor.execute('''
        INSERT INTO evaluations 
        (question_id, image, "Question Prompt", "Answer Ground Truth", type, markers)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            data['question_id'],
            data['image'],
            data['text'],  # Using "text" as "Question Prompt"
            data['answer'],  # Using "answer" as "Answer Ground Truth"
            data['type'],
            markers_str
        ))

# # Function to extract clean prediction from model response
# def extract_prediction(text):
#     # Remove "The answer is " pattern
#     pattern = r'the answer is (.*?)(?:\.\s*$|$)'
#     match = re.search(pattern, text.lower())
#     if match:
#         return match.group(1).strip()
#     return text.strip()

# # Read output data (eval_gpt4o_chart_5.json) if it exists
# output_file = f'../{engine}/eval_gpt4o_chart_5.json'
# if os.path.exists(output_file):
#     with open(output_file, 'r') as f:
#         for line in f:
#             data = json.loads(line)
            
#             # Extract clean prediction from model response
#             prediction = extract_prediction(data['text'])
            
#             # Update database with model response
#             cursor.execute('''
#             UPDATE evaluations
#             SET "Prediction" = ?,
#                 model_id = ?
#             WHERE question_id = ?
#             ''', (
#                 prediction,
#                 data.get('model_id', engine),
#                 data['question_id']
#             ))

# Commit and close
conn.commit()
print(f"Database created at {os.path.abspath(db_path)}")
print(f"Imported data from {input_file}")
# if os.path.exists(output_file):
#     print(f"Updated with model responses from {output_file}")

# Count rows
cursor.execute("SELECT COUNT(*) FROM evaluations")
count = cursor.fetchone()[0]
print(f"Total rows: {count}")

conn.close()
