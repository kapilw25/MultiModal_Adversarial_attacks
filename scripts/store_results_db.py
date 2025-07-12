#!/usr/bin/env python3
"""
Store VLM robustness evaluation results in a SQLite database.
This script reads evaluation results from a JSON file and stores them
in a SQLite database with a structured format (attack types as rows, models as columns).
Designed to be scalable for future expansion to multiple models, tasks, and questions.
"""

import os
import sqlite3
import pandas as pd
import json
from datetime import datetime

# Define the database path
DB_PATH = "results/robustness.db"
# Define the JSON results path
JSON_PATH = "results/robustness_temp.json"

def ensure_db_directory():
    """Ensure the directory for the database exists."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def create_database():
    """Create the database schema with a scalable structure."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create the main table with attack_type as rows and model data as columns
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS attack_comparison (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_name TEXT DEFAULT 'chart',
        attack_type TEXT NOT NULL,
        gpt4o_accuracy REAL NOT NULL,
        gpt4o_change REAL NOT NULL,
        qwen_accuracy REAL NOT NULL,
        qwen_change REAL NOT NULL,
        gemma_accuracy REAL NOT NULL,
        gemma_change REAL NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create an index on attack_type for faster lookups
    cursor.execute('''
    CREATE INDEX IF NOT EXISTS idx_attack_type ON attack_comparison(attack_type)
    ''')
    
    # Create an index on task_name for future scalability
    cursor.execute('''
    CREATE INDEX IF NOT EXISTS idx_task_name ON attack_comparison(task_name)
    ''')
    
    conn.commit()
    conn.close()

def load_results_from_json():
    """Load evaluation results from the JSON file."""
    if not os.path.exists(JSON_PATH):
        print(f"Error: JSON file {JSON_PATH} not found.")
        return None
    
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
    
    return data

def prepare_data(data):
    """Prepare the data for insertion into the database."""
    if not data:
        return []
    
    # Extract model names
    model_names = list(data["models"].keys())
    
    # Get all unique attack types across all models
    attack_types = set()
    for model_name in model_names:
        attack_types.update(data["models"][model_name].keys())
    
    # Create a list of rows for insertion
    rows = []
    for attack_type in attack_types:
        row = {
            "attack_type": attack_type,
            "task_name": data["metadata"]["task_name"]
        }
        
        # Add data for each model
        for model_name in model_names:
            model_data = data["models"][model_name].get(attack_type, {})
            
            # Use model name to determine column prefix
            if "gpt4o" in model_name.lower():
                prefix = "gpt4o"
            elif "qwen" in model_name.lower():
                prefix = "qwen"
            elif "gemma" in model_name.lower():
                prefix = "gemma"
            else:
                # For future models, we'll need to extend the database schema
                print(f"Warning: Unknown model {model_name}, skipping")
                continue
            
            # Add accuracy and change columns
            row[f"{prefix}_accuracy"] = model_data.get("accuracy", 0)
            row[f"{prefix}_change"] = model_data.get("change", 0)
        
        # Ensure all required columns have values (even if 0)
        for prefix in ["gpt4o", "qwen", "gemma"]:
            if f"{prefix}_accuracy" not in row:
                row[f"{prefix}_accuracy"] = 0
            if f"{prefix}_change" not in row:
                row[f"{prefix}_change"] = 0
        
        rows.append(row)
    
    return rows

def store_results_in_db(rows):
    """Store the prepared data in the SQLite database."""
    if not rows:
        print("No data to store.")
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Clear existing data for the chart task
    cursor.execute("DELETE FROM attack_comparison WHERE task_name = ?", (rows[0]["task_name"],))
    
    # Insert new data
    for row in rows:
        cursor.execute('''
        INSERT INTO attack_comparison 
        (task_name, attack_type, gpt4o_accuracy, gpt4o_change, qwen_accuracy, qwen_change, gemma_accuracy, gemma_change)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            row["task_name"],
            row["attack_type"],
            row["gpt4o_accuracy"],
            row["gpt4o_change"],
            row["qwen_accuracy"],
            row["qwen_change"],
            row["gemma_accuracy"],
            row["gemma_change"]
        ))
    
    conn.commit()
    conn.close()

def verify_database():
    """Verify the database was created correctly by running some test queries."""
    conn = sqlite3.connect(DB_PATH)
    
    # Use pandas to read the data for easier display
    df = pd.read_sql_query("SELECT * FROM attack_comparison", conn)
    
    print("\n=== Database Verification ===")
    print(f"Total rows (attack types): {len(df)}")
    
    # Display the table structure
    print("\nTable structure:")
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(attack_comparison)")
    for col in cursor.fetchall():
        print(f"  {col[1]} ({col[2]})")
    
    # Display a sample of the data
    print("\nSample data (first 5 rows):")
    print(df[['task_name', 'attack_type', 'gpt4o_accuracy', 'gpt4o_change', 
              'qwen_accuracy', 'qwen_change', 'gemma_accuracy', 'gemma_change']].head().to_string(index=False))
    
    # Calculate some statistics
    print("\nMost effective attacks (largest negative change) by model:")
    
    # For GPT4o
    min_gpt4o_idx = df['gpt4o_change'].idxmin()
    print(f"  GPT4o: {df.loc[min_gpt4o_idx, 'attack_type']} ({df.loc[min_gpt4o_idx, 'gpt4o_change']:.2f}%)")
    
    # For Qwen
    min_qwen_idx = df['qwen_change'].idxmin()
    print(f"  Qwen: {df.loc[min_qwen_idx, 'attack_type']} ({df.loc[min_qwen_idx, 'qwen_change']:.2f}%)")
    
    # For Gemma
    min_gemma_idx = df['gemma_change'].idxmin()
    print(f"  Gemma: {df.loc[min_gemma_idx, 'attack_type']} ({df.loc[min_gemma_idx, 'gemma_change']:.2f}%)")
    
    # Calculate average change by model
    print("\nAverage accuracy change by model:")
    print(f"  GPT4o: {df['gpt4o_change'].mean():.2f}%")
    print(f"  Qwen: {df['qwen_change'].mean():.2f}%")
    print(f"  Gemma: {df['gemma_change'].mean():.2f}%")
    
    # Show database size
    cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
    db_size = cursor.fetchone()[0]
    print(f"\nDatabase size: {db_size / 1024:.2f} KB")
    
    conn.close()

def main():
    """Main function to run the script."""
    print("Starting to store evaluation results in database...")
    
    # Ensure the database directory exists
    ensure_db_directory()
    
    # Create the database schema
    create_database()
    
    # Load results from JSON
    data = load_results_from_json()
    if not data:
        print("No data found. Exiting.")
        return
    
    # Prepare the data
    rows = prepare_data(data)
    print(f"Prepared {len(rows)} rows for insertion.")
    
    # Store results in the database
    store_results_in_db(rows)
    print(f"Stored results in database: {DB_PATH}")
    
    # Verify the database
    verify_database()
    
    print("\nDatabase creation complete!")

if __name__ == "__main__":
    main()
