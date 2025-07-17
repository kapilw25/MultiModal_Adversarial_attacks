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
import re

# Define the database path
DB_PATH = "results/robustness.db"
# Define the JSON results path
JSON_PATH = "results/robustness_temp.json"

def ensure_db_directory():
    """Ensure the directory for the database exists."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def create_database(model_columns):
    """
    Create the database schema with a scalable structure.
    
    Args:
        model_columns (list): List of column name tuples (model_name_accuracy, model_name_change)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='attack_comparison'")
    table_exists = cursor.fetchone() is not None
    
    if table_exists:
        # Get existing columns
        cursor.execute("PRAGMA table_info(attack_comparison)")
        existing_columns = [col[1] for col in cursor.fetchall()]
        
        # Add any missing columns
        for col_name, col_type in model_columns:
            if col_name not in existing_columns:
                cursor.execute(f"ALTER TABLE attack_comparison ADD COLUMN {col_name} {col_type}")
                print(f"Added column {col_name} to existing table")
    else:
        # Create the base columns
        columns = [
            "id INTEGER PRIMARY KEY AUTOINCREMENT",
            "task_name TEXT DEFAULT 'chart'",
            "attack_type TEXT NOT NULL",
            "timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        ]
        
        # Add model-specific columns
        for col_name, col_type in model_columns:
            columns.append(f"{col_name} {col_type}")
        
        # Create the table
        cursor.execute(f'''
        CREATE TABLE attack_comparison (
            {', '.join(columns)}
        )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX idx_attack_type ON attack_comparison(attack_type)')
        cursor.execute('CREATE INDEX idx_task_name ON attack_comparison(task_name)')
        
        print("Created new attack_comparison table with all model columns")
    
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

def normalize_model_name(model_name):
    """
    Normalize model name to create valid SQL column names.
    
    Args:
        model_name (str): Original model name
        
    Returns:
        str: Normalized model name suitable for SQL column
    """
    # Convert to lowercase
    name = model_name.lower()
    
    # Replace special characters with underscores
    name = re.sub(r'[^a-z0-9]', '_', name)
    
    # Remove consecutive underscores
    name = re.sub(r'_+', '_', name)
    
    # Remove leading/trailing underscores
    name = name.strip('_')
    
    return name

def get_model_columns(data):
    """
    Generate a list of all required model columns based on the data.
    
    Args:
        data (dict): The loaded JSON data
        
    Returns:
        list: List of tuples (column_name, column_type)
    """
    columns = []
    
    # Extract model names
    model_names = list(data["models"].keys())
    
    # Create accuracy and change columns for each model
    for model_name in model_names:
        norm_name = normalize_model_name(model_name)
        columns.append((f"{norm_name}_accuracy", "REAL DEFAULT 0"))
        columns.append((f"{norm_name}_change", "REAL DEFAULT 0"))
    
    return columns

def prepare_data(data):
    """
    Prepare the data for insertion into the database.
    
    Args:
        data (dict): The loaded JSON data
        
    Returns:
        list: List of dictionaries with column values
    """
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
            norm_name = normalize_model_name(model_name)
            
            # Add accuracy and change columns
            row[f"{norm_name}_accuracy"] = model_data.get("accuracy", 0)
            row[f"{norm_name}_change"] = model_data.get("change", 0)
        
        rows.append(row)
    
    return rows

def store_results_in_db(rows):
    """
    Store the prepared data in the SQLite database.
    
    Args:
        rows (list): List of dictionaries with column values
    """
    if not rows:
        print("No data to store.")
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Clear existing data for the chart task
    cursor.execute("DELETE FROM attack_comparison WHERE task_name = ?", (rows[0]["task_name"],))
    
    # Insert new data
    for row in rows:
        # Dynamically build the SQL query based on the columns in the row
        columns = list(row.keys())
        placeholders = ', '.join(['?' for _ in columns])
        values = [row[col] for col in columns]
        
        query = f"INSERT INTO attack_comparison ({', '.join(columns)}) VALUES ({placeholders})"
        cursor.execute(query, values)
    
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
    # Get all columns except id and timestamp
    display_cols = [col for col in df.columns if col not in ['id', 'timestamp']]
    print(df[display_cols].head().to_string(index=False))
    
    # Calculate some statistics for each model
    print("\nMost effective attacks (largest negative change) by model:")
    
    # Find all change columns
    change_cols = [col for col in df.columns if col.endswith('_change')]
    
    for col in change_cols:
        model_name = col.replace('_change', '')
        if len(df) > 0:  # Make sure we have data
            min_idx = df[col].idxmin()
            print(f"  {model_name}: {df.loc[min_idx, 'attack_type']} ({df.loc[min_idx, col]:.2f}%)")
    
    # Calculate average change by model
    print("\nAverage accuracy change by model:")
    for col in change_cols:
        model_name = col.replace('_change', '')
        print(f"  {model_name}: {df[col].mean():.2f}%")
    
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
    
    # Load results from JSON
    data = load_results_from_json()
    if not data:
        print("No data found. Exiting.")
        return
    
    # Get model columns
    model_columns = get_model_columns(data)
    
    # Create or update the database schema
    create_database(model_columns)
    
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
