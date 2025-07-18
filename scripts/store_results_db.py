#!/usr/bin/env python3
"""
Store VLM robustness evaluation results in a normalized SQLite database.
This script reads evaluation results from a JSON file and stores them
in a SQLite database with a normalized structure:
- 3 dimension tables (attack_types, model_families, size_categories)
- 1 fact table (results)

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

def create_database():
    """
    Create the normalized database schema with dimension and fact tables.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='results'")
    tables_exist = cursor.fetchone() is not None
    
    if not tables_exist:
        # Create attack_types dimension table
        cursor.execute('''
        CREATE TABLE attack_types (
            attack_id INTEGER PRIMARY KEY AUTOINCREMENT,
            attack_name TEXT NOT NULL UNIQUE,
            attack_category TEXT NOT NULL
        )
        ''')
        
        # Create model_families dimension table
        cursor.execute('''
        CREATE TABLE model_families (
            family_id INTEGER PRIMARY KEY AUTOINCREMENT,
            family_name TEXT NOT NULL UNIQUE
        )
        ''')
        
        # Create size_categories dimension table
        cursor.execute('''
        CREATE TABLE size_categories (
            size_id INTEGER PRIMARY KEY AUTOINCREMENT,
            size_range TEXT NOT NULL UNIQUE
        )
        ''')
        
        # Create tasks dimension table
        cursor.execute('''
        CREATE TABLE tasks (
            task_id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_name TEXT NOT NULL UNIQUE
        )
        ''')
        
        # Create models dimension table
        cursor.execute('''
        CREATE TABLE models (
            model_id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL UNIQUE,
            family_id INTEGER,
            size_id INTEGER,
            FOREIGN KEY (family_id) REFERENCES model_families(family_id),
            FOREIGN KEY (size_id) REFERENCES size_categories(size_id)
        )
        ''')
        
        # Create results fact table
        cursor.execute('''
        CREATE TABLE results (
            result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            attack_id INTEGER,
            model_id INTEGER,
            task_id INTEGER,
            accuracy REAL DEFAULT 0,
            accuracy_change REAL DEFAULT 0,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (attack_id) REFERENCES attack_types(attack_id),
            FOREIGN KEY (model_id) REFERENCES models(model_id),
            FOREIGN KEY (task_id) REFERENCES tasks(task_id)
        )
        ''')
        
        # Create a human-readable attack_effectiveness table
        # This table will have columns for each model showing both accuracy and degradation
        cursor.execute('''
        CREATE TABLE attack_effectiveness (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            attack_name TEXT NOT NULL,
            attack_category TEXT NOT NULL,
            task_name TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX idx_attack_id ON results(attack_id)')
        cursor.execute('CREATE INDEX idx_model_id ON results(model_id)')
        cursor.execute('CREATE INDEX idx_task_id ON results(task_id)')
        cursor.execute('CREATE INDEX idx_attack_effectiveness_name ON attack_effectiveness(attack_name)')
        cursor.execute('CREATE INDEX idx_attack_effectiveness_category ON attack_effectiveness(attack_category)')
        
        print("Created new normalized database schema with dimension and fact tables")
    
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

def get_attack_category(attack_type):
    """
    Determine the attack category based on the attack type.
    
    Args:
        attack_type (str): The type of attack
        
    Returns:
        str: Either 'Transfer' or 'Black-Box' or 'Original' for no attack
    """
    # Define transfer-based attacks
    transfer_attacks = [
        'FGSM', 'CW-L0', 'CW-L2', 'CW-L∞', 'L-BFGS', 'JSMA', 'DeepFool', 'PGD'
    ]
    
    # Define black-box attacks
    black_box_attacks = [
        'Square', 'HopSkipJump', 'Pixel', 'SimBA', 'ZOO', 'Boundary', 
        'Query-Efficient BB', 'Spatial', 'GeoDA'
    ]
    
    # Check if it's the original (no attack)
    if attack_type == 'Original':
        return 'Original'
    
    # Check if it's a transfer-based attack
    for attack in transfer_attacks:
        if attack.lower() == attack_type.lower():
            return 'Transfer'
    
    # Check if it's a black-box attack
    for attack in black_box_attacks:
        if attack.lower() == attack_type.lower():
            return 'Black-Box'
    
    # Default to Unknown if not recognized
    return 'Unknown'

def get_model_family(model_name):
    """
    Determine the model family based on the model name.
    
    Args:
        model_name (str): The name of the model
        
    Returns:
        str: The model family name
    """
    model_name_lower = model_name.lower()
    
    if 'deepseek' in model_name_lower and 'vl2' in model_name_lower:
        return 'DeepSeek VL2'
    elif 'deepseek' in model_name_lower:
        return 'DeepSeek VL'
    elif 'qwen' in model_name_lower:
        return 'Qwen VL'
    elif 'gemma' in model_name_lower or 'paligemma' in model_name_lower:
        return 'Google'
    elif 'smolvlm' in model_name_lower:
        return 'SmolVLM'
    elif 'glm' in model_name_lower:
        return 'GLM Edge'
    elif 'moondream' in model_name_lower:
        return 'Moondream'
    elif 'florence' in model_name_lower or 'phi' in model_name_lower:
        return 'Microsoft'
    elif 'llava' in model_name_lower:
        return 'LLaVA Hybrid'
    elif 'internvl' in model_name_lower:
        return 'InternVL'
    elif 'blip' in model_name_lower:
        return 'Salesforce'
    elif 'gpt4o' in model_name_lower:
        return 'OpenAI'
    else:
        return 'Other'

def get_size_category(model_name):
    """
    Determine the size category based on the model name.
    
    Args:
        model_name (str): The name of the model
        
    Returns:
        str: The size category
    """
    model_name_lower = model_name.lower()
    
    # Extract size information from model name
    size_mapping = {
        'pt25b': '(0-1]B',
        'pt5b': '(0-1]B',
        '1b': '(0-1]B',
        '1pt3b': '(1-2]B',
        '2b': '(1-2]B',
        '2pt2b': '(2-3]B',
        '3b': '(2-3]B',
        '4b': '(3-4]B',
        '7b': '(6-7]B'
    }
    
    # Special cases
    if 'florence2_pt23b' in model_name_lower:
        return '(0-1]B'
    elif 'florence2_pt77b' in model_name_lower:
        return '(0-1]B'
    elif 'qwen2_vl_2b' in model_name_lower:
        return '(2-3]B'  # 2.21B actual size
    elif 'qwen25_vl_3b' in model_name_lower:
        return '(3-4]B'  # 3.75B actual size
    elif 'phi3pt5_vision_4b' in model_name_lower:
        return '(4-5]B'  # 4.15B actual size
    elif 'gpt4o' in model_name_lower:
        return 'Cloud API'
    
    # Check for size indicators in the model name
    for size_key, size_category in size_mapping.items():
        if size_key in model_name_lower:
            return size_category
    
    return 'Unknown'

def populate_dimension_tables(data, conn):
    """
    Populate the dimension tables with data from the JSON file.
    
    Args:
        data (dict): The loaded JSON data
        conn (sqlite3.Connection): Database connection
        
    Returns:
        dict: Mapping of dimension values to IDs
    """
    cursor = conn.cursor()
    
    # Extract model names and attack types
    model_names = list(data["models"].keys())
    attack_types = set()
    for model_name in model_names:
        attack_types.update(data["models"][model_name].keys())
    
    # Populate attack_types table
    attack_id_map = {}
    for attack_type in attack_types:
        category = get_attack_category(attack_type)
        cursor.execute(
            "INSERT OR IGNORE INTO attack_types (attack_name, attack_category) VALUES (?, ?)",
            (attack_type, category)
        )
        cursor.execute("SELECT attack_id FROM attack_types WHERE attack_name = ?", (attack_type,))
        attack_id_map[attack_type] = cursor.fetchone()[0]
    
    # Populate model_families table
    family_id_map = {}
    for model_name in model_names:
        family = get_model_family(model_name)
        cursor.execute(
            "INSERT OR IGNORE INTO model_families (family_name) VALUES (?)",
            (family,)
        )
        cursor.execute("SELECT family_id FROM model_families WHERE family_name = ?", (family,))
        family_id_map[family] = cursor.fetchone()[0]
    
    # Populate size_categories table
    size_id_map = {}
    for model_name in model_names:
        size = get_size_category(model_name)
        cursor.execute(
            "INSERT OR IGNORE INTO size_categories (size_range) VALUES (?)",
            (size,)
        )
        cursor.execute("SELECT size_id FROM size_categories WHERE size_range = ?", (size,))
        size_id_map[size] = cursor.fetchone()[0]
    
    # Populate tasks table
    task_name = data["metadata"]["task_name"]
    cursor.execute(
        "INSERT OR IGNORE INTO tasks (task_name) VALUES (?)",
        (task_name,)
    )
    cursor.execute("SELECT task_id FROM tasks WHERE task_name = ?", (task_name,))
    task_id = cursor.fetchone()[0]
    
    # Populate models table
    model_id_map = {}
    for model_name in model_names:
        family = get_model_family(model_name)
        size = get_size_category(model_name)
        cursor.execute(
            "INSERT OR IGNORE INTO models (model_name, family_id, size_id) VALUES (?, ?, ?)",
            (model_name, family_id_map[family], size_id_map[size])
        )
        cursor.execute("SELECT model_id FROM models WHERE model_name = ?", (model_name,))
        model_id_map[model_name] = cursor.fetchone()[0]
    
    conn.commit()
    
    return {
        'attack_id_map': attack_id_map,
        'model_id_map': model_id_map,
        'task_id': task_id
    }

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

def create_attack_effectiveness_table(data, conn):
    """
    Create and populate the attack_effectiveness table with model-specific columns.
    This table will have a column for each model's accuracy and degradation.
    
    Args:
        data (dict): The loaded JSON data
        conn (sqlite3.Connection): Database connection
    """
    cursor = conn.cursor()
    
    # Extract model names and attack types
    model_names = list(data["models"].keys())
    attack_types = set()
    for model_name in model_names:
        attack_types.update(data["models"][model_name].keys())
    
    # First, check if the table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='attack_effectiveness'")
    table_exists = cursor.fetchone() is not None
    
    # If the table exists, we need to check if it has all the required columns
    if table_exists:
        # Get existing columns
        cursor.execute("PRAGMA table_info(attack_effectiveness)")
        existing_columns = [col[1] for col in cursor.fetchall()]
        
        # Add columns for each model if they don't exist
        for model_name in model_names:
            # Normalize model name for column names
            norm_name = normalize_model_name(model_name)
            
            # Add accuracy column if it doesn't exist
            accuracy_col = f"{norm_name}_accuracy"
            if accuracy_col not in existing_columns:
                cursor.execute(f"ALTER TABLE attack_effectiveness ADD COLUMN {accuracy_col} REAL DEFAULT 0")
            
            # Add degradation column if it doesn't exist
            degradation_col = f"{norm_name}_degradation"
            if degradation_col not in existing_columns:
                cursor.execute(f"ALTER TABLE attack_effectiveness ADD COLUMN {degradation_col} REAL DEFAULT 0")
    else:
        # Create the base table
        cursor.execute('''
        CREATE TABLE attack_effectiveness (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            attack_name TEXT NOT NULL,
            attack_category TEXT NOT NULL,
            task_name TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Add columns for each model
        for model_name in model_names:
            # Normalize model name for column names
            norm_name = normalize_model_name(model_name)
            
            # Add accuracy and degradation columns
            cursor.execute(f"ALTER TABLE attack_effectiveness ADD COLUMN {norm_name}_accuracy REAL DEFAULT 0")
            cursor.execute(f"ALTER TABLE attack_effectiveness ADD COLUMN {norm_name}_degradation REAL DEFAULT 0")
        
        # Create indexes
        cursor.execute('CREATE INDEX idx_attack_effectiveness_name ON attack_effectiveness(attack_name)')
        cursor.execute('CREATE INDEX idx_attack_effectiveness_category ON attack_effectiveness(attack_category)')
    
    # Clear existing data
    task_name = data["metadata"]["task_name"]
    cursor.execute("DELETE FROM attack_effectiveness WHERE task_name = ?", (task_name,))
    
    # Insert data for each attack type
    for attack_type in attack_types:
        # Get attack category
        attack_category = get_attack_category(attack_type)
        
        # Start building the SQL query
        columns = ["attack_name", "attack_category", "task_name"]
        values = [attack_type, attack_category, task_name]
        
        # Add data for each model
        for model_name in model_names:
            # Normalize model name for column names
            norm_name = normalize_model_name(model_name)
            
            # Get accuracy and change values
            model_data = data["models"][model_name].get(attack_type, {})
            accuracy = model_data.get("accuracy", 0)
            degradation = model_data.get("change", 0)
            
            # Add to columns and values
            columns.append(f"{norm_name}_accuracy")
            values.append(accuracy)
            columns.append(f"{norm_name}_degradation")
            values.append(degradation)
        
        # Build and execute the INSERT query
        placeholders = ", ".join(["?" for _ in values])
        query = f"INSERT INTO attack_effectiveness ({', '.join(columns)}) VALUES ({placeholders})"
        cursor.execute(query, values)
    
    conn.commit()

def store_results(data, id_maps, conn):
    """
    Store the results in the fact table.
    
    Args:
        data (dict): The loaded JSON data
        id_maps (dict): Mapping of dimension values to IDs
        conn (sqlite3.Connection): Database connection
    """
    cursor = conn.cursor()
    
    # Clear existing results for the task
    cursor.execute("DELETE FROM results WHERE task_id = ?", (id_maps['task_id'],))
    
    # Extract model names
    model_names = list(data["models"].keys())
    
    # Insert results
    for model_name in model_names:
        model_id = id_maps['model_id_map'][model_name]
        
        for attack_type, attack_data in data["models"][model_name].items():
            attack_id = id_maps['attack_id_map'][attack_type]
            
            cursor.execute(
                """
                INSERT INTO results 
                (attack_id, model_id, task_id, accuracy, accuracy_change) 
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    attack_id, 
                    model_id, 
                    id_maps['task_id'], 
                    attack_data.get("accuracy", 0), 
                    attack_data.get("change", 0)
                )
            )
    
    conn.commit()

def verify_database():
    """Verify the database was created correctly by running some test queries."""
    conn = sqlite3.connect(DB_PATH)
    
    print("\n=== Database Verification ===")
    
    # Display table structure
    print("\nTable structure:")
    cursor = conn.cursor()
    tables = ['attack_types', 'model_families', 'size_categories', 'tasks', 'models', 'results', 'attack_effectiveness']
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        print(f"\n  {table} table:")
        for col in cursor.fetchall():
            print(f"    {col[1]} ({col[2]})")
    
    # Display sample data from dimension tables
    print("\nAttack Types (first 5):")
    df_attacks = pd.read_sql_query("SELECT * FROM attack_types LIMIT 5", conn)
    print(df_attacks.to_string(index=False))
    
    print("\nModel Families:")
    df_families = pd.read_sql_query("SELECT * FROM model_families", conn)
    print(df_families.to_string(index=False))
    
    print("\nSize Categories:")
    df_sizes = pd.read_sql_query("SELECT * FROM size_categories", conn)
    print(df_sizes.to_string(index=False))
    
    # Display sample data from fact table with joins
    print("\nSample Results (first 5 rows with dimension lookups):")
    query = """
    SELECT 
        t.task_name,
        a.attack_name,
        a.attack_category,
        m.model_name,
        f.family_name,
        s.size_range,
        r.accuracy,
        r.accuracy_change
    FROM results r
    JOIN attack_types a ON r.attack_id = a.attack_id
    JOIN models m ON r.model_id = m.model_id
    JOIN tasks t ON r.task_id = t.task_id
    JOIN model_families f ON m.family_id = f.family_id
    JOIN size_categories s ON m.size_id = s.size_id
    LIMIT 5
    """
    df_results = pd.read_sql_query(query, conn)
    print(df_results.to_string(index=False))
    
    # Display sample data from the attack_effectiveness table
    print("\nAttack Effectiveness Table (first 5 rows):")
    df_effectiveness = pd.read_sql_query("SELECT * FROM attack_effectiveness LIMIT 5", conn)
    # Only show a subset of columns if there are too many
    if len(df_effectiveness.columns) > 10:
        # Show the first few columns (metadata) and a sample of model columns
        base_cols = ['id', 'attack_name', 'attack_category', 'task_name']
        model_cols = [col for col in df_effectiveness.columns if col not in base_cols and col != 'timestamp']
        # Take a sample of model columns (first model's accuracy and degradation)
        sample_model_cols = model_cols[:4] if len(model_cols) > 4 else model_cols
        display_cols = base_cols + sample_model_cols
        print(df_effectiveness[display_cols].to_string(index=False))
        print(f"... and {len(model_cols) - len(sample_model_cols)} more columns")
    else:
        print(df_effectiveness.to_string(index=False))
    
    # Calculate some statistics
    print("\nMost effective attacks (largest negative change) by model family:")
    query = """
    SELECT 
        f.family_name,
        a.attack_name,
        MIN(r.accuracy_change) as min_change
    FROM results r
    JOIN attack_types a ON r.attack_id = a.attack_id
    JOIN models m ON r.model_id = m.model_id
    JOIN model_families f ON m.family_id = f.family_id
    GROUP BY f.family_name
    """
    df_stats = pd.read_sql_query(query, conn)
    print(df_stats.to_string(index=False))
    
    # Show database size
    cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
    db_size = cursor.fetchone()[0]
    print(f"\nDatabase size: {db_size / 1024:.2f} KB")
    
    conn.close()

def main():
    """Main function to run the script."""
    print("Starting to store evaluation results in normalized database...")
    
    # Ensure the database directory exists
    ensure_db_directory()
    
    # Load results from JSON
    data = load_results_from_json()
    if not data:
        print("No data found. Exiting.")
        return
    
    # Create the database schema
    create_database()
    
    # Connect to the database
    conn = sqlite3.connect(DB_PATH)
    
    # Populate dimension tables
    print("Populating dimension tables...")
    id_maps = populate_dimension_tables(data, conn)
    
    # Store results in fact table
    print("Storing results in fact table...")
    store_results(data, id_maps, conn)
    
    # Create and populate the attack_effectiveness table
    print("Creating and populating attack effectiveness table...")
    create_attack_effectiveness_table(data, conn)
    
    # Close connection
    conn.close()
    
    print("Database population complete!")
    
    # Verify the database
    verify_database()

if __name__ == "__main__":
    main()
