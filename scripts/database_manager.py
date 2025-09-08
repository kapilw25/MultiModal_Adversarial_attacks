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
JSON_PATH = "results/robustness_chart.json"

def ensure_db_directory():
    """Ensure the directory for the database exists."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def backup_existing_files():
    """
    Backup existing robustness files before creating fresh ones.
    This ensures we don't lose previous data and always work with current pipeline models.
    """
    backup_files = []
    
    # Backup database file if it exists
    if os.path.exists(DB_PATH):
        backup_db = DB_PATH.replace('.db', '_backup.db')
        import shutil
        shutil.copy2(DB_PATH, backup_db)
        backup_files.append(backup_db)
        print(f"📁 Backed up database to {backup_db}")
    
    # Backup JSON file if it exists
    if os.path.exists(JSON_PATH):
        backup_json = JSON_PATH.replace('.json', '_backup.json')
        import shutil
        shutil.copy2(JSON_PATH, backup_json)
        backup_files.append(backup_json)
        print(f"📁 Backed up JSON to {backup_json}")
    
    return backup_files

def create_database():
    """
    Create the normalized database schema with dimension and fact tables.
    Always creates fresh tables to ensure current pipeline models and schema.
    """
    # Remove existing database file to ensure fresh start
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"🗑️  Removed existing database for fresh creation")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Always create fresh tables
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
    
    # Create performance_metrics table
    cursor.execute('''
    CREATE TABLE performance_metrics (
            metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
            attack_id INTEGER,
            model_id INTEGER,
            task_id INTEGER,
            avg_inference_time_seconds REAL DEFAULT 0,
            avg_gpu_memory_allocated_mb REAL DEFAULT 0,
            avg_gpu_memory_peak_mb REAL DEFAULT 0,
            avg_gpu_memory_reserved_mb REAL DEFAULT 0,
            total_gpu_memory_mb REAL DEFAULT 0,
            avg_cpu_memory_mb REAL DEFAULT 0,
            model_loading_time_seconds REAL DEFAULT 0,
            cache_hit_ratio REAL DEFAULT 0,
            total_questions INTEGER DEFAULT 0,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (attack_id) REFERENCES attack_types(attack_id),
            FOREIGN KEY (model_id) REFERENCES models(model_id),
            FOREIGN KEY (task_id) REFERENCES tasks(task_id)
    )
    ''')
    
    # Create attack_parameters table for adversarial attack generation metrics
    cursor.execute('''
    CREATE TABLE attack_parameters (
        param_id INTEGER PRIMARY KEY AUTOINCREMENT,
        attack_id INTEGER,
        task_id INTEGER,
        execution_id TEXT,
        image_path TEXT,
        image_name TEXT,
        execution_time_seconds REAL,
        success BOOLEAN,
        ssim REAL,
        mean_perturbation REAL,
        max_perturbation REAL,
        l2_norm REAL,
        l0_norm INTEGER,
        total_queries INTEGER,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (attack_id) REFERENCES attack_types(attack_id),
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
    cursor.execute('CREATE INDEX idx_performance_attack_id ON performance_metrics(attack_id)')
    cursor.execute('CREATE INDEX idx_performance_model_id ON performance_metrics(model_id)')
    cursor.execute('CREATE INDEX idx_performance_task_id ON performance_metrics(task_id)')
    cursor.execute('CREATE INDEX idx_attack_effectiveness_name ON attack_effectiveness(attack_name)')
    cursor.execute('CREATE INDEX idx_attack_effectiveness_category ON attack_effectiveness(attack_category)')
    cursor.execute('CREATE INDEX idx_attack_params_attack_id ON attack_parameters(attack_id)')
    cursor.execute('CREATE INDEX idx_attack_params_task_id ON attack_parameters(task_id)')
    cursor.execute('CREATE INDEX idx_attack_params_execution_id ON attack_parameters(execution_id)')
    cursor.execute('CREATE INDEX idx_attack_params_success ON attack_parameters(success)')
        
    print("✅ Created fresh normalized database schema with dimension and fact tables")
    
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
        str: Either 'White-Box' or 'Black-Box' or 'Original' for no attack
    """
    # Define white-box attacks (formerly transfer-based)
    whitebox_attacks = [
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
    
    # Check if it's a white-box attack
    for attack in whitebox_attacks:
        if attack.lower() == attack_type.lower():
            return 'White-Box'
    
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
    Determine the size category based on the model name using official HuggingFace model sizes.
    
    Args:
        model_name (str): The name of the model
        
    Returns:
        str: The size category
    """
    model_name_lower = model_name.lower()
    
    # Official model sizes and their correct size categories (based on actual parameter counts)
    # Use exact model name matching to avoid suffix collision issues
    exact_model_mapping = {
        # Qwen models
        'qwen25_vl_3b': '(3-4]B',      # 3.75B params
        'qwen25_vl_7b': '(8-9]B',      # 8.29B params
        'qwen2_vl_2b': '(2-3]B',       # 2.21B params
        
        # Google models
        'gemma3_vl_4b': '(4-5]B',      # 4.3B params
        'paligemma_vl_3b': '(2-3]B',   # 2.92B params
        
        # DeepSeek models
        'deepseek1_vl_1pt3b': '(1-2]B', # 1.98B params
        'deepseek1_vl_7b': '(7-8]B',    # 7.34B params
        
        # SmolVLM2 models
        'smolvlm2_pt25b': '(0-1]B',     # 0.256B params
        'smolvlm2_pt5b': '(0-1]B',      # 0.507B params
        'smolvlm2_2pt2b': '(2-3]B',     # 2.25B params
        
        # InternVL models
        'internvl3_1b': '(0-1]B',       # 0.938B params
        'internvl3_2b': '(2-3]B',       # 2.09B params
        'internvl25_4b': '(3-4]B',      # 3.71B params
        
        # Other models
        'moondream2_2b': '(1-2]B',      # 1.93B params
        'llava_1pt5_7b': '(7-8]B',      # 7.06B params
        'llava_v1pt6_mistral_7b': '(7-8]B', # 7.57B params
        
        # Legacy models
        'florence2_pt23b': '(0-1]B',
        'florence2_pt77b': '(0-1]B',
        'phi3pt5_vision_4b': '(4-5]B',  # 4.15B params
        'gpt4o': 'Cloud API',
        'glmedge_2b': '(1-2]B'
    }
    
    # First try exact model name matching (most accurate)
    if model_name_lower in exact_model_mapping:
        return exact_model_mapping[model_name_lower]
    
    # Fallback: Legacy suffix-based mapping (only for unknown models)
    legacy_size_mapping = {
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
    
    # Only use suffix matching for unknown models
    for size_key, size_category in legacy_size_mapping.items():
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

def store_performance_metrics(data, id_maps, conn):
    """
    Store performance metrics in the performance_metrics table.
    
    Args:
        data (dict): The loaded JSON data
        id_maps (dict): Mapping of dimension values to IDs
        conn (sqlite3.Connection): Database connection
    """
    cursor = conn.cursor()
    
    # Clear existing performance metrics for the task
    cursor.execute("DELETE FROM performance_metrics WHERE task_id = ?", (id_maps['task_id'],))
    
    # Extract model names
    model_names = list(data["models"].keys())
    
    # Insert performance metrics
    for model_name in model_names:
        model_id = id_maps['model_id_map'][model_name]
        
        for attack_type, attack_data in data["models"][model_name].items():
            attack_id = id_maps['attack_id_map'][attack_type]
            
            # Check if performance metrics exist for this attack
            if 'performance_metrics' in attack_data:
                pm = attack_data['performance_metrics']
                
                cursor.execute(
                    """
                    INSERT INTO performance_metrics 
                    (attack_id, model_id, task_id, avg_inference_time_seconds, 
                     avg_gpu_memory_allocated_mb, avg_gpu_memory_peak_mb, avg_gpu_memory_reserved_mb,
                     total_gpu_memory_mb, avg_cpu_memory_mb, model_loading_time_seconds, 
                     cache_hit_ratio, total_questions) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        attack_id, 
                        model_id, 
                        id_maps['task_id'], 
                        pm.get("avg_inference_time_seconds", 0),
                        pm.get("avg_gpu_memory_allocated_mb", 0),
                        pm.get("avg_gpu_memory_peak_mb", 0),
                        pm.get("avg_gpu_memory_reserved_mb", 0),
                        pm.get("total_gpu_memory_mb", 0),
                        pm.get("avg_cpu_memory_mb", 0),
                        pm.get("model_loading_time_seconds", 0),
                        pm.get("cache_hit_ratio", 0),
                        pm.get("total_questions", 0)
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
    tables = ['attack_types', 'model_families', 'size_categories', 'tasks', 'models', 'results', 'performance_metrics', 'attack_parameters', 'attack_effectiveness']
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
    
    # Display attack parameters data if available
    cursor.execute("SELECT COUNT(*) FROM attack_parameters")
    attack_param_count = cursor.fetchone()[0]
    
    if attack_param_count > 0:
        print(f"\nAttack Parameters ({attack_param_count} records):")
        print("Sample Attack Parameters (first 5 rows):")
        df_params = pd.read_sql_query("""
            SELECT 
                at.attack_name,
                t.task_name,
                ap.image_name,
                ap.execution_time_seconds,
                ap.success,
                ap.ssim,
                ap.mean_perturbation,
                ap.max_perturbation
            FROM attack_parameters ap
            JOIN attack_types at ON ap.attack_id = at.attack_id
            JOIN tasks t ON ap.task_id = t.task_id
            LIMIT 5
        """, conn)
        print(df_params.to_string(index=False))
        
        # Show attack parameter summary if view exists
        try:
            print("\nAttack Parameter Summary (by attack type and task):")
            df_param_summary = pd.read_sql_query("""
                SELECT 
                    attack_name,
                    task_name,
                    total_executions,
                    ROUND(success_rate, 3) as success_rate,
                    ROUND(avg_ssim, 4) as avg_ssim,
                    ROUND(avg_mean_perturbation, 2) as avg_mean_pert,
                    ROUND(avg_execution_time, 2) as avg_exec_time
                FROM attack_param_summary
                ORDER BY attack_category, attack_name, task_name
                LIMIT 10
            """, conn)
            print(df_param_summary.to_string(index=False))
            if len(df_param_summary) >= 10:
                print("... (showing first 10 rows)")
        except Exception as e:
            print(f"⚠️  Could not query attack parameter summary view: {e}")
    else:
        print("\nNo attack parameters data found in database.")
    
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

def normalize_attack_name_for_mapping(json_name, db_attack_names):
    """
    Normalize attack names to handle variations between JSON and database
    
    Args:
        json_name (str): Attack name from JSON file
        db_attack_names (list): List of attack names from database
        
    Returns:
        str: Matching database attack name or None if not found
    """
    # Direct match first
    if json_name in db_attack_names:
        return json_name
    
    # Handle specific name variations
    name_mappings = {
        "CW-Linf": "CW-L∞",
        "Query-Efficient-BB": "Query-Efficient BB",
        "CW-L_inf": "CW-L∞",
        "CW-Linfinity": "CW-L∞"
    }
    
    # Check mappings
    if json_name in name_mappings:
        mapped_name = name_mappings[json_name]
        if mapped_name in db_attack_names:
            return mapped_name
    
    # Try case-insensitive match
    for db_name in db_attack_names:
        if json_name.lower() == db_name.lower():
            return db_name
    
    # Try fuzzy matching (remove special characters and spaces)
    import re
    def clean_name(name):
        return re.sub(r'[^a-zA-Z0-9]', '', name.lower())
    
    json_clean = clean_name(json_name)
    for db_name in db_attack_names:
        if json_clean == clean_name(db_name):
            return db_name
    
    return None

def integrate_attack_parameters(attack_params_path, conn):
    """
    Integrate attack parameter data into the database
    
    Args:
        attack_params_path (str): Path to attack_parameters.json
        conn (sqlite3.Connection): Database connection
    """
    print(f"🔗 Integrating attack parameters from {attack_params_path}...")
    
    # Check if file exists
    if not os.path.exists(attack_params_path):
        print(f"⚠️  Attack parameters file not found: {attack_params_path}")
        return
    
    try:
        with open(attack_params_path, 'r') as f:
            attack_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON file: {e}")
        return
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return

    cursor = conn.cursor()

    # Get existing ID mappings
    attack_id_map = {}
    cursor.execute("SELECT attack_name, attack_id FROM attack_types")
    for name, aid in cursor.fetchall():
        attack_id_map[name] = aid

    # Get list of database attack names for normalization
    db_attack_names = list(attack_id_map.keys())

    task_id_map = {}
    cursor.execute("SELECT task_name, task_id FROM tasks")
    for name, tid in cursor.fetchall():
        task_id_map[name] = tid

    # Clear existing attack parameters data
    cursor.execute("DELETE FROM attack_parameters")
    print("🗑️  Cleared existing attack parameters data")

    inserted_records = 0
    skipped_records = 0

    # Process each attack's executions
    for attack_name, attack_info in attack_data["attacks"].items():
        # Try to normalize the attack name
        normalized_attack_name = normalize_attack_name_for_mapping(attack_name, db_attack_names)
        
        if normalized_attack_name:
            attack_id = attack_id_map[normalized_attack_name]
            if attack_name != normalized_attack_name:
                print(f"📝 Mapped '{attack_name}' → '{normalized_attack_name}'")
        else:
            print(f"⚠️  Attack '{attack_name}' not found in attack_types table, skipping...")
            skipped_records += len(attack_info.get("executions", []))
            continue

        for execution in attack_info["executions"]:
            # Extract task from image path
            try:
                path_parts = execution["image_path"].split('/')
                if len(path_parts) >= 3:
                    task_type = path_parts[2]  # data/clean/TASK/...
                else:
                    print(f"⚠️  Invalid image path format: {execution['image_path']}")
                    skipped_records += 1
                    continue
                    
                task_id = task_id_map.get(task_type)
                
                if not task_id:
                    print(f"⚠️  Task '{task_type}' not found in tasks table, skipping...")
                    skipped_records += 1
                    continue

                # Insert parameter record
                cursor.execute("""
                    INSERT INTO attack_parameters (
                        attack_id, task_id, execution_id, image_path, image_name,
                        execution_time_seconds, success, ssim, mean_perturbation, 
                        max_perturbation, l2_norm, l0_norm, total_queries, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    attack_id, task_id, execution["execution_id"],
                    execution["image_path"], execution["image_name"],
                    execution["execution_time_seconds"], execution["success"],
                    execution["parameters"]["ssim"],
                    execution["parameters"]["mean_perturbation"],
                    execution["parameters"]["max_perturbation"], 
                    execution["parameters"]["l2_norm"],
                    execution["parameters"]["l0_norm"],
                    execution["parameters"]["total_queries"],
                    execution["timestamp"]
                ))
                inserted_records += 1
                
            except KeyError as e:
                print(f"⚠️  Missing required field {e} in execution record, skipping...")
                skipped_records += 1
                continue
            except Exception as e:
                print(f"⚠️  Error processing execution record: {e}, skipping...")
                skipped_records += 1
                continue

    conn.commit()
    print(f"✅ Attack parameters integration complete: {inserted_records} records inserted, {skipped_records} skipped")

def create_attack_parameter_views(conn):
    """
    Create aggregate views for attack parameter analysis
    
    Args:
        conn (sqlite3.Connection): Database connection
    """
    cursor = conn.cursor()
    
    # Create the attack parameter summary view
    cursor.execute("""
    CREATE VIEW IF NOT EXISTS attack_param_summary AS
    SELECT 
        at.attack_name,
        at.attack_category,
        t.task_name,
        COUNT(*) as total_executions,
        AVG(CASE WHEN ap.success THEN 1.0 ELSE 0.0 END) as success_rate,
        AVG(ap.ssim) as avg_ssim,
        AVG(ap.mean_perturbation) as avg_mean_perturbation,
        AVG(ap.max_perturbation) as avg_max_perturbation,
        AVG(ap.execution_time_seconds) as avg_execution_time,
        AVG(ap.l2_norm) as avg_l2_norm,
        AVG(ap.l0_norm) as avg_l0_norm,
        AVG(ap.total_queries) as avg_total_queries,
        MIN(ap.ssim) as min_ssim,
        MAX(ap.ssim) as max_ssim,
        (MAX(ap.mean_perturbation) - MIN(ap.mean_perturbation)) as range_mean_perturbation
    FROM attack_parameters ap
    JOIN attack_types at ON ap.attack_id = at.attack_id  
    JOIN tasks t ON ap.task_id = t.task_id
    GROUP BY at.attack_id, ap.task_id
    """)
    
    # Create a comprehensive analysis view
    cursor.execute("""
    CREATE VIEW IF NOT EXISTS attack_comprehensive_analysis AS
    SELECT 
        at.attack_name,
        at.attack_category,
        t.task_name,
        -- Attack parameters
        COUNT(ap.param_id) as param_executions,
        AVG(CASE WHEN ap.success THEN 1.0 ELSE 0.0 END) as param_success_rate,
        AVG(ap.ssim) as avg_ssim,
        AVG(ap.mean_perturbation) as avg_mean_perturbation,
        AVG(ap.execution_time_seconds) as avg_attack_time,
        -- VLM robustness results (if available)
        COUNT(r.result_id) as vlm_evaluations,
        AVG(r.accuracy) as avg_accuracy,
        AVG(r.accuracy_change) as avg_accuracy_change,
        -- Performance metrics (if available)
        AVG(pm.avg_inference_time_seconds) as avg_inference_time,
        AVG(pm.avg_gpu_memory_allocated_mb) as avg_gpu_memory_mb
    FROM attack_types at
    LEFT JOIN attack_parameters ap ON at.attack_id = ap.attack_id
    LEFT JOIN tasks t ON ap.task_id = t.task_id
    LEFT JOIN results r ON at.attack_id = r.attack_id AND ap.task_id = r.task_id
    LEFT JOIN performance_metrics pm ON at.attack_id = pm.attack_id AND ap.task_id = pm.task_id
    GROUP BY at.attack_id, t.task_id
    HAVING param_executions > 0 OR vlm_evaluations > 0
    """)
    
    conn.commit()
    print("✅ Created attack parameter analysis views")

def main():
    """Main function to run the script."""
    print("🚀 Starting to store evaluation results in normalized database...")
    
    # Ensure the database directory exists
    ensure_db_directory()
    
    # Backup existing files before creating fresh ones
    backup_files = backup_existing_files()
    if backup_files:
        print(f"✅ Created {len(backup_files)} backup files")
    
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
    
    # Store performance metrics
    print("Storing performance metrics...")
    store_performance_metrics(data, id_maps, conn)
    
    # Create and populate the attack_effectiveness table
    print("Creating and populating attack effectiveness table...")
    create_attack_effectiveness_table(data, conn)
    
    # Integrate attack parameters if available
    attack_params_path = "results/attack_parameters.json"
    if os.path.exists(attack_params_path):
        integrate_attack_parameters(attack_params_path, conn)
        create_attack_parameter_views(conn)
    else:
        print(f"⚠️  Attack parameters file not found at {attack_params_path}, skipping integration...")
    
    # Close connection
    conn.close()
    
    print("Database population complete!")
    
    # Verify the database
    verify_database()

def integrate_attack_parameters_standalone(attack_params_path="results/attack_parameters.json"):
    """
    Standalone function to integrate attack parameters into an existing database
    
    Args:
        attack_params_path (str): Path to attack_parameters.json file
    """
    print("🔗 Standalone attack parameters integration...")
    
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at {DB_PATH}. Please run main() first to create the database.")
        return
    
    conn = sqlite3.connect(DB_PATH)
    
    try:
        # Integrate attack parameters
        integrate_attack_parameters(attack_params_path, conn)
        create_attack_parameter_views(conn)
        
        print("✅ Standalone attack parameters integration complete!")
        
        # Quick verification
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM attack_parameters")
        count = cursor.fetchone()[0]
        print(f"📊 Total attack parameter records: {count}")
        
    except Exception as e:
        print(f"❌ Error during integration: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    import sys
    
    # Check if user wants to run standalone attack parameter integration
    if len(sys.argv) > 1 and sys.argv[1] == "--integrate-attack-params":
        attack_params_path = sys.argv[2] if len(sys.argv) > 2 else "results/attack_parameters.json"
        integrate_attack_parameters_standalone(attack_params_path)
    else:
        main()
