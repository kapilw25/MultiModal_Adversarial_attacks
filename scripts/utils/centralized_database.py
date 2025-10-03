#!/usr/bin/env python3
"""
Centralized Database Manager for VLM Adversarial Attack Pipeline
Replaces all JSON files with direct SQLite database operations
"""

import sqlite3
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

DB_PATH = "results/centralized_pipeline.db"

def create_centralized_schema():
    """Create comprehensive database schema for all pipeline data"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Attack Executions (replaces attack_parameters.json) - EPSILON ONLY
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS attack_executions (
        execution_id TEXT PRIMARY KEY,
        attack_name TEXT NOT NULL,
        attack_category TEXT NOT NULL,
        image_path TEXT NOT NULL,
        adversarial_image_path TEXT NOT NULL,
        task_type TEXT NOT NULL,
        execution_time_seconds INTEGER NOT NULL,
        success BOOLEAN NOT NULL,
        timestamp TEXT NOT NULL,
        -- Epsilon Parameters (Core metrics only)
        epsilon_level TEXT,
        epsilon_target REAL,
        epsilon_l_inf REAL
    )
    ''')

    # 2. Ground Truth Questions (replaces ground_truth JSON files)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ground_truth_questions (
        question_id TEXT PRIMARY KEY,
        image_path TEXT NOT NULL,
        question_text TEXT NOT NULL,
        answer TEXT NOT NULL,
        question_type TEXT NOT NULL,
        markers TEXT  -- JSON string for array
    )
    ''')

    # 3. Processed Images (replaces processed_images.json)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS processed_images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_type TEXT NOT NULL,
        image_path TEXT NOT NULL,
        UNIQUE(task_type, image_path)
    )
    ''')

    # 3.1. Image Registry (ML research best practice: separate image metadata)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS image_registry (
        image_id TEXT PRIMARY KEY,
        task_type TEXT NOT NULL,
        image_path TEXT NOT NULL,
        filename TEXT NOT NULL,
        is_processed BOOLEAN DEFAULT 0,
        creation_timestamp TEXT,
        UNIQUE(task_type, image_path)
    )
    ''')

    # 4. Model Inference Results (replaces inference JSON files) - EPSILON ONLY
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS inference_results (
        result_id TEXT PRIMARY KEY,
        question_id TEXT NOT NULL,
        prompt TEXT NOT NULL,
        model_response TEXT NOT NULL,
        ground_truth TEXT NOT NULL,
        question_type TEXT NOT NULL,
        answer_id TEXT,
        markers TEXT,  -- JSON string
        model_id TEXT NOT NULL,
        -- Metadata (EPSILON ONLY - SSIM removed)
        adversarial BOOLEAN NOT NULL,
        task TEXT NOT NULL,
        attack_type TEXT NOT NULL,
        epsilon_target REAL NOT NULL,        -- Target epsilon value (goal)
        epsilon_l_inf REAL NOT NULL,        -- Actual achieved epsilon value
        inference_image_path TEXT NOT NULL,  -- Image fed to model (clean or adversarial)
        clean_image_path TEXT NOT NULL,      -- Original unperturbed reference
        timestamp TEXT NOT NULL,
        -- Performance Metrics
        inference_time_seconds REAL,
        gpu_before_mb REAL,
        gpu_after_mb REAL,
        gpu_peak_mb REAL,
        gpu_reserved_mb REAL,
        total_gpu_mb REAL,
        cpu_before_mb REAL,
        cpu_after_mb REAL,
        was_cached BOOLEAN,
        loading_time_seconds REAL,
        batch_size INTEGER,
        position_in_batch INTEGER,
        total_batch_questions INTEGER
    )
    ''')

    # 5. Robustness Evaluation (replaces robustness JSON files)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS robustness_evaluation (
        eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL,
        attack_type TEXT NOT NULL,
        task_name TEXT NOT NULL,
        accuracy REAL NOT NULL,
        accuracy_change REAL NOT NULL,
        effect TEXT NOT NULL,
        -- Aggregated Performance Metrics
        avg_inference_time_seconds REAL,
        avg_gpu_memory_allocated_mb REAL,
        avg_gpu_memory_peak_mb REAL,
        avg_gpu_memory_reserved_mb REAL,
        total_gpu_memory_mb REAL,
        avg_cpu_memory_mb REAL,
        model_loading_time_seconds REAL,
        cache_hit_ratio REAL,
        total_questions INTEGER,
        evaluation_timestamp TEXT NOT NULL,
        version TEXT DEFAULT '2.0',
        UNIQUE(model_name, attack_type, task_name)
    )
    ''')

    # Create indexes for better performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_attack_executions_attack_name ON attack_executions(attack_name)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_attack_executions_task_type ON attack_executions(task_type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_ground_truth_questions_type ON ground_truth_questions(question_type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_processed_images_task_type ON processed_images(task_type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_inference_results_model_id ON inference_results(model_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_inference_results_attack_type ON inference_results(attack_type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_robustness_evaluation_model_name ON robustness_evaluation(model_name)')

    conn.commit()
    conn.close()
    print(f"✅ Created centralized database schema at: {DB_PATH}")

# Database operation functions
class CentralizedDB:
    def __init__(self):
        self.db_path = DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        create_centralized_schema()

    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)

    def save_attack_execution(self, execution_data: Dict[str, Any]):
        """Save attack execution data (replaces JSON append)"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Verify table exists before attempting insert
        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='attack_executions'")
            table_exists = cursor.fetchone()

            if not table_exists:
                print("⚠️  attack_executions table not found, recreating schema...")
                conn.close()
                create_centralized_schema()
                conn = self.get_connection()
                cursor = conn.cursor()

            cursor.execute('''
            INSERT OR REPLACE INTO attack_executions VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            ''', (
                execution_data['execution_id'],
                execution_data['attack_name'],
                execution_data['attack_category'],
                execution_data['image_path'],
                execution_data['adversarial_image_path'],
                execution_data['task_type'],
                execution_data['execution_time_seconds'],
                execution_data['success'],
                execution_data['timestamp'],
                execution_data['parameters']['epsilon_level'],
                execution_data['parameters']['epsilon_target'],
                execution_data['parameters']['epsilon_l_inf']
            ))

            conn.commit()

        except Exception as e:
            print(f"❌ Database operation failed: {e}")
            print(f"🔍 Attempting to recreate schema and retry...")
            try:
                conn.close()
                create_centralized_schema()
                conn = self.get_connection()
                cursor = conn.cursor()

                cursor.execute('''
                INSERT OR REPLACE INTO attack_executions VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                ''', (
                    execution_data['execution_id'],
                    execution_data['attack_name'],
                    execution_data['attack_category'],
                    execution_data['image_path'],
                    execution_data['adversarial_image_path'],
                    execution_data['task_type'],
                    execution_data['execution_time_seconds'],
                    execution_data['success'],
                    execution_data['timestamp'],
                    execution_data['parameters']['epsilon_level'],
                    execution_data['parameters']['epsilon_target'],
                    execution_data['parameters']['epsilon_l_inf']
                ))

                conn.commit()
                print("✅ Retry successful after schema recreation")

            except Exception as retry_error:
                print(f"❌ Retry also failed: {retry_error}")
                raise
        finally:
            conn.close()

    def save_ground_truth_question(self, question_data: Dict[str, Any]):
        """Save ground truth question (replaces ground truth JSON files)"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
        INSERT OR REPLACE INTO ground_truth_questions VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            question_data['question_id'],
            question_data['image'],
            question_data['text'],
            question_data['answer'],
            question_data['type'],
            json.dumps(question_data['markers'])
        ))

        conn.commit()
        conn.close()

    def save_processed_image(self, task_type: str, image_path: str):
        """Save processed image entry (replaces processed_images.json)"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
        INSERT OR IGNORE INTO processed_images (task_type, image_path) VALUES (?, ?)
        ''', (task_type, image_path))

        conn.commit()
        conn.close()

    def save_inference_result(self, result_data: Dict[str, Any]):
        """Save model inference result (replaces JSON file writing)"""
        conn = self.get_connection()
        cursor = conn.cursor()

        pm = result_data.get('performance_metrics', {})
        gpu = pm.get('gpu_memory', {})
        cpu = pm.get('cpu_memory', {})
        loading = pm.get('model_loading', {})
        batch = pm.get('batch_info', {})

        result_id = f"{result_data['question_id']}_{result_data['model_id']}_{result_data['metadata']['attack_type']}"

        cursor.execute('''
        INSERT OR REPLACE INTO inference_results VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        ''', (
            result_id,
            result_data['question_id'],
            result_data['prompt'],
            result_data['text'],
            result_data['truth'],
            result_data['type'],
            result_data['answer_id'],
            json.dumps(result_data['markers']),
            result_data['model_id'],
            result_data['metadata']['adversarial'],
            result_data['metadata']['task'],
            result_data['metadata']['attack_type'],
            result_data['metadata'].get('epsilon_target', 0.0),
            result_data['metadata'].get('epsilon_l_inf', 0.0),
            result_data['metadata']['image_path'],
            result_data['metadata']['original_image_path'],
            result_data['metadata']['timestamp'],
            pm.get('inference_time_seconds'),
            gpu.get('before_inference_mb'),
            gpu.get('after_inference_mb'),
            gpu.get('peak_mb'),
            gpu.get('reserved_mb'),
            gpu.get('total_gpu_mb'),
            cpu.get('before_inference_mb'),
            cpu.get('after_inference_mb'),
            loading.get('was_cached'),
            loading.get('loading_time_seconds'),
            batch.get('batch_size'),
            batch.get('position_in_batch'),
            batch.get('total_batch_questions')
        ))

        conn.commit()
        conn.close()

    def insert_attack_result(self, result_data: Dict[str, Any]):
        """Insert attack result (compatible interface for attack runners)"""

        # CRITICAL FIX: Generate execution_id from adversarial_image_path to ensure uniqueness
        # Same path → Same execution_id → INSERT OR REPLACE works correctly
        adversarial_image_path = result_data.get('adversarial_image_path', '')

        if adversarial_image_path:
            # Derive execution_id from path to ensure deterministic replacement
            # Example: data/adversarial/whitebox/fgsm/eps_0016/chart/image.png
            #       → data_adversarial_whitebox_fgsm_eps_0016_chart_image
            execution_id = adversarial_image_path.replace('/', '_').replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
        else:
            # Fallback to timestamp only if no adversarial path (shouldn't happen in normal flow)
            execution_id = result_data.get('execution_id', f"attack_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}")

        # Convert to save_attack_execution format
        execution_data = {
            'execution_id': execution_id,
            'attack_name': result_data.get('attack_type', ''),
            'attack_category': result_data.get('attack_category', ''),
            'image_path': result_data.get('image_path', ''),
            'adversarial_image_path': adversarial_image_path,
            'task_type': result_data.get('task_type', ''),
            'execution_time_seconds': result_data.get('execution_time', 0),
            'success': result_data.get('success', False),
            'timestamp': result_data.get('timestamp', datetime.now().isoformat()),
            'parameters': {
                'epsilon_level': result_data.get('epsilon_level', ''),
                'epsilon_target': result_data.get('epsilon_target', 0.0),
                'epsilon_l_inf': result_data.get('epsilon_achieved', 0.0)
            }
        }

        # Use existing save_attack_execution method
        self.save_attack_execution(execution_data)

    def save_robustness_evaluation(self, model_name: str, attack_type: str, task_name: str, eval_data: Dict[str, Any]):
        """Save robustness evaluation (replaces robustness JSON)"""
        conn = self.get_connection()
        cursor = conn.cursor()

        pm = eval_data.get('performance_metrics', {})

        cursor.execute('''
        INSERT OR REPLACE INTO robustness_evaluation VALUES (
            NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        ''', (
            model_name, attack_type, task_name,
            eval_data['accuracy'], eval_data['change'], eval_data['effect'],
            pm.get('avg_inference_time_seconds'),
            pm.get('avg_gpu_memory_allocated_mb'),
            pm.get('avg_gpu_memory_peak_mb'),
            pm.get('avg_gpu_memory_reserved_mb'),
            pm.get('total_gpu_memory_mb'),
            pm.get('avg_cpu_memory_mb'),
            pm.get('model_loading_time_seconds'),
            pm.get('cache_hit_ratio'),
            pm.get('total_questions'),
            datetime.now().isoformat(),
            '2.0'
        ))

        conn.commit()
        conn.close()

    def get_attack_parameters(self) -> Dict[str, Any]:
        """Get attack parameters data with epsilon tracking (replaces reading attack_parameters.json)"""
        conn = self.get_connection()
        cursor = conn.cursor()

        query = '''
        SELECT image_path, adversarial_image_path, epsilon_l_inf, epsilon_target, epsilon_level, attack_name, attack_category
        FROM attack_executions
        WHERE success = 1
        '''

        results = cursor.execute(query).fetchall()
        conn.close()

        # Create mapping compatible with existing code
        epsilon_mapping = {}
        for row in results:
            composite_key = (row[0], row[1])  # (image_path, adversarial_image_path)
            epsilon_mapping[composite_key] = {
                'epsilon_l_inf': row[2],
                'epsilon_target': row[3],
                'epsilon_level': row[4],
                'adversarial_image_path': row[1],
                'attack_type': row[5],
                'attack_category': row[6]
            }

        return epsilon_mapping

    def generate_industry_standard_id(self, model_name: str, adversarial_image_path: str, question_id: str) -> str:
        """Generate industry-standard deterministic result_id using content-based hashing

        Args:
            model_name: Name of the model (e.g., "Qwen25_VL_3B")
            adversarial_image_path: Path to adversarial image (e.g., "data/adversarial/whitebox/fgsm/eps_050/chart/20231107140031466140.png")
            question_id: Question identifier for uniqueness

        Returns:
            12-character deterministic hash (industry standard)

        Example:
            SAME INPUT = SAME HASH = SAME ID = REPLACEMENT
            - Qwen25_VL_3B + fgsm/085/chart/image1.png + q001 → a1b2c3d4e5f6
            - Re-run same combo → Same hash → Replaces record
        """
        import hashlib

        # Create deterministic content hash
        content = f"{model_name}_{adversarial_image_path}_{question_id}"
        hash_obj = hashlib.sha256(content.encode())

        # Take first 12 characters (industry standard for short IDs)
        return hash_obj.hexdigest()[:12]

    def save_simple_inference_result(self, inference_data: Dict[str, Any]):
        """Save simple inference result for model_inference.py compatibility"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Extract data for industry-standard ID generation
        model_name = inference_data.get('model_name', 'unknown')
        adversarial_image_path = inference_data.get('image_path', '')
        # Use evaluation_id (path-based format) as PRIMARY KEY result_id
        result_id = inference_data.get('evaluation_id', f"q_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}")

        # Extract question identifier for question_id field
        question_id = result_id.split('_')[-1] if '_q' in result_id else result_id

        cursor.execute('''
        INSERT OR REPLACE INTO inference_results (
            result_id, question_id, prompt, model_response, ground_truth, question_type, answer_id, markers,
            model_id, adversarial, task, attack_type,
            epsilon_target, epsilon_l_inf, inference_image_path, clean_image_path, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result_id,
            question_id,                               # question_id (for traceability)
            inference_data.get('question', ''),        # prompt
            inference_data.get('model_response', ''),  # model_response
            inference_data.get('ground_truth', ''),    # ground_truth
            inference_data.get('task_type', ''),       # question_type
            '',                                        # answer_id
            '[]',                                      # markers (empty JSON array)
            model_name,                                # model_id
            inference_data.get('adversarial', False),  # adversarial
            inference_data.get('task_type', ''),       # task
            inference_data.get('attack_type', ''),     # attack_type
            inference_data.get('epsilon_target', 1.0), # epsilon_target (goal)
            inference_data.get('epsilon_l_inf', 0.0),  # epsilon_l_inf (achieved)
            adversarial_image_path,                    # inference_image_path (full path for inference)
            inference_data.get('clean_image_path', ''), # clean_image_path (original unperturbed reference)
            inference_data.get('timestamp', datetime.now().isoformat() + 'Z')  # timestamp
        ))

        conn.commit()
        conn.close()

    def get_ground_truth_questions(self, task_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get ground truth questions (replaces reading ground truth JSON files)"""
        conn = self.get_connection()
        cursor = conn.cursor()

        if task_type:
            query = '''
            SELECT question_id, image_path, question_text, answer, question_type, markers
            FROM ground_truth_questions
            WHERE question_type = ?
            '''
            results = cursor.execute(query, (task_type,)).fetchall()
        else:
            query = '''
            SELECT question_id, image_path, question_text, answer, question_type, markers
            FROM ground_truth_questions
            '''
            results = cursor.execute(query).fetchall()

        conn.close()

        questions = []
        for row in results:
            questions.append({
                'question_id': row[0],
                'image': row[1],
                'text': row[2],
                'answer': row[3],
                'type': row[4],
                'markers': json.loads(row[5])
            })

        return questions

    def get_processed_images(self) -> Dict[str, List[str]]:
        """Get processed images data (replaces reading processed_images.json)"""
        conn = self.get_connection()
        cursor = conn.cursor()

        query = '''
        SELECT task_type, image_path
        FROM processed_images
        ORDER BY task_type, image_path
        '''

        results = cursor.execute(query).fetchall()
        conn.close()

        processed_images = {}
        for row in results:
            task_type, image_path = row
            if task_type not in processed_images:
                processed_images[task_type] = []
            processed_images[task_type].append(image_path)

        return processed_images

    def get_inference_results(self, model_id: Optional[str] = None, task: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get inference results (replaces reading inference JSON files)"""
        conn = self.get_connection()
        cursor = conn.cursor()

        query = '''
        SELECT question_id, prompt, model_response, ground_truth, question_type,
               attack_type, adversarial, epsilon_target, epsilon_l_inf, inference_image_path, clean_image_path,
               inference_time_seconds, gpu_peak_mb, was_cached, loading_time_seconds
        FROM inference_results
        '''

        params = []
        conditions = []

        if model_id:
            conditions.append("model_id = ?")
            params.append(model_id)

        if task:
            conditions.append("task = ?")
            params.append(task)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        results = cursor.execute(query, params).fetchall()
        conn.close()

        inference_results = []
        for row in results:
            inference_results.append({
                'question_id': row[0],
                'prompt': row[1],
                'text': row[2],
                'truth': row[3],
                'type': row[4],
                'attack_type': row[5],
                'adversarial': row[6],
                'epsilon_target': row[7],
                'epsilon_l_inf': row[8],
                'inference_image_path': row[9],
                'clean_image_path': row[10],
                'inference_time_seconds': row[11],
                'gpu_peak_mb': row[12],
                'was_cached': row[13],
                'loading_time_seconds': row[14]
            })

        return inference_results

    def get_robustness_evaluations(self, model_name: Optional[str] = None, task_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get robustness evaluations (replaces reading robustness JSON files)"""
        conn = self.get_connection()
        cursor = conn.cursor()

        query = '''
        SELECT model_name, attack_type, task_name, accuracy, accuracy_change, effect,
               avg_inference_time_seconds, avg_gpu_memory_peak_mb, cache_hit_ratio,
               total_questions, evaluation_timestamp
        FROM robustness_evaluation
        '''

        params = []
        conditions = []

        if model_name:
            conditions.append("model_name = ?")
            params.append(model_name)

        if task_name:
            conditions.append("task_name = ?")
            params.append(task_name)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        results = cursor.execute(query, params).fetchall()
        conn.close()

        evaluations = []
        for row in results:
            evaluations.append({
                'model_name': row[0],
                'attack_type': row[1],
                'task_name': row[2],
                'accuracy': row[3],
                'accuracy_change': row[4],
                'effect': row[5],
                'avg_inference_time_seconds': row[6],
                'avg_gpu_memory_peak_mb': row[7],
                'cache_hit_ratio': row[8],
                'total_questions': row[9],
                'evaluation_timestamp': row[10]
            })

        return evaluations

    def import_existing_json_data(self):
        """Import existing JSON data into the centralized database"""
        print("🔄 Importing existing JSON data into centralized database...")

        # Import attack parameters
        attack_params_file = "results/attack_parameters.json"
        if os.path.exists(attack_params_file):
            with open(attack_params_file, 'r') as f:
                attack_data = json.load(f)

            for attack_name, attack_info in attack_data.get('attacks', {}).items():
                for execution in attack_info.get('executions', []):
                    # Extract image name from path for execution_id
                    image_name = os.path.basename(execution.get('image_path', 'unknown'))
                    execution_data = {
                        'execution_id': f"{execution.get('execution_id', 'unknown')}_{attack_name}_{image_name}",
                        'attack_name': attack_name,
                        'attack_category': attack_info.get('attack_category', 'Unknown'),
                        'image_path': execution.get('image_path', ''),
                        'adversarial_image_path': execution.get('adversarial_image_path', ''),
                        'task_type': execution.get('task_type', ''),
                        'execution_time_seconds': execution.get('execution_time_seconds', 0),
                        'success': execution.get('success', False),
                        'timestamp': execution.get('timestamp', ''),
                        'parameters': execution.get('parameters', {})
                    }
                    self.save_attack_execution(execution_data)

            print(f"✅ Imported attack parameters data")

        # Import ground truth questions
        ground_truth_files = [
            'results/ground_truth/eval_chart.json',
            'results/ground_truth/eval_table.json',
            'results/ground_truth/eval_road_map.json',
            'results/ground_truth/eval_dashboard.json',
            'results/ground_truth/eval_flowchart.json',
            'results/ground_truth/eval_relation_graph.json',
            'results/ground_truth/eval_planar_layout.json',
            'results/ground_truth/eval_visual_puzzle.json'
        ]

        for file_path in ground_truth_files:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            question_data = json.loads(line)
                            self.save_ground_truth_question(question_data)

                print(f"✅ Imported {os.path.basename(file_path)}")

        # Import processed images
        processed_images_file = "data/processed_images.json"
        if os.path.exists(processed_images_file):
            with open(processed_images_file, 'r') as f:
                processed_data = json.load(f)

            for task_type, image_list in processed_data.items():
                for image_path in image_list:
                    self.save_processed_image(task_type, image_path)

            print(f"✅ Imported processed images data")

        print("🎉 JSON data import completed!")

def main():
    """Initialize centralized database and optionally import existing data"""
    print("🚀 Initializing Centralized Database...")

    db = CentralizedDB()

    # Ask user if they want to import existing JSON data
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--import":
        db.import_existing_json_data()

    print(f"✅ Centralized database ready at: {DB_PATH}")
    print("📊 Database schema created with tables:")
    print("   - attack_executions (replaces attack_parameters.json)")
    print("   - ground_truth_questions (replaces ground_truth JSON files)")
    print("   - processed_images (replaces processed_images.json)")
    print("   - inference_results (replaces inference JSON files)")
    print("   - robustness_evaluation (replaces robustness JSON files)")
    print("\n💡 To import existing JSON data, run:")
    print("   python scripts/utils/centralized_database.py --import")

if __name__ == "__main__":
    main()