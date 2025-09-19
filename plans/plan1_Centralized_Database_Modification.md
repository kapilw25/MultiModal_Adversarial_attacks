● Centralized Database Modification Plan

Summary of Changes

  1. Create centralized database schema replacing all JSON files
  2. Modify attack_runner.py to save directly to database
  3. Modify model_inference.py to read from database and save directly
  4. Modify model_evaluation.py to load from database instead of files
  5. Modify model_analysis_visualizer.py to use centralized database
  6. Remove JSON file dependencies throughout the pipeline

  This approach eliminates all intermediate JSON files and creates a single source of truth in results/centralized_pipeline.db.
#####################################
● Centralized Database Modification Plan
  1. Create Enhanced Database Schema (scripts/utils/centralized_database.py)

  #!/usr/bin/env python3
  """
  Centralized Database Manager for VLM Adversarial Attack Pipeline
  Replaces all JSON files with direct SQLite database operations
  """

  import sqlite3
  import os
  from datetime import datetime
  from pathlib import Path

  DB_PATH = "results/centralized_pipeline.db"

  def create_centralized_schema():
      """Create comprehensive database schema for all pipeline data"""
      conn = sqlite3.connect(DB_PATH)
      cursor = conn.cursor()

      # 1. Attack Executions (replaces attack_parameters.json)
      cursor.execute('''
      CREATE TABLE IF NOT EXISTS attack_executions (
          execution_id TEXT PRIMARY KEY,
          attack_name TEXT NOT NULL,
          attack_category TEXT NOT NULL,
          image_path TEXT NOT NULL,
          adversarial_image_path TEXT NOT NULL,
          image_name TEXT NOT NULL,
          task_type TEXT NOT NULL,
          execution_time_seconds INTEGER NOT NULL,
          success BOOLEAN NOT NULL,
          timestamp TEXT NOT NULL,
          -- Parameters
          ssim_target REAL,
          ssim REAL,
          mean_perturbation REAL,
          max_perturbation REAL,
          l2_norm REAL,
          l0_norm REAL,
          total_queries INTEGER,
          -- Metadata
          execution_date TEXT,
          description TEXT,
          ssim_threshold REAL,
          completed_attacks INTEGER
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

      # 4. Model Inference Results (replaces inference JSON files)
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
          -- Metadata
          adversarial BOOLEAN NOT NULL,
          task TEXT NOT NULL,
          attack_type TEXT NOT NULL,
          ssim REAL NOT NULL,
          image_path TEXT NOT NULL,
          original_image_path TEXT NOT NULL,
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

      conn.commit()
      conn.close()

  # Database operation functions
  class CentralizedDB:
      def __init__(self):
          self.db_path = DB_PATH
          os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
          create_centralized_schema()

      def save_attack_execution(self, execution_data):
          """Save attack execution data (replaces JSON append)"""
          conn = sqlite3.connect(self.db_path)
          cursor = conn.cursor()

          cursor.execute('''
          INSERT OR REPLACE INTO attack_executions VALUES (
              ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
          )
          ''', (
              execution_data['execution_id'],
              execution_data['attack_name'],
              execution_data['attack_category'],
              execution_data['image_path'],
              execution_data['adversarial_image_path'],
              execution_data['image_name'],
              execution_data['task_type'],
              execution_data['execution_time_seconds'],
              execution_data['success'],
              execution_data['timestamp'],
              execution_data['parameters']['ssim_target'],
              execution_data['parameters']['ssim'],
              execution_data['parameters']['mean_perturbation'],
              execution_data['parameters']['max_perturbation'],
              execution_data['parameters']['l2_norm'],
              execution_data['parameters']['l0_norm'],
              execution_data['parameters']['total_queries'],
              execution_data.get('execution_date'),
              execution_data.get('description'),
              execution_data.get('ssim_threshold'),
              execution_data.get('completed_attacks')
          ))

          conn.commit()
          conn.close()

      def save_inference_result(self, result_data):
          """Save model inference result (replaces JSON file writing)"""
          conn = sqlite3.connect(self.db_path)
          cursor = conn.cursor()

          pm = result_data.get('performance_metrics', {})
          gpu = pm.get('gpu_memory', {})
          cpu = pm.get('cpu_memory', {})
          loading = pm.get('model_loading', {})
          batch = pm.get('batch_info', {})

          cursor.execute('''
          INSERT OR REPLACE INTO inference_results VALUES (
              ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
          )
          ''', (
              f"{result_data['question_id']}_{result_data['model_id']}_{result_data['metadata']['attack_type']}",
              result_data['question_id'],
              result_data['prompt'],
              result_data['text'],
              result_data['truth'],
              result_data['type'],
              result_data['answer_id'],
              str(result_data['markers']),
              result_data['model_id'],
              result_data['metadata']['adversarial'],
              result_data['metadata']['task'],
              result_data['metadata']['attack_type'],
              result_data['metadata']['ssim'],
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

      def save_robustness_evaluation(self, model_name, attack_type, task_name, eval_data):
          """Save robustness evaluation (replaces robustness JSON)"""
          conn = sqlite3.connect(self.db_path)
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

  2. Modify scripts/attack_runner.py

  # Add at top of file
  from utils.centralized_database import CentralizedDB

  class AttackOrchestrator:
      def __init__(self, config: AttackConfig):
          # ... existing code ...
          self.db = CentralizedDB()  # Add database connection

      def save_attack_results(self, attack_name: str, image_path: str,
                            execution_time: int, success: bool,
                            metrics: Dict[str, Any], attack_category: str,
                            ssim_threshold: float = None):
          """MODIFIED: Save directly to database instead of JSON"""

          # Construct execution data
          execution_data = {
              'execution_id': f"{self.execution_id}_{attack_name}_{os.path.basename(image_path)}",
              'attack_name': attack_name,
              'attack_category': attack_category,
              'image_path': image_path,
              'adversarial_image_path': self.construct_adversarial_image_path(
                  image_path, attack_name, ssim_threshold or self.config.ssim_threshold,
                  attack_category == "Black-Box"
              ),
              'image_name': os.path.basename(image_path),
              'task_type': image_path.split('/')[2] if len(image_path.split('/')) > 2 else "unknown",
              'execution_time_seconds': execution_time,
              'success': success,
              'timestamp': datetime.now().isoformat(),
              'parameters': {
                  'ssim_target': ssim_threshold or self.config.ssim_threshold,
                  'ssim': metrics.get('ssim'),
                  'mean_perturbation': metrics.get('mean_perturbation'),
                  'max_perturbation': metrics.get('max_perturbation'),
                  'l2_norm': metrics.get('l2_norm'),
                  'l0_norm': metrics.get('l0_norm'),
                  'total_queries': metrics.get('total_queries')
              },
              'execution_date': datetime.now().isoformat(),
              'description': "Universal attack results with SSIM optimization",
              'ssim_threshold': self.config.ssim_threshold,
              'completed_attacks': self.success_count + self.failure_count + 1
          }

          # Save to database instead of JSON
          self.db.save_attack_execution(execution_data)

          logger.info(f"[{attack_name.upper()}] Saved to DB: SSIM={metrics.get('ssim'):.4f}, "
                     f"Mean_Pert={metrics.get('mean_perturbation'):.2f}, Success={success}")

  3. Modify scripts/model_inference.py

  # Add at top of file
  from utils.centralized_database import CentralizedDB

  def load_attack_parameters():
      """MODIFIED: Load from database instead of JSON"""
      db = CentralizedDB()
      conn = sqlite3.connect(db.db_path)

      query = '''
      SELECT image_path, adversarial_image_path, ssim, ssim_target, attack_name, attack_category
      FROM attack_executions 
      WHERE success = 1
      '''

      results = conn.execute(query).fetchall()
      conn.close()

      # Create mapping compatible with existing code
      ssim_mapping = {}
      for row in results:
          composite_key = (row[0], row[1])  # (image_path, adversarial_image_path)
          ssim_mapping[composite_key] = {
              'ssim': row[2],
              'ssim_target': row[3],
              'adversarial_image_path': row[1],
              'attack_type': row[4],
              'attack_category': row[5]
          }

      print(f"📊 Loaded SSIM data for {len(ssim_mapping)} image-attack combinations from database")
      return ssim_mapping

  def run_evaluation(...):
      """MODIFIED: Save results directly to database"""
      # ... existing code until result processing ...

      db = CentralizedDB()

      # Save results directly to database instead of JSON file
      for result in res_list:
          db.save_inference_result(result)

      print(f"\n✅ Inference completed! Results saved to centralized database")
      print(f"📊 Processed {len(res_list)} questions across {len(image_data_list)} images")

  4. Modify scripts/model_evaluation.py

  # Add at top of file
  from utils.centralized_database import CentralizedDB

  def load_inference_results_from_db(engine, task):
      """NEW: Load inference results from database instead of JSON files"""
      db = CentralizedDB()
      conn = sqlite3.connect(db.db_path)

      query = '''
      SELECT question_id, prompt, model_response as text, ground_truth as truth, 
             question_type as type, attack_type, adversarial, ssim
      FROM inference_results 
      WHERE model_id = ? AND task = ?
      '''

      results = conn.execute(query, (engine, task)).fetchall()
      conn.close()

      # Convert to format expected by evaluator
      eval_data = []
      for row in results:
          eval_data.append({
              'question_id': row[0],
              'prompt': row[1],
              'text': row[2],
              'truth': row[3],
              'type': row[4],
              'attack_type': row[5],
              'adversarial': row[6],
              'ssim': row[7]
          })

      return eval_data

  def save_results_to_database(engine, task, change_data, performance_metrics=None):
      """MODIFIED: Save directly to database instead of JSON"""
      if not change_data:
          return

      db = CentralizedDB()

      # Process each row of change data
      for row in change_data:
          attack_type = row[0]
          accuracy = float(row[2].strip('%'))
          change = float(row[3].strip('+-%')) if row[3] != "0.00%" else 0.0
          if row[3].startswith('-'):
              change = -change
          effect = row[4]

          # Create evaluation data
          eval_data = {
              "accuracy": accuracy,
              "change": change,
              "effect": effect
          }

          # Add performance metrics if available
          if performance_metrics and attack_type in performance_metrics:
              eval_data["performance_metrics"] = performance_metrics[attack_type]

          # Save to database
          db.save_robustness_evaluation(engine, attack_type, task, eval_data)

      print(f"Results saved to centralized database")
      print(f"Performance metrics integrated for {len(performance_metrics) if performance_metrics else 0} attack types")

  def evaluate_all_files(engine, task, random_count=None):
      """MODIFIED: Load from database instead of file globbing"""

      # Load results from database instead of files
      eval_data = load_inference_results_from_db(engine, task)

      if not eval_data:
          print(f"No evaluation data found for {engine} on task '{task}' in database")
          return

      # Group by attack type for evaluation
      attack_groups = {}
      for item in eval_data:
          attack_type = item['attack_type']
          if attack_type not in attack_groups:
              attack_groups[attack_type] = []
          attack_groups[attack_type].append(item)

      # Evaluate each attack type
      results = {}
      for attack_type, data in attack_groups.items():
          # Run evaluator on this group
          _, _, accuracy, file_type = evaluator_from_data(data)  # NEW function needed
          results[attack_type] = accuracy

      # ... rest of existing evaluation logic ...

      # Save results to database instead of JSON
      save_results_to_database(engine, task, change_data, performance_metrics)

  5. Modify scripts/model_analysis_visualizer.py

  class VLMDataAnalyzer:
      def __init__(self):
          self.db_path = "results/centralized_pipeline.db"  # CHANGED: Use centralized DB
          # ... rest of existing code ...

      def load_robustness_data(self):
          """MODIFIED: Load from centralized database"""
          try:
              query = '''
              SELECT 
                  eval_id as result_id,
                  task_name,
                  attack_type as attack_name,
                  CASE 
                      WHEN attack_type = 'Original' THEN 'Original'
                      WHEN attack_type IN ('fgsm', 'pgd', 'cw_linf', 'deepfool') THEN 'White-Box'
                      ELSE 'Black-Box'
                  END as attack_category,
                  model_name,
                  'Unknown' as model_family,  -- Will be derived from model_name
                  'Unknown' as size_category,  -- Will be derived from model_name
                  accuracy,
                  accuracy_change,
                  evaluation_timestamp as timestamp
              FROM robustness_evaluation
              '''

              df = pd.read_sql_query(query, self.conn)
              print(f"✅ Loaded robustness data: {len(df)} records from centralized database")
              return df

          except Exception as e:
              print(f"❌ Error loading robustness data: {e}")
              return None

      def load_performance_data(self):
          """MODIFIED: Load from centralized database"""
          try:
              query = '''
              SELECT 
                  eval_id as metric_id,
                  model_name,
                  'Unknown' as family_name,  -- Will be derived
                  'Unknown' as size_category,  -- Will be derived
                  attack_type as attack_name,
                  avg_inference_time_seconds,
                  avg_gpu_memory_allocated_mb,
                  avg_gpu_memory_peak_mb,
                  avg_gpu_memory_reserved_mb,
                  total_gpu_memory_mb,
                  avg_cpu_memory_mb,
                  model_loading_time_seconds,
                  cache_hit_ratio,
                  total_questions,
                  evaluation_timestamp as timestamp
              FROM robustness_evaluation
              WHERE avg_inference_time_seconds IS NOT NULL
              '''

              df = pd.read_sql_query(query, self.conn)
              print(f"✅ Loaded performance data: {len(df)} records from centralized database")
              return df

          except Exception as e:
              print(f"❌ Error loading performance data: {e}")
              return None

  6. Remove Database Manager Dependency

  # In scripts/model_evaluation.py - REMOVE these lines:
  # def update_database_from_json(json_path):
  #     # Delete this entire function

  # In save_results_to_database() - REMOVE:
  # update_database_from_json(json_path)  # Remove this call

  Summary of Changes

  1. Create centralized database schema replacing all JSON files
  2. Modify attack_runner.py to save directly to database
  3. Modify model_inference.py to read from database and save directly
  4. Modify model_evaluation.py to load from database instead of files
  5. Modify model_analysis_visualizer.py to use centralized database
  6. Remove JSON file dependencies throughout the pipeline

  This approach eliminates all intermediate JSON files and creates a single source of truth in results/centralized_pipeline.db.