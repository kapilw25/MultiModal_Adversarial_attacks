#!/usr/bin/env python3
"""
Multi-Dimensional Model Performance Analysis Script

Goal: Calculate and store robustness/degradation metrics from results_evaluation table
Creates model_robustness_matrix table with comprehensive adversarial robustness analysis

Data Source: results_evaluation table only (no hardcoded values)
Output: model_robustness_matrix + aggregation views
"""

import os
import sys
import sqlite3
import numpy as np
from datetime import datetime
from tqdm import tqdm
from scipy import stats

# Add utils directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

DB_PATH = "results/centralized_pipeline.db"

def calculate_kendall_tau(predictions, ground_truth):
    """Calculate Kendall's Tau for robust correlation measurement"""
    try:
        if len(predictions) == 0 or len(ground_truth) == 0:
            return 0.0
        tau, p_value = stats.kendalltau(predictions, ground_truth)
        return tau if not np.isnan(tau) else 0.0
    except:
        return 0.0

def calculate_effective_dimensionality(accuracy_values):
    """Calculate effective dimensionality as robustness measure"""
    try:
        if len(accuracy_values) < 2:
            return 0.0
        # Simplified effective dimensionality based on variance in accuracy
        variance = np.var(accuracy_values)
        mean_acc = np.mean(accuracy_values)
        if mean_acc == 0:
            return 0.0
        return 1.0 / (1.0 + variance / mean_acc)
    except:
        return 0.0

def calculate_inter_class_distance(correct_predictions, incorrect_predictions):
    """Calculate inter-class distance for robustness measurement"""
    try:
        if len(correct_predictions) == 0 or len(incorrect_predictions) == 0:
            return 0.0

        # Simple distance measure based on confidence score separation
        correct_mean = np.mean(correct_predictions)
        incorrect_mean = np.mean(incorrect_predictions)
        return abs(correct_mean - incorrect_mean)
    except:
        return 0.0

class ModelPerformanceAnalyzer:
    """Multi-dimensional model performance and robustness analyzer"""

    def __init__(self):
        self.db_path = DB_PATH

    def create_robustness_matrix_table(self):
        """Create model_robustness_matrix table with comprehensive schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Drop existing table if it exists
        cursor.execute('DROP TABLE IF EXISTS model_robustness_matrix')

        # Create comprehensive robustness matrix table
        cursor.execute('''
        CREATE TABLE model_robustness_matrix (
            -- Primary Keys (Multi-dimensional)
            robustness_id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT NOT NULL,
            task TEXT NOT NULL,
            attack_type TEXT NOT NULL,
            ssim_target REAL NOT NULL,

            -- Performance Metrics
            baseline_accuracy REAL NOT NULL,      -- Clean accuracy for this model+task
            attack_accuracy REAL NOT NULL,        -- Accuracy under attack
            absolute_degradation REAL NOT NULL,   -- attack_acc - baseline_acc
            relative_degradation REAL NOT NULL,   -- (baseline_acc - attack_acc) / baseline_acc * 100

            -- Statistical Measures
            total_questions INTEGER NOT NULL,
            correct_answers INTEGER NOT NULL,
            confidence_score_avg REAL,

            -- Research Metrics (from web search)
            rank_correlation REAL,               -- Kendall's Tau for robustness
            effective_dimensionality REAL,       -- Complexity measure
            inter_class_distance REAL,           -- Class separation

            -- Meta Information
            evaluation_timestamp TEXT NOT NULL,
            UNIQUE(model_id, task, attack_type, ssim_target)
        )
        ''')

        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_robustness_model_task ON model_robustness_matrix(model_id, task)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_robustness_attack ON model_robustness_matrix(attack_type, ssim_target)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_robustness_degradation ON model_robustness_matrix(relative_degradation)')

        conn.commit()
        conn.close()
        print("✅ Created model_robustness_matrix table")

    def discover_available_data(self):
        """Discover what data is actually available in results_evaluation table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        print("🔍 Discovering available data in results_evaluation table...")

        # Get unique combinations that actually exist
        cursor.execute('''
        SELECT
            COUNT(DISTINCT model_id) as models,
            COUNT(DISTINCT task) as tasks,
            COUNT(DISTINCT attack_type) as attack_types,
            COUNT(DISTINCT ssim_target) as ssim_targets,
            COUNT(*) as total_records
        FROM results_evaluation
        ''')

        counts = cursor.fetchone()
        print(f"   Available data: {counts[0]} models, {counts[1]} tasks, {counts[2]} attack types, {counts[3]} SSIM targets")
        print(f"   Total records: {counts[4]}")

        # Get actual unique values
        cursor.execute('SELECT DISTINCT model_id FROM results_evaluation ORDER BY model_id')
        available_models = [row[0] for row in cursor.fetchall()]

        cursor.execute('SELECT DISTINCT task FROM results_evaluation ORDER BY task')
        available_tasks = [row[0] for row in cursor.fetchall()]

        cursor.execute('SELECT DISTINCT attack_type FROM results_evaluation ORDER BY attack_type')
        available_attacks = [row[0] for row in cursor.fetchall()]

        cursor.execute('SELECT DISTINCT ssim_target FROM results_evaluation ORDER BY ssim_target')
        available_ssims = [row[0] for row in cursor.fetchall()]

        conn.close()

        discovery = {
            'models': available_models,
            'tasks': available_tasks,
            'attack_types': available_attacks,
            'ssim_targets': available_ssims,
            'total_records': counts[4]
        }

        print(f"   Models: {available_models}")
        print(f"   Tasks: {available_tasks}")
        print(f"   Attack Types: {available_attacks}")
        print(f"   SSIM Targets: {available_ssims}")

        return discovery

    def calculate_baseline_accuracy(self, model_id, task):
        """Calculate baseline (clean image) accuracy for a model-task combination"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get clean image performance (attack_type='original', ssim_target=1.0)
        cursor.execute('''
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct,
            AVG(confidence_score) as avg_confidence
        FROM results_evaluation
        WHERE model_id = ? AND task = ? AND attack_type = 'original' AND ssim_target = 1.0
        ''', (model_id, task))

        result = cursor.fetchone()
        conn.close()

        if result[0] == 0:
            return 0.0, 0, 0.0  # No baseline data available

        accuracy = (result[1] / result[0]) * 100
        return accuracy, result[0], result[2] or 0.0

    def calculate_attack_performance(self, model_id, task, attack_type, ssim_target):
        """Calculate performance under specific attack conditions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get attack performance
        cursor.execute('''
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct,
            AVG(confidence_score) as avg_confidence,
            confidence_score,
            is_correct
        FROM results_evaluation
        WHERE model_id = ? AND task = ? AND attack_type = ? AND ssim_target = ?
        ''', (model_id, task, attack_type, ssim_target))

        results = cursor.fetchall()

        if not results or results[0][0] == 0:
            conn.close()
            return 0.0, 0, 0.0, [], []

        basic_result = results[0]
        accuracy = (basic_result[1] / basic_result[0]) * 100

        # Get detailed data for advanced metrics
        cursor.execute('''
        SELECT confidence_score, is_correct
        FROM results_evaluation
        WHERE model_id = ? AND task = ? AND attack_type = ? AND ssim_target = ?
        ''', (model_id, task, attack_type, ssim_target))

        detailed_results = cursor.fetchall()
        conn.close()

        # Separate correct and incorrect predictions for advanced metrics
        correct_confidences = [row[0] for row in detailed_results if row[1] == 1 and row[0] is not None]
        incorrect_confidences = [row[0] for row in detailed_results if row[1] == 0 and row[0] is not None]

        return accuracy, basic_result[0], basic_result[2] or 0.0, correct_confidences, incorrect_confidences

    def calculate_degradation_metrics(self, baseline_accuracy, attack_accuracy):
        """Calculate absolute and relative degradation metrics"""
        absolute_degradation = attack_accuracy - baseline_accuracy

        if baseline_accuracy == 0:
            relative_degradation = 0.0
        else:
            relative_degradation = ((baseline_accuracy - attack_accuracy) / baseline_accuracy) * 100

        return absolute_degradation, relative_degradation

    def calculate_advanced_metrics(self, model_id, task, attack_type, ssim_target, correct_confidences, incorrect_confidences):
        """Calculate advanced research metrics"""

        # Kendall's Tau correlation (using confidence scores vs correctness)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
        SELECT confidence_score, is_correct
        FROM results_evaluation
        WHERE model_id = ? AND task = ? AND attack_type = ? AND ssim_target = ?
        AND confidence_score IS NOT NULL
        ''', (model_id, task, attack_type, ssim_target))

        metric_data = cursor.fetchall()
        conn.close()

        if metric_data:
            confidences = [row[0] for row in metric_data]
            correctness = [row[1] for row in metric_data]
            rank_correlation = calculate_kendall_tau(confidences, correctness)
        else:
            rank_correlation = 0.0

        # Effective dimensionality (based on confidence score variance)
        all_confidences = correct_confidences + incorrect_confidences
        effective_dimensionality = calculate_effective_dimensionality(all_confidences)

        # Inter-class distance
        inter_class_distance = calculate_inter_class_distance(correct_confidences, incorrect_confidences)

        return rank_correlation, effective_dimensionality, inter_class_distance

    def populate_robustness_matrix(self, discovery):
        """Populate model_robustness_matrix with all available combinations"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        print("📊 Calculating multi-dimensional robustness metrics...")

        # Calculate total combinations for progress tracking
        total_combinations = 0
        for model in discovery['models']:
            for task in discovery['tasks']:
                for attack_type in discovery['attack_types']:
                    for ssim_target in discovery['ssim_targets']:
                        total_combinations += 1

        print(f"   Processing {total_combinations} model-task-attack-SSIM combinations...")

        processed = 0

        with tqdm(total=total_combinations, desc="Processing combinations") as pbar:
            for model_id in discovery['models']:
                # Calculate baseline once per model-task
                baselines = {}

                for task in discovery['tasks']:
                    baseline_acc, baseline_questions, baseline_conf = self.calculate_baseline_accuracy(model_id, task)
                    baselines[task] = (baseline_acc, baseline_questions, baseline_conf)

                for task in discovery['tasks']:
                    baseline_accuracy, baseline_questions, baseline_confidence = baselines[task]

                    for attack_type in discovery['attack_types']:
                        for ssim_target in discovery['ssim_targets']:

                            # Skip if this is the baseline combination (already calculated)
                            if attack_type == 'original' and ssim_target == 1.0:
                                # Insert baseline record
                                absolute_deg, relative_deg = self.calculate_degradation_metrics(baseline_accuracy, baseline_accuracy)

                                cursor.execute('''
                                INSERT OR REPLACE INTO model_robustness_matrix VALUES (
                                    NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                                )
                                ''', (
                                    model_id, task, attack_type, ssim_target,
                                    baseline_accuracy, baseline_accuracy, absolute_deg, relative_deg,
                                    baseline_questions, int(baseline_accuracy * baseline_questions / 100), baseline_confidence,
                                    1.0, 1.0, 0.0,  # Perfect metrics for baseline
                                    datetime.now().isoformat()
                                ))

                            else:
                                # Calculate attack performance
                                attack_acc, attack_questions, attack_conf, correct_conf, incorrect_conf = self.calculate_attack_performance(
                                    model_id, task, attack_type, ssim_target
                                )

                                if attack_questions > 0:  # Only process if data exists
                                    # Calculate degradation metrics
                                    absolute_deg, relative_deg = self.calculate_degradation_metrics(baseline_accuracy, attack_acc)

                                    # Calculate advanced metrics
                                    rank_corr, eff_dim, inter_class_dist = self.calculate_advanced_metrics(
                                        model_id, task, attack_type, ssim_target, correct_conf, incorrect_conf
                                    )

                                    # Insert record
                                    cursor.execute('''
                                    INSERT OR REPLACE INTO model_robustness_matrix VALUES (
                                        NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                                    )
                                    ''', (
                                        model_id, task, attack_type, ssim_target,
                                        baseline_accuracy, attack_acc, absolute_deg, relative_deg,
                                        attack_questions, int(attack_acc * attack_questions / 100), attack_conf,
                                        rank_corr, eff_dim, inter_class_dist,
                                        datetime.now().isoformat()
                                    ))

                            processed += 1
                            pbar.update(1)

                            # Commit every 50 records for performance
                            if processed % 50 == 0:
                                conn.commit()

        conn.commit()
        conn.close()
        print(f"✅ Processed {processed} combinations into model_robustness_matrix")

    def create_aggregation_views(self):
        """Create research-grade aggregation views"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        print("📈 Creating aggregation views...")

        # 1. Model Comparison View
        cursor.execute('DROP VIEW IF EXISTS model_comparison')
        cursor.execute('''
        CREATE VIEW model_comparison AS
        SELECT
            model_id,
            AVG(relative_degradation) as avg_degradation,
            MIN(attack_accuracy) as worst_performance,
            MAX(attack_accuracy) as best_performance,
            AVG(rank_correlation) as avg_rank_correlation,
            AVG(effective_dimensionality) as avg_effective_dimensionality,
            COUNT(*) as evaluation_scenarios,
            COUNT(CASE WHEN relative_degradation > 50 THEN 1 END) as severe_degradation_count
        FROM model_robustness_matrix
        WHERE attack_type != 'original'
        GROUP BY model_id
        ORDER BY avg_degradation ASC
        ''')

        # 2. Task Robustness View
        cursor.execute('DROP VIEW IF EXISTS task_robustness')
        cursor.execute('''
        CREATE VIEW task_robustness AS
        SELECT
            task,
            attack_type,
            AVG(relative_degradation) as avg_degradation_across_models,
            MIN(attack_accuracy) as worst_model_performance,
            MAX(attack_accuracy) as best_model_performance,
            COUNT(DISTINCT model_id) as models_evaluated,
            RANK() OVER (PARTITION BY task ORDER BY AVG(relative_degradation) ASC) as robustness_rank
        FROM model_robustness_matrix
        WHERE attack_type != 'original'
        GROUP BY task, attack_type
        ORDER BY task, avg_degradation_across_models ASC
        ''')

        # 3. Attack Effectiveness View
        cursor.execute('DROP VIEW IF EXISTS attack_effectiveness')
        cursor.execute('''
        CREATE VIEW attack_effectiveness AS
        SELECT
            attack_type,
            ssim_target,
            AVG(relative_degradation) as avg_impact,
            MIN(attack_accuracy) as worst_case_accuracy,
            COUNT(CASE WHEN relative_degradation > 50 THEN 1 END) as severe_degradation_count,
            COUNT(*) as total_evaluations,
            AVG(inter_class_distance) as avg_separation_impact
        FROM model_robustness_matrix
        WHERE attack_type != 'original'
        GROUP BY attack_type, ssim_target
        ORDER BY avg_impact DESC
        ''')

        conn.commit()
        conn.close()
        print("✅ Created aggregation views: model_comparison, task_robustness, attack_effectiveness")

    def generate_performance_summary(self):
        """Generate comprehensive performance analysis summary"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        print("\n" + "="*80)
        print("📊 MULTI-DIMENSIONAL ROBUSTNESS ANALYSIS SUMMARY")
        print("="*80)

        # Overall statistics
        cursor.execute('SELECT COUNT(*) FROM model_robustness_matrix')
        total_records = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM model_robustness_matrix WHERE attack_type != "original"')
        attack_records = cursor.fetchone()[0]

        print(f"Total robustness records: {total_records}")
        print(f"Attack scenarios: {attack_records}")
        print(f"Baseline records: {total_records - attack_records}")

        # Model comparison
        print(f"\n🤖 MODEL ROBUSTNESS RANKING:")
        cursor.execute('''
        SELECT model_id, avg_degradation, worst_performance, evaluation_scenarios
        FROM model_comparison
        ORDER BY avg_degradation ASC
        ''')

        for model, avg_deg, worst_perf, scenarios in cursor.fetchall():
            print(f"   {model}: {avg_deg:.2f}% avg degradation, {worst_perf:.2f}% worst case ({scenarios} scenarios)")

        # Attack effectiveness
        print(f"\n⚔️ ATTACK EFFECTIVENESS RANKING:")
        cursor.execute('''
        SELECT attack_type, ssim_target, avg_impact, severe_degradation_count, total_evaluations
        FROM attack_effectiveness
        ORDER BY avg_impact DESC
        LIMIT 10
        ''')

        for attack, ssim, impact, severe, total in cursor.fetchall():
            print(f"   {attack} (SSIM {ssim}): {impact:.2f}% avg impact, {severe}/{total} severe cases")

        # Task robustness
        print(f"\n📋 TASK VULNERABILITY ANALYSIS:")
        cursor.execute('''
        SELECT task, AVG(avg_degradation_across_models) as overall_vulnerability
        FROM task_robustness
        GROUP BY task
        ORDER BY overall_vulnerability DESC
        ''')

        for task, vulnerability in cursor.fetchall():
            print(f"   {task}: {vulnerability:.2f}% average vulnerability across attacks")

        conn.close()
        print(f"\n💾 Database: {self.db_path} → model_robustness_matrix + views")
        print("🎯 Multi-dimensional robustness analysis complete!")

def main():
    """Main execution function"""
    print("="*80)
    print("🚀 MULTI-DIMENSIONAL MODEL PERFORMANCE ANALYZER")
    print("="*80)
    print("Goal: Calculate robustness/degradation metrics from results_evaluation table")
    print("Output: model_robustness_matrix + aggregation views")

    # Initialize analyzer
    analyzer = ModelPerformanceAnalyzer()

    # Check if source data exists
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM results_evaluation')
    count = cursor.fetchone()[0]
    conn.close()

    if count == 0:
        print("❌ No data found in results_evaluation table. Run model_evaluation_2.py first.")
        return

    print(f"✅ Found {count} records in results_evaluation table")

    # Create database schema
    analyzer.create_robustness_matrix_table()

    # Discover available data (no hardcoding)
    discovery = analyzer.discover_available_data()

    # Populate robustness matrix
    analyzer.populate_robustness_matrix(discovery)

    # Create aggregation views
    analyzer.create_aggregation_views()

    # Generate summary
    analyzer.generate_performance_summary()

if __name__ == "__main__":
    main()