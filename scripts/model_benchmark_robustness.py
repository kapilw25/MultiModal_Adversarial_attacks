#!/usr/bin/env python3
"""
Model Benchmark Robustness Analysis Script

Goal: Calculate and store robustness/degradation metrics from results_evaluation table
Creates model_robustness_matrix table with comprehensive adversarial robustness analysis

Data Source: results_evaluation table only (no hardcoded values)
Output: model_robustness_matrix + aggregation views + research plots

Usage:
    python scripts/model_benchmark_robustness.py           # Interactive menu
    python scripts/model_benchmark_robustness.py --auto    # Auto-run both (metrics + plots)
"""

import os
import sys
import sqlite3
import numpy as np
import argparse
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

def verify_database_integrity():
    """
    Verify database integrity before running any analysis.
    Returns: (success: bool, message: str, stats: dict)
    """
    stats = {}

    # Check 1: Database file exists
    if not os.path.exists(DB_PATH):
        return False, f"Database not found: {DB_PATH}", stats

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Check 2: results_evaluation table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='results_evaluation'
        """)
        if not cursor.fetchone():
            conn.close()
            return False, "Table 'results_evaluation' not found. Run model_evaluation.py first.", stats

        # Check 3: Required columns exist
        cursor.execute("PRAGMA table_info(results_evaluation)")
        columns = {row[1] for row in cursor.fetchall()}
        required_columns = {'is_correct', 'epsilon_target', 'model_id', 'attack_type', 'task', 'confidence_score'}
        missing = required_columns - columns
        if missing:
            conn.close()
            return False, f"Missing required columns: {missing}", stats

        # Check 4: Record counts
        cursor.execute("SELECT COUNT(*) FROM results_evaluation")
        total_records = cursor.fetchone()[0]
        stats['total_records'] = total_records

        if total_records == 0:
            conn.close()
            return False, "No records in results_evaluation table. Run model_evaluation.py first.", stats

        # Check 5: Evaluation completeness (is_correct should not be NULL)
        cursor.execute("SELECT COUNT(*) FROM results_evaluation WHERE is_correct IS NULL")
        null_evaluations = cursor.fetchone()[0]
        stats['null_evaluations'] = null_evaluations

        if null_evaluations > 0:
            conn.close()
            return False, f"{null_evaluations} records have NULL is_correct. Re-run model_evaluation.py.", stats

        # Check 6: Get data summary
        cursor.execute("SELECT COUNT(DISTINCT model_id) FROM results_evaluation")
        stats['models'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT attack_type) FROM results_evaluation")
        stats['attack_types'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT task) FROM results_evaluation")
        stats['tasks'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT epsilon_target) FROM results_evaluation")
        stats['epsilon_levels'] = cursor.fetchone()[0]

        conn.close()
        return True, "Database integrity verified", stats

    except sqlite3.Error as e:
        return False, f"Database error: {e}", stats

def check_robustness_matrix_exists():
    """Check if model_robustness_matrix table has data"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='model_robustness_matrix'
        """)
        if not cursor.fetchone():
            conn.close()
            return False, 0

        cursor.execute("SELECT COUNT(*) FROM model_robustness_matrix")
        count = cursor.fetchone()[0]
        conn.close()
        return count > 0, count
    except:
        return False, 0

def display_interactive_menu():
    """
    Display interactive menu for robustness analysis options.
    Returns: selected option (1, 2, or 3)
    """
    print("\n" + "="*60)
    print("  ROBUSTNESS ANALYSIS OPTIONS")
    print("="*60)

    # Check if robustness matrix already exists
    matrix_exists, matrix_count = check_robustness_matrix_exists()

    print("\n  [1] Calculate robustness metrics only")
    print("      → Creates model_robustness_matrix + aggregation views")

    if matrix_exists:
        print(f"\n  [2] Generate plots only")
        print(f"      → Uses existing metrics ({matrix_count} records)")
    else:
        print(f"\n  [2] Generate plots only (UNAVAILABLE)")
        print(f"      → No metrics found. Run option 1 first.")

    print("\n  [3] Both (metrics + plots)")
    print("      → Full analysis pipeline")

    print("\n" + "-"*60)

    while True:
        try:
            choice = input("\nEnter choice [1-3]: ").strip()
            if choice in ['1', '2', '3']:
                choice = int(choice)

                # Validate option 2
                if choice == 2 and not matrix_exists:
                    print("   No metrics available. Please run option 1 first.")
                    continue

                return choice
            else:
                print("   Invalid choice. Enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\n\nOperation cancelled.")
            sys.exit(0)

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
            epsilon_target REAL NOT NULL,

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
            UNIQUE(model_id, task, attack_type, epsilon_target)
        )
        ''')

        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_robustness_model_task ON model_robustness_matrix(model_id, task)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_robustness_attack ON model_robustness_matrix(attack_type, epsilon_target)')
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
            COUNT(DISTINCT epsilon_target) as epsilon_targets,
            COUNT(*) as total_records
        FROM results_evaluation
        ''')

        counts = cursor.fetchone()
        print(f"   Available data: {counts[0]} models, {counts[1]} tasks, {counts[2]} attack types, {counts[3]} epsilon targets")
        print(f"   Total records: {counts[4]}")

        # Get actual unique values
        cursor.execute('SELECT DISTINCT model_id FROM results_evaluation ORDER BY model_id')
        available_models = [row[0] for row in cursor.fetchall()]

        cursor.execute('SELECT DISTINCT task FROM results_evaluation ORDER BY task')
        available_tasks = [row[0] for row in cursor.fetchall()]

        cursor.execute('SELECT DISTINCT attack_type FROM results_evaluation ORDER BY attack_type')
        available_attacks = [row[0] for row in cursor.fetchall()]

        cursor.execute('SELECT DISTINCT epsilon_target FROM results_evaluation ORDER BY epsilon_target')
        available_epsilons = [row[0] for row in cursor.fetchall()]

        conn.close()

        discovery = {
            'models': available_models,
            'tasks': available_tasks,
            'attack_types': available_attacks,
            'epsilon_targets': available_epsilons,
            'total_records': counts[4]
        }

        print(f"   Models: {available_models}")
        print(f"   Tasks: {available_tasks}")
        print(f"   Attack Types: {available_attacks}")
        print(f"   Epsilon Targets: {available_epsilons}")

        return discovery

    def calculate_baseline_accuracy(self, model_id, task):
        """Calculate baseline (clean image) accuracy for a model-task combination"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get clean image performance (attack_type='original', epsilon_target=1.0)
        cursor.execute('''
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct,
            AVG(confidence_score) as avg_confidence
        FROM results_evaluation
        WHERE model_id = ? AND task = ? AND attack_type = 'original' AND epsilon_target = 1.0
        ''', (model_id, task))

        result = cursor.fetchone()
        conn.close()

        if result[0] == 0:
            return 0.0, 0, 0.0  # No baseline data available

        accuracy = (result[1] / result[0]) * 100
        return accuracy, result[0], result[2] or 0.0

    def calculate_attack_performance(self, model_id, task, attack_type, epsilon_target):
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
        WHERE model_id = ? AND task = ? AND attack_type = ? AND epsilon_target = ?
        ''', (model_id, task, attack_type, epsilon_target))

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
        WHERE model_id = ? AND task = ? AND attack_type = ? AND epsilon_target = ?
        ''', (model_id, task, attack_type, epsilon_target))

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

    def calculate_advanced_metrics(self, model_id, task, attack_type, epsilon_target, correct_confidences, incorrect_confidences):
        """Calculate advanced research metrics"""

        # Kendall's Tau correlation (using confidence scores vs correctness)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
        SELECT confidence_score, is_correct
        FROM results_evaluation
        WHERE model_id = ? AND task = ? AND attack_type = ? AND epsilon_target = ?
        AND confidence_score IS NOT NULL
        ''', (model_id, task, attack_type, epsilon_target))

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
                    for epsilon_target in discovery['epsilon_targets']:
                        total_combinations += 1

        print(f"   Processing {total_combinations} model-task-attack-epsilon combinations...")

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
                        for epsilon_target in discovery['epsilon_targets']:

                            # Skip if this is the baseline combination (already calculated)
                            if attack_type == 'original' and epsilon_target == 1.0:
                                # Insert baseline record
                                absolute_deg, relative_deg = self.calculate_degradation_metrics(baseline_accuracy, baseline_accuracy)

                                cursor.execute('''
                                INSERT OR REPLACE INTO model_robustness_matrix VALUES (
                                    NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                                )
                                ''', (
                                    model_id, task, attack_type, epsilon_target,
                                    baseline_accuracy, baseline_accuracy, absolute_deg, relative_deg,
                                    baseline_questions, int(baseline_accuracy * baseline_questions / 100), baseline_confidence,
                                    1.0, 1.0, 0.0,  # Perfect metrics for baseline
                                    datetime.now().isoformat()
                                ))

                            else:
                                # Calculate attack performance
                                attack_acc, attack_questions, attack_conf, correct_conf, incorrect_conf = self.calculate_attack_performance(
                                    model_id, task, attack_type, epsilon_target
                                )

                                if attack_questions > 0:  # Only process if data exists
                                    # Calculate degradation metrics
                                    absolute_deg, relative_deg = self.calculate_degradation_metrics(baseline_accuracy, attack_acc)

                                    # Calculate advanced metrics
                                    rank_corr, eff_dim, inter_class_dist = self.calculate_advanced_metrics(
                                        model_id, task, attack_type, epsilon_target, correct_conf, incorrect_conf
                                    )

                                    # Insert record
                                    cursor.execute('''
                                    INSERT OR REPLACE INTO model_robustness_matrix VALUES (
                                        NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                                    )
                                    ''', (
                                        model_id, task, attack_type, epsilon_target,
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
            epsilon_target,
            AVG(relative_degradation) as avg_impact,
            MIN(attack_accuracy) as worst_case_accuracy,
            COUNT(CASE WHEN relative_degradation > 50 THEN 1 END) as severe_degradation_count,
            COUNT(*) as total_evaluations,
            AVG(inter_class_distance) as avg_separation_impact
        FROM model_robustness_matrix
        WHERE attack_type != 'original'
        GROUP BY attack_type, epsilon_target
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
        SELECT attack_type, epsilon_target, avg_impact, severe_degradation_count, total_evaluations
        FROM attack_effectiveness
        ORDER BY avg_impact DESC
        LIMIT 10
        ''')

        for attack, epsilon, impact, severe, total in cursor.fetchall():
            print(f"   {attack} (epsilon {epsilon}): {impact:.2f}% avg impact, {severe}/{total} severe cases")

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

def run_metrics_calculation(analyzer):
    """Run robustness metrics calculation"""
    print("\n" + "="*60)
    print("  CALCULATING ROBUSTNESS METRICS")
    print("="*60)

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

def run_visualization():
    """Run visualization plot generation"""
    print("\n" + "="*60)
    print("  GENERATING RESEARCH VISUALIZATIONS")
    print("="*60)
    try:
        from utils.model_visualizer import VLMDataAnalyzer, PLOT_DIR
        visualizer = VLMDataAnalyzer()
        visualizer.connect_db()
        if visualizer.load_dynamic_configurations():
            visualizer.generate_all_plots()
        visualizer.close_db()
        print(f"✅ Plots saved to: {PLOT_DIR}")
        return True
    except ImportError as e:
        print(f"❌ Visualization failed: {e}")
        print("   Ensure utils/model_visualizer.py exists")
        return False
    except Exception as e:
        print(f"❌ Visualization error: {e}")
        return False

def main():
    """Main execution function with interactive menu"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Model Benchmark Robustness Analyzer")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-run both metrics + plots (no menu)")
    args = parser.parse_args()

    print("="*80)
    print("  MODEL BENCHMARK ROBUSTNESS ANALYZER")
    print("="*80)
    print("Goal: Calculate robustness/degradation metrics from results_evaluation table")
    print("Output: model_robustness_matrix + aggregation views + research plots")

    # STEP 1: Automatic database health check (defensive programming)
    print("\n" + "-"*60)
    print("  DATABASE HEALTH CHECK")
    print("-"*60)

    success, message, stats = verify_database_integrity()

    if not success:
        print(f"❌ {message}")
        print("\nPipeline order:")
        print("  1. python scripts/attack_runner.py")
        print("  2. python scripts/model_inference_vLLM.py")
        print("  3. python scripts/model_evaluation.py")
        print("  4. python scripts/model_benchmark_robustness.py  ← You are here")
        return

    print(f"✅ {message}")
    print(f"   Records: {stats['total_records']}")
    print(f"   Models: {stats['models']} | Attacks: {stats['attack_types']} | Tasks: {stats['tasks']} | Epsilon levels: {stats['epsilon_levels']}")

    # Initialize analyzer
    analyzer = ModelPerformanceAnalyzer()

    # STEP 2: Auto mode or interactive menu
    if args.auto:
        # Auto mode: run both metrics + plots
        print("\n[AUTO MODE] Running full analysis pipeline...")
        run_metrics_calculation(analyzer)
        run_visualization()
    else:
        # Interactive menu
        choice = display_interactive_menu()

        if choice == 1:
            # Metrics only
            run_metrics_calculation(analyzer)
            print("\n💡 To generate plots, run again and select option 2 or 3")

        elif choice == 2:
            # Plots only (metrics must exist)
            run_visualization()

        elif choice == 3:
            # Both
            run_metrics_calculation(analyzer)
            run_visualization()

    print("\n" + "="*60)
    print("  ANALYSIS COMPLETE")
    print("="*60)
    print(f"💾 Database: {DB_PATH}")
    print(f"📊 Plots: results/research_plots/")

if __name__ == "__main__":
    main()