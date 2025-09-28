#!/usr/bin/env python3
"""
Universal Attack Runner (White-Box + Black-Box)

A comprehensive orchestrator for adversarial attacks following ML research standards.
Supports both whitebox and blackbox attacks through unified frameworks with
direct epsilon parameter control.

Features (OPTIMIZED FOR 7.6GB GPU):
- 5 Fast Whitebox attacks: FGSM, PGD, AutoPGD, AutoConjugateGradient, BasicIterativeMethod
- 4 Fast Blackbox attacks: Square, SimBA, Boundary, SignOPT
- 100% epsilon parameter support for precise perturbation control
- Interactive task and attack selection
- Automatic results logging to centralized database
- Direct epsilon control (no optimization needed)
- Fast execution with predictable results

Usage:
    python scripts/attack_runner.py

Author: AI/ML Research Pipeline
"""

import os
import sys
import json
import subprocess
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import re
import glob
import shutil
from dataclasses import dataclass

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import centralized database and query counter
from scripts.utils.centralized_database import CentralizedDB

# Import query counter for metrics tracking
try:
    from attack_models.utils import query_counter
except ImportError:
    # Fallback if import fails
    class DummyCounter:
        def reset(self): pass
        def get_count(self): return 0
    query_counter = DummyCounter()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class AttackConfig:
    """Configuration for attack execution"""
    epsilon_levels: List[str] = None  # Support multiple epsilon levels
    max_trials_whitebox: int = 1  # No trials needed for direct epsilon
    max_trials_blackbox: int = 1   # No trials needed for direct epsilon

    def __post_init__(self):
        # If epsilon_levels is not provided, use default moderate
        if self.epsilon_levels is None:
            self.epsilon_levels = ['moderate']

class AttackOrchestrator:
    """Main orchestrator for adversarial attacks"""

    # Epsilon level mapping for direct perturbation control - Standard Research Values
    EPSILON_LEVELS = {
        'minimal': {
            'epsilon': 4/255,  # ≈ 0.016 - ImageNet standard
            'description': 'ImageNet standard (ε=4/255≈0.016)',
            'strength': 'Minimal - Barely perceptible'
        },
        'standard': {
            'epsilon': 8/255,  # ≈ 0.031 - CIFAR-10 standard
            'description': 'CIFAR-10 standard (ε=8/255≈0.031)',
            'strength': 'Standard - Imperceptible'
        },
        'moderate': {
            'epsilon': 16/255,  # ≈ 0.063 - Higher but invisible
            'description': 'Higher perturbations (ε=16/255≈0.063)',
            'strength': 'Moderate - Still invisible'
        }
    }

    # Attack mappings - 100% EPSILON PARAMETER SUPPORT (9 fast attacks)
    # ALL attacks support direct epsilon parameter control
    WHITEBOX_ATTACKS = {
        1: "fgsm", 2: "pgd", 3: "auto_pgd", 4: "auto_conjugate_gradient", 5: "basic_iterative", 6: "all"
        # ALL attacks support epsilon parameter for consistent perturbation control
    }

    BLACKBOX_ATTACKS = {
        1: "square", 2: "simba", 3: "boundary", 4: "sign_opt", 5: "all"
        # ALL attacks support epsilon parameter for consistent perturbation control
    }

    TASK_MAPPINGS = {
        1: "chart", 2: "table", 3: "road_map", 4: "dashboard",
        5: "flowchart", 6: "relation_graph", 7: "planar_layout",
        8: "visual_puzzle", 9: "all"
    }

    def __init__(self, config: AttackConfig):
        self.config = config
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)

        # Initialize execution tracking
        self.execution_id = f"universal_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.success_count = 0
        self.failure_count = 0
        self.is_replacement_run = False

        # Initialize centralized database connection
        self.db = CentralizedDB()

    def get_epsilon_from_level(self, epsilon_level: str) -> float:
        """Get epsilon value from epsilon level - direct mapping"""
        if epsilon_level.startswith('custom_'):
            # Extract epsilon from custom level name (e.g., 'custom_0.030')
            return float(epsilon_level.split('_')[1])
        return self.EPSILON_LEVELS.get(epsilon_level, {}).get('epsilon', 0.05)

    def display_header(self):
        """Display application header"""
        print("🎯 Universal Attack Runner (White-Box + Black-Box) - 100% EPSILON CONTROL")
        print("=" * 70)
        print("🚀 Whitebox: 5 epsilon-based attacks (FGSM, PGD, AutoPGD, AutoConjugateGradient, BasicIterative)")
        print("🔒 Blackbox: 4 epsilon-based attacks (Square, SimBA, Boundary, SignOPT)")
        print("⚡ Direct epsilon parameter control - no optimization needed")
        print("✅ Fast, predictable execution with precise perturbation bounds")
        print("🎯 100% epsilon parameter support across all attacks")
        print("=" * 70)
        print()

    def check_environment(self):
        """Check virtual environment and dependencies"""
        if not os.environ.get('VIRTUAL_ENV'):
            logger.warning("No virtual environment detected")

        # Check if required scripts exist
        whitebox_script = Path("attack_models/white_box_universal.py")
        blackbox_script = Path("attack_models/black_box_universal.py")

        if not whitebox_script.exists():
            raise FileNotFoundError(f"Whitebox script not found: {whitebox_script}")
        if not blackbox_script.exists():
            raise FileNotFoundError(f"Blackbox script not found: {blackbox_script}")

        logger.info("✅ Environment check passed")

    def select_task(self) -> str:
        """Interactive task selection"""
        print("Select the task to run attacks on:")
        print("  [1] Chart interpretation (4 images)")
        print("  [2] Table data extraction (3 images)")
        print("  [3] Road map navigation (3 images)")
        print("  [4] Dashboard analysis (3 images)")
        print("  [5] Flowchart understanding (3 images)")
        print("  [6] Relation graph analysis (3 images)")
        print("  [7] Planar layout interpretation (3 images)")
        print("  [8] Visual puzzle solving (3 images)")
        print("  [9] ALL tasks (25 images total)")
        print()

        while True:
            try:
                choice = int(input("Enter your choice (1-9): "))
                if choice in self.TASK_MAPPINGS:
                    task_name = self.TASK_MAPPINGS[choice]
                    print(f"✅ Selected task: {task_name}")
                    return task_name
                else:
                    print("❌ Invalid choice. Please enter 1-9.")
            except (ValueError, KeyboardInterrupt):
                print("❌ Please enter a valid number (1-9).")

    def select_category(self) -> str:
        """Interactive attack category selection"""
        print("\nSelect attack category (100% EPSILON PARAMETER SUPPORT):")
        print("  [1] White-Box Attacks (5 epsilon attacks: FGSM, PGD, AutoPGD, AutoConjugateGradient, BasicIterative)")
        print("  [2] Black-Box Attacks (4 epsilon attacks: Square, SimBA, Boundary, SignOPT)")
        print("  [3] All Attacks (9 epsilon attacks total: 5 White-Box + 4 Black-Box)")
        print("\n✅ ALL ATTACKS SUPPORT EPSILON:")
        print("   • Direct epsilon parameter control")
        print("   • Fast execution (no optimization needed)")
        print("   • Predictable perturbation bounds")
        print()

        while True:
            try:
                choice = int(input("Enter your choice (1-3): "))
                if choice == 1:
                    print("✅ Selected category: White-Box Attacks")
                    return "whitebox"
                elif choice == 2:
                    print("✅ Selected category: Black-Box Attacks")
                    return "blackbox"
                elif choice == 3:
                    print("✅ Selected category: All Attacks (9 epsilon attacks total)")
                    return "all"
                else:
                    print("❌ Invalid choice. Please enter 1-3.")
            except (ValueError, KeyboardInterrupt):
                print("❌ Please enter a valid number (1-3).")

    def select_epsilon_level(self) -> List[str]:
        """Interactive epsilon level selection"""
        print("\nSelect epsilon level(s) for adversarial attacks:")
        print("  [1] Minimal (ε = 4/255 ≈ 0.016) - ImageNet standard, barely perceptible")
        print("  [2] Standard (ε = 8/255 ≈ 0.031) - CIFAR-10 standard, imperceptible")
        print("  [3] Moderate (ε = 16/255 ≈ 0.063) - Higher but still invisible perturbations")
        print("  [4] ALL levels (Subtle + Moderate + Strong)")
        print("  [5] Custom epsilon (enter manually)")
        print("\n✅ DIRECT EPSILON CONTROL:")
        print("     • 100% epsilon parameter support")
        print("     • Fast execution (no optimization)")
        print("     • Precise perturbation bounds")
        print("     • Predictable results")
        print()

        while True:
            try:
                choice = int(input("Enter your choice (1-5): "))
                if choice == 1:
                    print("✅ Selected epsilon: Minimal (ε = 4/255 ≈ 0.016)")
                    return ['minimal']
                elif choice == 2:
                    print("✅ Selected epsilon: Standard (ε = 8/255 ≈ 0.031)")
                    return ['standard']
                elif choice == 3:
                    print("✅ Selected epsilon: Moderate (ε = 16/255 ≈ 0.063)")
                    return ['moderate']
                elif choice == 4:
                    print("✅ Selected ALL epsilon levels: [Minimal, Standard, Moderate]")
                    return ['minimal', 'standard', 'moderate']
                elif choice == 5:
                    while True:
                        try:
                            custom_eps = float(input("Enter custom epsilon (0.001-0.3): "))
                            if 0.001 <= custom_eps <= 0.3:
                                epsilon_name = f"custom_{custom_eps:.3f}"
                                print(f"✅ Selected custom epsilon: ε = {custom_eps}")
                                return [epsilon_name]
                            else:
                                print("❌ Epsilon must be between 0.001 and 0.3")
                        except ValueError:
                            print("❌ Please enter a valid number")
                else:
                    print("❌ Invalid choice. Please enter 1-5.")
            except (ValueError, KeyboardInterrupt):
                print("❌ Please enter a valid number (1-5).")

    def select_attack(self, category: str) -> str:
        """Interactive attack selection based on category"""
        if category == "whitebox":
            print("\nSelect whitebox attack type (100% EPSILON SUPPORT):")
            print("  [1] FGSM Attack (Fast Gradient Sign Method) - ε parameter")
            print("  [2] PGD Attack (Projected Gradient Descent) - ε parameter")
            print("  [3] AutoPGD Attack (Auto Projected Gradient Descent) - ε parameter")
            print("  [4] AutoConjugateGradient Attack - ε parameter")
            print("  [5] BasicIterativeMethod Attack - ε parameter")
            print("  [6] ALL WHITEBOX ATTACKS (5 epsilon attacks total)")
            print("\n✅ ALL ATTACKS SUPPORT EPSILON:")
            print("   • Direct epsilon parameter control")
            print("   • Fast execution (no optimization)")
            print("   • Predictable perturbation bounds")
            print("\n💡 Multiple selection examples:")
            print("   • '1,2,3' → FGSM + PGD + AutoPGD attacks")
            print("   • '1,5' → FGSM + BasicIterative attacks")
            print("   • '2' → Single PGD attack")
            print()

            while True:
                try:
                    user_input = input("Enter your choice(s) (1-6 or comma-separated): ").strip()

                    # Handle comma-separated input
                    if ',' in user_input:
                        choices = [int(x.strip()) for x in user_input.split(',')]
                        selected_attacks = []

                        for choice in choices:
                            if choice in self.WHITEBOX_ATTACKS:
                                attack_name = self.WHITEBOX_ATTACKS[choice]
                                selected_attacks.append(attack_name)
                            else:
                                print(f"❌ Invalid choice: {choice}. Please enter 1-6.")
                                selected_attacks = []
                                break

                        if selected_attacks:
                            print(f"✅ Selected attacks: {', '.join(selected_attacks)} (whitebox)")
                            return selected_attacks  # Return list of attacks

                    else:
                        # Handle single choice
                        choice = int(user_input)
                        if choice in self.WHITEBOX_ATTACKS:
                            attack_name = self.WHITEBOX_ATTACKS[choice]
                            print(f"✅ Selected attack: {attack_name} (whitebox)")
                            return attack_name
                        else:
                            print("❌ Invalid choice. Please enter 1-6.")
                except (ValueError, KeyboardInterrupt):
                    print("❌ Please enter a valid number or comma-separated numbers (1-6).")

        elif category == "blackbox":
            print("\nSelect blackbox attack type (100% EPSILON SUPPORT):")
            print("  [1] Square Attack (Score-based black-box) - ε parameter")
            print("  [2] SimBA Attack (Simple black-box) - ε parameter")
            print("  [3] Boundary Attack (Decision boundary) - ε parameter")
            print("  [4] SignOPT Attack (Sign-based optimization) - ε parameter")
            print("  [5] ALL BLACKBOX ATTACKS (4 epsilon attacks total)")
            print("\n✅ ALL ATTACKS SUPPORT EPSILON:")
            print("   • Direct epsilon parameter control")
            print("   • Fast execution (no optimization)")
            print("   • Predictable perturbation bounds")
            print("\n💡 Multiple selection examples:")
            print("   • '1,2,3' → Square + SimBA + Boundary attacks")
            print("   • '1,4' → Square + SignOPT attacks")
            print("   • '2' → Single SimBA attack")
            print()

            while True:
                try:
                    user_input = input("Enter your choice(s) (1-5 or comma-separated): ").strip()

                    # Handle comma-separated input
                    if ',' in user_input:
                        choices = [int(x.strip()) for x in user_input.split(',')]
                        selected_attacks = []

                        for choice in choices:
                            if choice in self.BLACKBOX_ATTACKS:
                                attack_name = self.BLACKBOX_ATTACKS[choice]
                                selected_attacks.append(attack_name)
                            else:
                                print(f"❌ Invalid choice: {choice}. Please enter 1-5.")
                                selected_attacks = []
                                break

                        if selected_attacks:
                            print(f"✅ Selected attacks: {', '.join(selected_attacks)} (blackbox)")
                            return selected_attacks  # Return list of attacks

                    else:
                        # Handle single choice
                        choice = int(user_input)
                        if choice in self.BLACKBOX_ATTACKS:
                            attack_name = self.BLACKBOX_ATTACKS[choice]
                            print(f"✅ Selected attack: {attack_name} (blackbox)")
                            return attack_name
                        else:
                            print("❌ Invalid choice. Please enter 1-5.")
                except (ValueError, KeyboardInterrupt):
                    print("❌ Please enter a valid number or comma-separated numbers (1-5).")
        else:
            # category == "all"
            return "all"

    def load_task_images(self, task_name: str) -> List[str]:
        """Load images for the specified task"""
        images_json = Path("data/processed_images.json")

        if not images_json.exists():
            raise FileNotFoundError(f"Images JSON not found: {images_json}")

        with open(images_json, 'r') as f:
            data = json.load(f)

        if task_name == "all":
            # Get all images from all tasks
            all_images = []
            for task_images in data.values():
                all_images.extend([f"data/clean/{img}" for img in task_images])
            return all_images
        else:
            # Get images for specific task
            if task_name not in data:
                raise ValueError(f"Task '{task_name}' not found in processed images")
            return [f"data/clean/{img}" for img in data[task_name]]

    def check_existing_images(self, category: str) -> bool:
        """Check for existing adversarial images and ask for confirmation"""
        existing_found = False

        whitebox_dir = Path("data/adversarial/whitebox")
        blackbox_dir = Path("data/adversarial/blackbox")
        adversarial_dir = Path("data/adversarial")

        wb_count = 0
        bb_count = 0

        if (category in ["whitebox", "all"]) and whitebox_dir.exists():
            wb_count = len(list(whitebox_dir.glob("**/*.png")))
            if wb_count > 0:
                existing_found = True

        if (category in ["blackbox", "all"]) and blackbox_dir.exists():
            bb_count = len(list(blackbox_dir.glob("**/*.png")))
            if bb_count > 0:
                existing_found = True

        if existing_found:
            print("\n⚠️  EXISTING ADVERSARIAL IMAGES DETECTED")
            print("=" * 50)
            if wb_count > 0:
                print(f"🚀 Whitebox: {wb_count} existing images found")
            if bb_count > 0:
                print(f"🔒 Blackbox: {bb_count} existing images found")
            print("=" * 50)
            print()
            print("📋 REPLACEMENT OPTIONS:")
            print("=" * 50)
            print("🔄 [Y] COMPLETE REPLACEMENT:")
            print("   • Deletes ALL existing adversarial images")
            print("   • Clears ALL existing database entries")
            print("   • Starts completely fresh")
            print("   • Only selected attacks will remain")
            print()
            print("✏️  [N] SELECTIVE OVERWRITE:")
            print("   • Keeps ALL existing adversarial images")
            print("   • Preserves ALL existing database entries")
            print("   • Only overwrites images for selected attacks")
            print("   • Other attacks remain untouched")
            print()
            print("💡 Recommendation: Use [N] to keep successful results and improve specific attacks")
            print("=" * 50)

            while True:
                choice = input("Choose replacement mode - Complete [Y] or Selective [N]: ").strip().lower()
                if choice in ['y', 'yes']:
                    print("✅ Will replace existing adversarial images and reset database metadata")
                    self.is_replacement_run = True

                    # Database-only approach - no JSON file backup needed

                    # Backup adversarial images directory
                    if adversarial_dir.exists() and any(adversarial_dir.iterdir()):
                        backup_dir = Path("data/adversarial_backup")
                        if backup_dir.exists():
                            shutil.rmtree(backup_dir)
                        shutil.copytree(adversarial_dir, backup_dir)
                        print(f"💾 Backed up adversarial images to: {backup_dir}")

                        # Clear existing adversarial images
                        if category in ["whitebox", "all"] and whitebox_dir.exists():
                            shutil.rmtree(whitebox_dir)
                            print("🗑️  Cleared existing whitebox images")
                        if category in ["blackbox", "all"] and blackbox_dir.exists():
                            shutil.rmtree(blackbox_dir)
                            print("🗑️  Cleared existing blackbox images")

                    break
                elif choice in ['n', 'no']:
                    print("❌ Keeping existing adversarial images. New images will be added/overwritten per attack.")
                    self.is_replacement_run = False
                    break
                else:
                    print("❌ Please enter Y or N.")
            print()

        return existing_found

    def extract_metrics_from_log(self, log_content: str) -> Dict[str, Any]:
        """Extract metrics from attack log output"""
        metrics = {
            'epsilon_target': 0.0,
            'epsilon_l_inf': 0.0,
            'mean_perturbation': 0.0,
            'max_perturbation': 0.0,
            'l2_norm': 0.0,
            'l0_norm': 0,
            'total_queries': 0
        }

        # Extract epsilon values - target and post-processed L∞ epsilon
        epsilon_target_match = re.search(r'Target epsilon:\s*([0-9.-]+)', log_content)
        if epsilon_target_match:
            metrics['epsilon_target'] = float(epsilon_target_match.group(1))

        # Extract post-processed Epsilon (L∞) from print_attack_info - this is the ONLY true final epsilon
        epsilon_linf_match = re.search(r'Epsilon \(L∞\):\s*([0-9.-]+)', log_content)
        if epsilon_linf_match:
            metrics['epsilon_l_inf'] = float(epsilon_linf_match.group(1))
        else:
            # If Epsilon (L∞) not found, attack likely failed
            metrics['epsilon_l_inf'] = 0.0

        # Extract mean perturbation
        mean_pert_match = re.search(r'Mean perturbation[^:]*:\s*([0-9.-]+)', log_content)
        if mean_pert_match:
            metrics['mean_perturbation'] = float(mean_pert_match.group(1))

        # Extract max perturbation
        max_pert_match = re.search(r'Max perturbation[^:]*:\s*([0-9.-]+)', log_content)
        if max_pert_match:
            metrics['max_perturbation'] = float(max_pert_match.group(1))

        # Extract L2 norm
        l2_match = re.search(r'L2 norm:\s*([0-9.-]+)', log_content)
        if l2_match:
            metrics['l2_norm'] = float(l2_match.group(1))

        # Extract L0 norm
        l0_match = re.search(r'L0 norm:\s*([0-9]+)', log_content)
        if l0_match:
            metrics['l0_norm'] = int(l0_match.group(1))

        # Extract total queries
        queries_match = re.search(r'Total queries:\s*([0-9]+)', log_content)
        if queries_match:
            metrics['total_queries'] = int(queries_match.group(1))

        return metrics

    def _execute_attacks_with_epsilon_iteration(self, image_list: List[str], attack_type: str,
                                               is_blackbox: bool = False, attack_num_offset: int = 0, trial_number: int = 1):
        """
        Execute attacks across all epsilon levels for given images and attack type.
        Eliminates code duplication across all attack execution patterns.

        Args:
            image_list: List of image paths to process
            attack_type: Type of attack to execute
            is_blackbox: Whether this is a blackbox attack
            attack_num_offset: Starting number for attack progress display
        """
        total_images = len(image_list)

        for epsilon_level in self.config.epsilon_levels:
            epsilon_value = self.get_epsilon_from_level(epsilon_level)
            print(f"\n📊 Epsilon Level: {epsilon_level.title()} (ε = {epsilon_value:.3f})")
            print("=" * 30)

            for i, image_path in enumerate(image_list):
                if not Path(image_path).exists():
                    print(f"❌ Image not found: {image_path}")
                    self.failure_count += 1
                    continue

                # Execute the appropriate attack type
                if is_blackbox:
                    success = self.run_blackbox_attack(image_path, attack_type,
                                                     i+1+attack_num_offset, total_images, epsilon_level, trial_number)
                else:
                    success = self.run_whitebox_attack(image_path, attack_type,
                                                     i+1+attack_num_offset, total_images, epsilon_level, trial_number)

                if success:
                    self.success_count += 1
                else:
                    self.failure_count += 1

    def check_epsilon_success(self, target_epsilon: float, actual_epsilon: float, tolerance_percent: float = 5.0) -> bool:
        """
        Check if actual epsilon is within tolerance of target epsilon

        Args:
            target_epsilon: Target epsilon value (e.g., 0.02)
            actual_epsilon: Actual epsilon achieved
            tolerance_percent: Tolerance percentage (default: 5%)

        Returns:
            True if actual epsilon is within tolerance of target
        """
        if target_epsilon == 0:
            return actual_epsilon == 0

        tolerance = target_epsilon * (tolerance_percent / 100.0)
        deviation = abs(actual_epsilon - target_epsilon)
        return deviation <= tolerance

    def run_whitebox_attack(self, image_path: str, attack_type: str,
                           attack_num: int, total_attacks: int, epsilon_level: str = None, trial_number: int = 1) -> bool:
        """Execute a whitebox attack"""
        # Reset query counter before each attack
        query_counter.reset()

        # Use provided epsilon level or fall back to config default
        current_epsilon_level = epsilon_level if epsilon_level is not None else self.config.epsilon_levels[0]
        epsilon_value = self.get_epsilon_from_level(current_epsilon_level)

        print(f"[{attack_num}/{total_attacks}] Running Universal {attack_type} Attack on {image_path}...")
        print(f"🎯 Epsilon Level: {current_epsilon_level.title()} (ε = {epsilon_value})")
        print(f"🔄 Direct execution (no optimization needed)")

        # Build command - direct epsilon control (no optimization needed)
        cmd = [
            "./venv_MM/bin/python3", "attack_models/white_box_universal.py",
            "--image_path", image_path,
            "--attack_type", attack_type,
            "--epsilon", str(epsilon_value),
            "--trial_number", str(trial_number)
        ]

        try:
            # Execute attack with virtual environment
            import os
            env = os.environ.copy()
            env['PYTHONPATH'] = os.getcwd()

            start_time = datetime.now()
            print("🔍 Executing attack command...")
            print(f"Command: {' '.join(cmd)}")

            # Use Popen for real-time output
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, env=env, bufsize=1, universal_newlines=True)

            output_lines = []
            # Display output in real-time and collect it
            for line in iter(process.stdout.readline, ''):
                line = line.rstrip()
                if line:
                    print(line)  # Display on terminal
                    output_lines.append(line)  # Collect for logging

            process.wait()
            end_time = datetime.now()

            # Create result-like object
            class Result:
                def __init__(self, returncode, stdout):
                    self.returncode = returncode
                    self.stdout = stdout
                    self.stderr = ""

            result = Result(process.returncode, '\n'.join(output_lines))

            execution_time = int((end_time - start_time).total_seconds())

            # Check for process success
            process_success = result.returncode == 0 and not any(
                keyword in result.stdout.lower()
                for keyword in ['failed', 'error', 'exception']
            )

            # Extract metrics first
            log_content = result.stdout + result.stderr
            metrics = self.extract_metrics_from_log(log_content)

            # For epsilon-based attacks, success requires both process completion AND epsilon tolerance
            epsilon_success = self.check_epsilon_success(epsilon_value, metrics.get('epsilon_l_inf', 0))
            success = process_success and epsilon_success

            if process_success and not epsilon_success:
                target_eps = epsilon_value
                actual_eps = metrics.get('epsilon_l_inf', 0)
                tolerance = target_eps * 0.05  # 5% tolerance
                print(f"⚠️  Process completed but epsilon target not met:")
                print(f"   Target: {target_eps:.6f}, Actual: {actual_eps:.6f}, Tolerance: ±{tolerance:.6f}")
                print(f"   Deviation: {abs(actual_eps - target_eps):.6f} (should be ≤ {tolerance:.6f})")

            # Save to database
            self.save_attack_results(
                attack_name=attack_type,
                image_path=image_path,
                execution_time=execution_time,
                success=success,
                metrics=metrics,
                attack_category="White-Box",
                epsilon_level=current_epsilon_level
            )

            if success:
                target_eps = epsilon_value
                actual_eps = metrics.get('epsilon_l_inf', 0)
                print(f"✅ {attack_type} whitebox attack completed successfully!")
                print(f"🎯 Epsilon tolerance achieved: Target={target_eps:.6f}, Actual={actual_eps:.6f} (±5%)")
                return True
            else:
                print(f"❌ {attack_type} whitebox attack failed!")
                return False

        except subprocess.TimeoutExpired:
            print(f"❌ {attack_type} whitebox attack timed out!")
            return False
        except Exception as e:
            print(f"❌ {attack_type} whitebox attack error: {e}")
            return False

    def run_blackbox_attack(self, image_path: str, attack_type: str,
                          attack_num: int, total_attacks: int, epsilon_level: str = None, trial_number: int = 1) -> bool:
        """Execute a blackbox attack"""
        # Reset query counter before each attack
        query_counter.reset()

        # Use provided epsilon level or fall back to config default
        current_epsilon_level = epsilon_level if epsilon_level is not None else self.config.epsilon_levels[0]
        epsilon_value = self.get_epsilon_from_level(current_epsilon_level)

        print(f"[{attack_num}/{total_attacks}] Running Universal {attack_type} Black-Box Attack on {image_path}...")
        print(f"🎯 Epsilon Level: {current_epsilon_level.title()} (ε = {epsilon_value})")
        print(f"🔄 Direct execution (no optimization needed)")

        # Build command - direct epsilon control (no optimization needed)
        cmd = [
            "./venv_MM/bin/python3", "attack_models/black_box_universal.py",
            "--image_path", image_path,
            "--attack_type", attack_type,
            "--epsilon", str(epsilon_value),
            "--trial_number", str(trial_number)
        ]

        try:
            # Execute attack with virtual environment
            import os
            env = os.environ.copy()
            env['PYTHONPATH'] = os.getcwd()

            start_time = datetime.now()
            print("🔍 Executing blackbox attack command...")
            print(f"Command: {' '.join(cmd)}")

            # Use Popen for real-time output
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, env=env, bufsize=1, universal_newlines=True)

            output_lines = []
            # Display output in real-time and collect it
            for line in iter(process.stdout.readline, ''):
                line = line.rstrip()
                if line:
                    print(line)  # Display on terminal
                    output_lines.append(line)  # Collect for logging

            process.wait()
            end_time = datetime.now()

            # Create result-like object
            class Result:
                def __init__(self, returncode, stdout):
                    self.returncode = returncode
                    self.stdout = stdout
                    self.stderr = ""

            result = Result(process.returncode, '\n'.join(output_lines))

            execution_time = int((end_time - start_time).total_seconds())

            # Check for process success
            process_success = result.returncode == 0 and not any(
                keyword in result.stdout.lower()
                for keyword in ['failed', 'error', 'exception']
            )

            # Extract metrics first
            log_content = result.stdout + result.stderr
            metrics = self.extract_metrics_from_log(log_content)

            # For epsilon-based attacks, success requires both process completion AND epsilon tolerance
            epsilon_success = self.check_epsilon_success(epsilon_value, metrics.get('epsilon_l_inf', 0))
            success = process_success and epsilon_success

            if process_success and not epsilon_success:
                target_eps = epsilon_value
                actual_eps = metrics.get('epsilon_l_inf', 0)
                tolerance = target_eps * 0.05  # 5% tolerance
                print(f"⚠️  Process completed but epsilon target not met:")
                print(f"   Target: {target_eps:.6f}, Actual: {actual_eps:.6f}, Tolerance: ±{tolerance:.6f}")
                print(f"   Deviation: {abs(actual_eps - target_eps):.6f} (should be ≤ {tolerance:.6f})")

            # Save to database
            self.save_attack_results(
                attack_name=attack_type,
                image_path=image_path,
                execution_time=execution_time,
                success=success,
                metrics=metrics,
                attack_category="Black-Box",
                epsilon_level=current_epsilon_level
            )

            if success:
                target_eps = epsilon_value
                actual_eps = metrics.get('epsilon_l_inf', 0)
                print(f"✅ {attack_type} blackbox attack completed successfully!")
                print(f"🎯 Epsilon tolerance achieved: Target={target_eps:.6f}, Actual={actual_eps:.6f} (±5%)")
                return True
            else:
                print(f"❌ {attack_type} blackbox attack failed!")
                return False

        except subprocess.TimeoutExpired:
            print(f"❌ {attack_type} blackbox attack timed out!")
            return False
        except Exception as e:
            print(f"❌ {attack_type} blackbox attack error: {e}")
            return False

    def construct_adversarial_image_path(self, original_image_path: str, attack_type: str,
                                        epsilon_value: float, is_blackbox: bool = False) -> str:
        """Construct the adversarial image path based on directory structure"""
        filename = os.path.basename(original_image_path)

        # Extract task from image path (data/clean/task/image.png)
        path_parts = original_image_path.split('/')
        if len(path_parts) >= 3 and path_parts[1] == 'clean':
            task = path_parts[2]
        else:
            # Fallback: extract task from image path
            task = os.path.dirname(original_image_path).split('/')[-1]

        # Build output path: data/adversarial/{blackbox|whitebox}/{attack_type}/eps_{epsilon}/{task}/{filename}
        box_type = 'blackbox' if is_blackbox else 'whitebox'
        eps_dir = f"eps_{epsilon_value:.3f}".replace(".", "")  # 0.050 -> eps_050

        return f"data/adversarial/{box_type}/{attack_type}/{eps_dir}/{task}/{filename}"

    def save_attack_results(self, attack_name: str, image_path: str,
                          execution_time: int, success: bool,
                          metrics: Dict[str, Any], attack_category: str,
                          epsilon_level: str = None):
        """Save attack results to database"""

        # Extract image info
        path_parts = image_path.split('/')
        task_type = path_parts[2] if len(path_parts) > 2 else "unknown"
        image_name = os.path.basename(image_path)

        # Get epsilon value from level
        epsilon_value = self.get_epsilon_from_level(epsilon_level) if epsilon_level else 0.05

        # Construct adversarial image path
        is_blackbox = attack_category == "Black-Box"
        adversarial_image_path = self.construct_adversarial_image_path(
            image_path, attack_name, epsilon_value, is_blackbox
        )

        # Generate execution_id from adversarial_image_path pattern
        # Convert: data/adversarial/whitebox/fgsm/eps_050/chart/20231107140031466140.png
        # To: data_adversarial_whitebox_fgsm_eps_050_chart_20231107140031466140
        execution_id = adversarial_image_path.replace('/', '_').replace('.png', '').replace('.jpg', '').replace('.jpeg', '')

        # Construct execution data for database
        execution_data = {
            'execution_id': execution_id,
            'attack_name': attack_name,
            'attack_category': attack_category,
            'image_path': image_path,
            'adversarial_image_path': adversarial_image_path,
            'image_name': image_name,
            'task_type': task_type,
            'execution_time_seconds': execution_time,
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'epsilon_level': epsilon_level,
                'epsilon_target': metrics.get('epsilon_target', epsilon_value),
                'epsilon_l_inf': metrics.get('epsilon_l_inf', 0.0),
                'mean_perturbation': metrics.get('mean_perturbation'),
                'max_perturbation': metrics.get('max_perturbation'),
                'l2_norm': metrics.get('l2_norm'),
                'l0_norm': metrics.get('l0_norm'),
                'total_queries': metrics.get('total_queries')
            },
            'execution_date': datetime.now().isoformat(),
            'description': "Universal attack results with epsilon control",
            'completed_attacks': self.success_count + self.failure_count + 1
        }

        # Save to database instead of JSON
        self.db.save_attack_execution(execution_data)

        logger.info(f"[{attack_name.upper()}] Saved to DB: Epsilon={metrics.get('epsilon_l_inf', 0.0):.4f}, "
                   f"Mean_Pert={metrics.get('mean_perturbation', 0.0):.4f}, Success={success}")

    def execute_attacks(self, category: str, attack_name: str,
                       task_name: str, image_list: List[str]):
        """Execute the selected attacks"""
        # Define attack lists - 100% EPSILON PARAMETER SUPPORT (9 fast attacks)
        whitebox_attacks = ["fgsm", "pgd", "auto_pgd", "auto_conjugate_gradient", "basic_iterative"]
        blackbox_attacks = ["square", "simba", "boundary", "sign_opt"]

        if category == "all":
            # Run all attacks from both categories
            print("\n🔄 Running ALL WHITEBOX attacks (5 epsilon attacks)...")
            print("=" * 50)

            for current_attack in whitebox_attacks:
                print(f"\n🔄 Running {current_attack} whitebox attacks on all images...")
                self._execute_attacks_with_epsilon_iteration(image_list, current_attack, is_blackbox=False)

            print("\n🔄 Running ALL BLACKBOX attacks (4 epsilon attacks)...")
            print("=" * 50)

            for current_attack in blackbox_attacks:
                print(f"\n🔄 Running {current_attack} blackbox attacks on all images...")
                self._execute_attacks_with_epsilon_iteration(image_list, current_attack, is_blackbox=True)

        elif category == "whitebox":
            if attack_name == "all":
                # Run all whitebox attacks
                for current_attack in whitebox_attacks:
                    print(f"\n🔄 Running {current_attack} whitebox attacks on all images...")
                    print("=" * 50)
                    self._execute_attacks_with_epsilon_iteration(image_list, current_attack, is_blackbox=False)
            elif isinstance(attack_name, list):
                # Run multiple selected whitebox attacks
                for current_attack in attack_name:
                    print(f"\n🔄 Running {current_attack} whitebox attacks on all images...")
                    print("=" * 50)
                    self._execute_attacks_with_epsilon_iteration(image_list, current_attack, is_blackbox=False)
            else:
                # Run single whitebox attack
                print(f"\n🖼️  Processing all images with {attack_name} attack")
                print("=" * 50)
                self._execute_attacks_with_epsilon_iteration(image_list, attack_name, is_blackbox=False)

        elif category == "blackbox":
            if attack_name == "all":
                # Run all blackbox attacks
                for current_attack in blackbox_attacks:
                    print(f"\n🔄 Running {current_attack} blackbox attacks on all images...")
                    print("=" * 50)
                    self._execute_attacks_with_epsilon_iteration(image_list, current_attack, is_blackbox=True)
            elif isinstance(attack_name, list):
                # Run multiple selected blackbox attacks
                for current_attack in attack_name:
                    print(f"\n🔄 Running {current_attack} blackbox attacks on all images...")
                    print("=" * 50)
                    self._execute_attacks_with_epsilon_iteration(image_list, current_attack, is_blackbox=True)
            else:
                # Run single blackbox attack
                print(f"\n🖼️  Processing all images with {attack_name} attack")
                print("=" * 50)
                self._execute_attacks_with_epsilon_iteration(image_list, attack_name, is_blackbox=True)

    def display_summary(self, category: str):
        """Display execution summary"""
        total_attacks = self.success_count + self.failure_count

        print("\n🏆 EXECUTION SUMMARY")
        print("=" * 50)
        print(f"📊 Total attacks processed: {total_attacks}")
        print(f"✅ Successful attacks: {self.success_count}")
        print(f"❌ Failed attacks: {self.failure_count}")

        if total_attacks > 0:
            success_rate = (self.success_count * 100) // total_attacks
            print(f"📈 Success rate: {success_rate}%")

        print("\n🎯 100% EPSILON-BASED ATTACK FRAMEWORK RESULTS:")
        print("✅ Direct epsilon parameter control")
        print("✅ NO optimization needed")
        print("✅ Fast and predictable execution")
        print("✅ Precise perturbation bounds")
        print("🚀 Whitebox attacks: 5 epsilon-based attacks")
        print("🔒 Blackbox attacks: 4 epsilon-based attacks")

        print(f"\n📊 Attack results saved to centralized database: results/centralized_pipeline.db")
        print("🎯 Check adversarial images in:")

        if category in ["whitebox", "all"]:
            print("   - data/adversarial/whitebox/ (5 epsilon whitebox attacks)")
        if category in ["blackbox", "all"]:
            print("   - data/adversarial/blackbox/ (4 epsilon blackbox attacks)")

        print("=" * 50)

    def run(self):
        """Main execution workflow"""
        try:
            # Setup and checks
            self.display_header()
            self.check_environment()

            # Interactive selections
            task_name = self.select_task()
            category = self.select_category()
            attack_name = self.select_attack(category) if category != "all" else "all"

            # Select epsilon level(s)
            selected_epsilon_levels = self.select_epsilon_level()
            self.config.epsilon_levels = selected_epsilon_levels

            # Load images
            image_list = self.load_task_images(task_name)
            total_images = len(image_list)

            if total_images == 0:
                print("❌ No images found for the selected task")
                return 1

            print(f"📊 Found {total_images} images to process")

            # Check existing images
            self.check_existing_images(category)

            # Log execution start
            logger.info(f"🚀 Starting Universal Attack Execution")
            logger.info(f"Task: {task_name}, Category: {category}, Attack: {attack_name}")
            logger.info(f"Total images: {total_images}, Epsilon levels: {self.config.epsilon_levels}")

            # Execute attacks
            self.execute_attacks(category, attack_name, task_name, image_list)

            # Display results
            self.display_summary(category)

            logger.info("🏁 Universal attack execution completed")
            return 0

        except KeyboardInterrupt:
            print("\n❌ Execution interrupted by user")
            return 1
        except Exception as e:
            logger.error(f"❌ Execution failed: {e}")
            return 1

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Universal Attack Runner for White-Box and Black-Box adversarial attacks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/attack_runner.py                    # Interactive mode
    python scripts/attack_runner.py --epsilon 0.1      # Custom epsilon value
    python scripts/attack_runner.py --max-trials 100   # Custom max trials
        """
    )

    parser.add_argument("--epsilon", type=float, default=0.05,
                       help="Target epsilon value (default: 0.05)")
    parser.add_argument("--epsilon-list", type=str, default=None,
                       help="Comma-separated list of epsilon values (e.g., '0.016,0.031,0.063')")
    parser.add_argument("--max-trials-wb", type=int, default=1,
                       help="Max trials for whitebox attacks (default: 1)")
    parser.add_argument("--max-trials-bb", type=int, default=1,
                       help="Max trials for blackbox attacks (default: 1)")

    args = parser.parse_args()

    # Handle epsilon levels
    epsilon_levels = None
    if args.epsilon_list:
        # Parse comma-separated epsilon values
        try:
            epsilon_values = [float(x.strip()) for x in args.epsilon_list.split(',')]
            epsilon_levels = [f"custom_{eps:.3f}" for eps in epsilon_values]
            print(f"📊 Multiple epsilon levels specified: {epsilon_values}")
        except ValueError:
            print("❌ Invalid epsilon list format. Use comma-separated floats (e.g., '0.016,0.031,0.063')")
            return 1

    # Create configuration
    config = AttackConfig(
        epsilon_levels=epsilon_levels,
        max_trials_whitebox=args.max_trials_wb,
        max_trials_blackbox=args.max_trials_bb
    )

    # Run orchestrator
    orchestrator = AttackOrchestrator(config)
    return orchestrator.run()

if __name__ == "__main__":
    exit(main())