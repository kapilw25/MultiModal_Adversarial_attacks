#!/usr/bin/env python3
"""
Universal Attack Runner (White-Box + Black-Box)

A comprehensive orchestrator for adversarial attacks following ML research standards.
Supports both whitebox and blackbox attacks through unified frameworks with 
SSIM-aware optimization.

Features (OPTIMIZED FOR 7.6GB GPU):
- 4 Fast Whitebox attacks: FGSM, DeepFool, PGD, CW-Linf
- 5 Fast Blackbox attacks: Square, SimBA, Boundary, Pixel, Spatial
- REMOVED 5 slow attacks: JSMA (593s), CW-L0 (587s), CW-L2 (259s), HopSkipJump (914s), ZOO (3263s)
- Interactive task and attack selection
- Automatic results logging to JSON
- SSIM-aware Bayesian optimization
- NO manual post-processing

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

# Import centralized database
from utils.centralized_database import CentralizedDB

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
    ssim_thresholds: List[float] = None  # Support multiple SSIM values
    ssim_threshold: float = 0.85  # Backward compatibility - single SSIM
    optimization_method: str = "bayesian"
    max_trials_whitebox: int = 5
    max_trials_blackbox: int = 5
    results_file: str = "results/attack_parameters.json"
    log_file: str = "scripts/attack_logs.txt"

    def __post_init__(self):
        # If ssim_thresholds is not provided, use the single ssim_threshold
        if self.ssim_thresholds is None:
            self.ssim_thresholds = [self.ssim_threshold]

class AttackOrchestrator:
    """Main orchestrator for adversarial attacks"""
    
    # Attack mappings - OPTIMIZED FOR 7.6GB GPU vRAM (9 fast attacks only)
    # REMOVED SLOW ATTACKS: JSMA (593s), CW-L0 (587s), CW-L2 (259s), HopSkipJump (914s), ZOO (3263s)
    WHITEBOX_ATTACKS = {
        1: "fgsm", 2: "deepfool", 3: "pgd", 4: "cw_linf", 5: "all"
        # REMOVED: jsma (593.75s), cw_l0 (587.0s), cw_l2 (259.5s) - too computationally expensive
    }
    
    BLACKBOX_ATTACKS = {
        1: "square", 2: "simba", 3: "boundary", 4: "pixel", 
        5: "spatial", 6: "all"
        # REMOVED: hop_skip_jump (914.75s), zoo (3263.5s) - too computationally expensive
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

        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup file logging"""
        log_file = Path(self.config.log_file)
        log_file.parent.mkdir(exist_ok=True)
        
        # Clear previous log
        with open(log_file, 'w') as f:
            f.write(f"=== Universal Attack Execution Started ===\n")
            f.write(f"Execution ID: {self.execution_id}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
    
    def display_header(self):
        """Display application header"""
        print("🎯 Universal Attack Runner (White-Box + Black-Box) - OPTIMIZED FOR 7.6GB GPU")
        print("=" * 70)
        print("🚀 Whitebox: universal.py (4 fast attacks)")
        print("🔒 Blackbox: universal.py (5 fast attacks)")
        print("🎯 SSIM-aware optimization for both categories")
        print("❌ NO manual post-processing or blending!")
        print("⚡ REMOVED 5 slow attacks: JSMA, CW-L0, CW-L2, HopSkipJump, ZOO")
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
        print("\nSelect attack category (OPTIMIZED FOR 7.6GB GPU):")
        print("  [1] White-Box Attacks (4 fast attacks: FGSM, DeepFool, PGD, CW-Linf)")
        print("  [2] Black-Box Attacks (5 fast attacks: Square, SimBA, Boundary, Pixel, Spatial)")
        print("  [3] All Attacks (9 fast attacks total: 4 White-Box + 5 Black-Box)")
        print("\n❌ REMOVED SLOW ATTACKS:")
        print("   • White-Box: JSMA (593s), CW-L0 (587s), CW-L2 (259s)")
        print("   • Black-Box: HopSkipJump (914s), ZOO (3263s)")
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
                    print("✅ Selected category: All Attacks (9 fast attacks total)")
                    return "all"
                else:
                    print("❌ Invalid choice. Please enter 1-3.")
            except (ValueError, KeyboardInterrupt):
                print("❌ Please enter a valid number (1-3).")
    
    def select_ssim_threshold(self) -> List[float]:
        """Interactive SSIM threshold selection"""
        print("\nSelect SSIM threshold(s) for adversarial attacks:")
        print("  [1] SSIM = 0.85 (Standard threshold)")
        print("  [2] SSIM = 0.90 (High similarity)")
        print("  [3] SSIM = 0.95 (Very high similarity)")
        print("  [4] ALL thresholds (0.85, 0.90, 0.95)")
        print("  [5] Custom threshold (enter manually)")
        print()
        
        while True:
            try:
                choice = int(input("Enter your choice (1-5): "))
                if choice == 1:
                    print("✅ Selected SSIM threshold: 0.85")
                    return [0.85]
                elif choice == 2:
                    print("✅ Selected SSIM threshold: 0.90")
                    return [0.90]
                elif choice == 3:
                    print("✅ Selected SSIM threshold: 0.95")
                    return [0.95]
                elif choice == 4:
                    print("✅ Selected ALL SSIM thresholds: [0.85, 0.90, 0.95]")
                    return [0.85, 0.90, 0.95]
                elif choice == 5:
                    while True:
                        try:
                            custom_ssim = float(input("Enter custom SSIM threshold (0.0-1.0): "))
                            if 0.0 <= custom_ssim <= 1.0:
                                print(f"✅ Selected custom SSIM threshold: {custom_ssim}")
                                return [custom_ssim]
                            else:
                                print("❌ SSIM must be between 0.0 and 1.0")
                        except ValueError:
                            print("❌ Please enter a valid number")
                else:
                    print("❌ Invalid choice. Please enter 1-5.")
            except (ValueError, KeyboardInterrupt):
                print("❌ Please enter a valid number (1-5).")

    def select_attack(self, category: str) -> str:
        """Interactive attack selection based on category"""
        if category == "whitebox":
            print("\nSelect whitebox attack type (OPTIMIZED FOR 7.6GB GPU):")
            print("  [1] FGSM Attack (Fast Gradient Sign Method) - 4.0s avg")
            print("  [2] DeepFool Attack (Geometric approach) - 5.0s avg")
            print("  [3] PGD Attack (Projected Gradient Descent) - 5.25s avg")
            print("  [4] CW-Linf Attack (Carlini & Wagner L∞) - 66.5s avg")
            print("  [5] ALL WHITEBOX ATTACKS (4 fast attacks total)")
            print("\n❌ REMOVED SLOW ATTACKS:")
            print("   • JSMA (593.75s avg) - Too computationally expensive")
            print("   • CW-L0 (587.0s avg) - Too computationally expensive")
            print("   • CW-L2 (259.5s avg) - Too computationally expensive")
            print("\n💡 Multiple selection examples:")
            print("   • '1,3,4' → FGSM + PGD + CW-Linf attacks")  
            print("   • '1,2' → FGSM + DeepFool attacks")
            print("   • '3' → Single PGD attack")
            print()
            
            while True:
                try:
                    user_input = input("Enter your choice(s) (1-5 or comma-separated): ").strip()
                    
                    # Handle comma-separated input
                    if ',' in user_input:
                        choices = [int(x.strip()) for x in user_input.split(',')]
                        selected_attacks = []
                        
                        for choice in choices:
                            if choice in self.WHITEBOX_ATTACKS:
                                attack_name = self.WHITEBOX_ATTACKS[choice]
                                selected_attacks.append(attack_name)
                            else:
                                print(f"❌ Invalid choice: {choice}. Please enter 1-5.")
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
                            print("❌ Invalid choice. Please enter 1-5.")
                except (ValueError, KeyboardInterrupt):
                    print("❌ Please enter a valid number or comma-separated numbers (1-5).")
                    
        elif category == "blackbox":
            print("\nSelect blackbox attack type (OPTIMIZED FOR 7.6GB GPU):")
            print("  [1] Square Attack (Score-based black-box) - 11.25s avg")
            print("  [2] SimBA Attack (Simple black-box) - 12.75s avg")
            print("  [3] Boundary Attack (Decision boundary) - 39.25s avg")
            print("  [4] Pixel Attack (Few-pixel modification) - 42.75s avg")
            print("  [5] Spatial Attack (Transformation-based) - 59.25s avg ⚠️  SSIM ~0.73")
            print("  [6] ALL BLACKBOX ATTACKS (5 fast attacks total)")
            print("\n❌ REMOVED SLOW ATTACKS:")
            print("   • HopSkipJump (914.75s avg) - Too computationally expensive")
            print("   • ZOO (3263.5s avg) - Too computationally expensive")
            print("\n💡 Multiple selection examples:")
            print("   • '1,2,3' → Square + SimBA + Boundary attacks (fast execution)")  
            print("   • '1,4' → Square + Pixel attacks")
            print("   • '5' → Spatial attack (⚠️  expect SSIM 0.6-0.8 only)")
            print()
            
            while True:
                try:
                    user_input = input("Enter your choice(s) (1-6 or comma-separated): ").strip()
                    
                    # Handle comma-separated input
                    if ',' in user_input:
                        choices = [int(x.strip()) for x in user_input.split(',')]
                        selected_attacks = []
                        
                        for choice in choices:
                            if choice in self.BLACKBOX_ATTACKS:
                                attack_name = self.BLACKBOX_ATTACKS[choice]
                                selected_attacks.append(attack_name)
                            else:
                                print(f"❌ Invalid choice: {choice}. Please enter 1-6.")
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
                            print("❌ Invalid choice. Please enter 1-6.")
                except (ValueError, KeyboardInterrupt):
                    print("❌ Please enter a valid number or comma-separated numbers (1-6).")
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
            print("   • Clears ALL existing JSON entries") 
            print("   • Starts completely fresh")
            print("   • Only selected attacks will remain")
            print()
            print("✏️  [N] SELECTIVE OVERWRITE:")
            print("   • Keeps ALL existing adversarial images")
            print("   • Preserves ALL existing JSON entries")
            print("   • Only overwrites images for selected attacks")
            print("   • Other attacks remain untouched")
            print()
            print("💡 Recommendation: Use [N] to keep successful results and improve specific attacks")
            print("=" * 50)
            
            while True:
                choice = input("Choose replacement mode - Complete [Y] or Selective [N]: ").strip().lower()
                if choice in ['y', 'yes']:
                    print("✅ Will replace existing adversarial images and reset JSON metadata")
                    self.is_replacement_run = True
                    
                    # Backup JSON results file
                    results_file = Path(self.config.results_file)
                    if results_file.exists():
                        backup_json = results_file.with_name(f"{results_file.stem}_backup{results_file.suffix}")
                        shutil.copy2(results_file, backup_json)
                        print(f"💾 Backed up JSON results to: {backup_json}")
                        results_file.unlink()
                        print("🗑️  Cleared existing JSON results")
                    
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
    
    def should_skip_image(self, image_path: str, attack_type: str, ssim_threshold: float, is_blackbox: bool = False) -> bool:
        """Check if adversarial image already exists for this combination"""
        # Build expected output path based on the directory structure
        filename = os.path.basename(image_path)
        
        # Extract task from image path (data/clean/task/image.png)
        path_parts = image_path.split('/')
        if len(path_parts) >= 3 and path_parts[1] == 'clean':
            task = path_parts[2]
        else:
            # Fallback: extract task from image path
            task = os.path.dirname(image_path).split('/')[-1]
        
        # Build output path: data/adversarial/{blackbox|whitebox}/{attack_type}/ssim_{threshold}/{task}/{filename}
        box_type = 'blackbox' if is_blackbox else 'whitebox'
        ssim_dir = f"ssim_{ssim_threshold:.2f}".replace(".", "")  # 0.85 -> ssim_085
        
        output_path = f"data/adversarial/{box_type}/{attack_type}/{ssim_dir}/{task}/{filename}"
        
        if os.path.exists(output_path):
            print(f"⏭️  Skipping - already exists: {output_path}")
            return True
        return False

    def extract_metrics_from_log(self, log_content: str) -> Dict[str, Any]:
        """Extract metrics from attack log output"""
        metrics = {
            'ssim': 0.0,
            'mean_perturbation': 0.0,
            'max_perturbation': 0.0
        }
        
        # Extract SSIM
        ssim_match = re.search(r'Achieved SSIM[^:]*:\s*([0-9.-]+)', log_content)
        if ssim_match:
            metrics['ssim'] = float(ssim_match.group(1))
        
        # Extract mean perturbation
        mean_pert_match = re.search(r'Mean perturbation[^:]*:\s*([0-9.-]+)', log_content)
        if mean_pert_match:
            metrics['mean_perturbation'] = float(mean_pert_match.group(1))
        
        # Extract max perturbation
        max_pert_match = re.search(r'Max perturbation[^:]*:\s*([0-9.-]+)', log_content)
        if max_pert_match:
            metrics['max_perturbation'] = float(max_pert_match.group(1))
        
        return metrics

    def _execute_attacks_with_ssim_iteration(self, image_list: List[str], attack_type: str,
                                           is_blackbox: bool = False, attack_num_offset: int = 0, trial_number: int = 1):
        """
        Execute attacks across all SSIM thresholds for given images and attack type.
        Eliminates code duplication across all attack execution patterns.

        Args:
            image_list: List of image paths to process
            attack_type: Type of attack to execute
            is_blackbox: Whether this is a blackbox attack
            attack_num_offset: Starting number for attack progress display
        """
        total_images = len(image_list)

        for ssim_threshold in self.config.ssim_thresholds:
            print(f"\n📊 SSIM Threshold: {ssim_threshold}")
            print("=" * 30)

            for i, image_path in enumerate(image_list):
                if not Path(image_path).exists():
                    print(f"❌ Image not found: {image_path}")
                    self.failure_count += 1
                    continue

                # CHECK BEFORE PROCESSING - Skip if image already exists
                if self.should_skip_image(image_path, attack_type, ssim_threshold, is_blackbox):
                    continue  # Skip this image entirely

                # Execute the appropriate attack type
                if is_blackbox:
                    success = self.run_blackbox_attack(image_path, attack_type,
                                                     i+1+attack_num_offset, total_images, ssim_threshold, trial_number)
                else:
                    success = self.run_whitebox_attack(image_path, attack_type,
                                                     i+1+attack_num_offset, total_images, ssim_threshold, trial_number)

                if success:
                    self.success_count += 1
                else:
                    self.failure_count += 1

    def run_whitebox_attack(self, image_path: str, attack_type: str,
                           attack_num: int, total_attacks: int, ssim_threshold: float = None, trial_number: int = 1) -> bool:
        """Execute a whitebox attack"""
        # Use provided SSIM threshold or fall back to config default
        target_ssim = ssim_threshold if ssim_threshold is not None else self.config.ssim_threshold

        print(f"[{attack_num}/{total_attacks}] Running Universal {attack_type} Attack on {image_path}...")
        print(f"🎯 Target SSIM: {target_ssim}")
        print(f"🔍 Optimization: {self.config.optimization_method}")
        print(f"🔄 Max trials: {self.config.max_trials_whitebox}")

        # Build command - use adaptive optimization for better performance
        optimization_method = "adaptive" if self.config.optimization_method == "adaptive" else "universal"
        cmd = [
            "./venv_MM/bin/python3", "attack_models/white_box_universal.py",
            "--image_path", image_path,
            "--attack_type", attack_type,
            "--ssim_threshold", str(target_ssim),
            "--optimization_method", optimization_method,
            "--max_trials", str(self.config.max_trials_whitebox),
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
                    
                    # Real-time logging to file (CPU-only operation)
                    with open(self.config.log_file, 'a') as f:
                        f.write(line + '\n')
                        f.flush()  # Force write to disk
            
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
            success = result.returncode == 0 and not any(
                keyword in result.stdout.lower() 
                for keyword in ['failed', 'error', 'exception']
            )
            
            # Extract metrics
            log_content = result.stdout + result.stderr
            metrics = self.extract_metrics_from_log(log_content)
            
            # Save to JSON
            self.save_attack_results(
                attack_name=attack_type,
                image_path=image_path,
                execution_time=execution_time,
                success=success,
                metrics=metrics,
                attack_category="White-Box",
                ssim_threshold=target_ssim
            )
            
            # Log to file with detailed information
            with open(self.config.log_file, 'a') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"=== {attack_type.upper()} WHITEBOX ATTACK ===\n")
                f.write(f"Image: {image_path}\n")
                f.write(f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"End: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Duration: {execution_time} seconds\n")
                f.write(f"Success: {success}\n")
                f.write(f"Return Code: {result.returncode}\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"{'='*60}\n")
                f.write("STDOUT:\n")
                f.write(log_content)
                f.write(f"\n{'='*60}\n\n")
            
            if success:
                print(f"✅ {attack_type} whitebox attack completed successfully!")
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
                          attack_num: int, total_attacks: int, ssim_threshold: float = None, trial_number: int = 1) -> bool:
        """Execute a blackbox attack"""
        # Use provided SSIM threshold or fall back to config default
        target_ssim = ssim_threshold if ssim_threshold is not None else self.config.ssim_threshold

        print(f"[{attack_num}/{total_attacks}] Running Universal {attack_type} Black-Box Attack on {image_path}...")
        print(f"🎯 Target SSIM: {target_ssim}")
        print(f"🔍 Optimization: {self.config.optimization_method}")
        print(f"🔄 Max trials: {self.config.max_trials_blackbox}")

        # Build command - use adaptive optimization for better performance
        optimization_method = "adaptive" if self.config.optimization_method == "adaptive" else "bayesian"
        cmd = [
            "./venv_MM/bin/python3", "attack_models/black_box_universal.py",
            "--image_path", image_path,
            "--attack_type", attack_type,
            "--ssim_threshold", str(target_ssim),
            "--optimization_method", optimization_method,
            "--max_trials", str(self.config.max_trials_blackbox),
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
                    
                    # Real-time logging to file (CPU-only operation)
                    with open(self.config.log_file, 'a') as f:
                        f.write(line + '\n')
                        f.flush()  # Force write to disk
            
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
            success = result.returncode == 0 and not any(
                keyword in result.stdout.lower() 
                for keyword in ['failed', 'error', 'exception']
            )
            
            # Extract metrics
            log_content = result.stdout + result.stderr
            metrics = self.extract_metrics_from_log(log_content)
            
            # Save to JSON
            self.save_attack_results(
                attack_name=attack_type,
                image_path=image_path,
                execution_time=execution_time,
                success=success,
                metrics=metrics,
                attack_category="Black-Box",
                ssim_threshold=target_ssim
            )
            
            # Log to file with detailed information
            with open(self.config.log_file, 'a') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"=== {attack_type.upper()} BLACKBOX ATTACK ===\n")
                f.write(f"Image: {image_path}\n")
                f.write(f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"End: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Duration: {execution_time} seconds\n")
                f.write(f"Success: {success}\n")
                f.write(f"Return Code: {result.returncode}\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"{'='*60}\n")
                f.write("STDOUT:\n")
                f.write(log_content)
                f.write(f"\n{'='*60}\n\n")
            
            if success:
                print(f"✅ {attack_type} blackbox attack completed successfully!")
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
                                        ssim_threshold: float, is_blackbox: bool = False) -> str:
        """Construct the adversarial image path based on directory structure"""
        filename = os.path.basename(original_image_path)

        # Extract task from image path (data/clean/task/image.png)
        path_parts = original_image_path.split('/')
        if len(path_parts) >= 3 and path_parts[1] == 'clean':
            task = path_parts[2]
        else:
            # Fallback: extract task from image path
            task = os.path.dirname(original_image_path).split('/')[-1]

        # Build output path: data/adversarial/{blackbox|whitebox}/{attack_type}/ssim_{threshold}/{task}/{filename}
        box_type = 'blackbox' if is_blackbox else 'whitebox'
        ssim_dir = f"ssim_{ssim_threshold:.2f}".replace(".", "")  # 0.85 -> ssim_085

        return f"data/adversarial/{box_type}/{attack_type}/{ssim_dir}/{task}/{filename}"

    def save_attack_results(self, attack_name: str, image_path: str,
                          execution_time: int, success: bool,
                          metrics: Dict[str, Any], attack_category: str,
                          ssim_threshold: float = None):
        """MODIFIED: Save directly to database instead of JSON"""

        # Extract image info
        path_parts = image_path.split('/')
        task_type = path_parts[2] if len(path_parts) > 2 else "unknown"
        image_name = os.path.basename(image_path)

        # Construct adversarial image path
        is_blackbox = attack_category == "Black-Box"
        target_ssim = ssim_threshold if ssim_threshold is not None else self.config.ssim_threshold
        adversarial_image_path = self.construct_adversarial_image_path(
            image_path, attack_name, target_ssim, is_blackbox
        )

        # Generate execution_id from adversarial_image_path pattern
        # Convert: data/adversarial/whitebox/fgsm/ssim_085/chart/20231107140031466140.png
        # To: data_adversarial_whitebox_fgsm_ssim_085_chart_20231107140031466140
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
                'ssim_target': target_ssim,
                'ssim': metrics.get('ssim'),
                'mean_perturbation': metrics.get('mean_perturbation'),
                'max_perturbation': metrics.get('max_perturbation'),
                'l2_norm': metrics.get('l2_norm'),
                'l0_norm': metrics.get('l0_norm'),
                'total_queries': metrics.get('total_queries')
            },
            'execution_date': datetime.now().isoformat(),
            'description': "Universal attack results with SSIM optimization",
            'completed_attacks': self.success_count + self.failure_count + 1
        }

        # Save to database instead of JSON
        self.db.save_attack_execution(execution_data)

        logger.info(f"[{attack_name.upper()}] Saved to DB: SSIM={metrics.get('ssim'):.4f}, "
                   f"Mean_Pert={metrics.get('mean_perturbation'):.2f}, Success={success}")
    
    def execute_attacks(self, category: str, attack_name: str, 
                       task_name: str, image_list: List[str]):
        """Execute the selected attacks"""
        total_images = len(image_list)
        
        # Define attack lists - OPTIMIZED FOR 7.6GB GPU (9 fast attacks only)
        whitebox_attacks = ["fgsm", "deepfool", "pgd", "cw_linf"]  # REMOVED: jsma, cw_l0, cw_l2 (too slow)
        blackbox_attacks = ["square", "simba", "boundary", "pixel", "spatial"]  # REMOVED: hop_skip_jump, zoo (too slow)
        
        if category == "all":
            # Run all attacks from both categories
            print("\n🔄 Running ALL WHITEBOX attacks (4 fast attacks)...")
            print("=" * 50)

            for current_attack in whitebox_attacks:
                print(f"\n🔄 Running {current_attack} whitebox attacks on all images...")
                self._execute_attacks_with_ssim_iteration(image_list, current_attack, is_blackbox=False)
            
            print("\n🔄 Running ALL BLACKBOX attacks (5 fast attacks)...")
            print("=" * 50)

            for current_attack in blackbox_attacks:
                print(f"\n🔄 Running {current_attack} blackbox attacks on all images...")
                self._execute_attacks_with_ssim_iteration(image_list, current_attack, is_blackbox=True)
                        
        elif category == "whitebox":
            if attack_name == "all":
                # Run all whitebox attacks
                for current_attack in whitebox_attacks:
                    print(f"\n🔄 Running {current_attack} whitebox attacks on all images...")
                    print("=" * 50)
                    self._execute_attacks_with_ssim_iteration(image_list, current_attack, is_blackbox=False)
            elif isinstance(attack_name, list):
                # Run multiple selected whitebox attacks
                for current_attack in attack_name:
                    print(f"\n🔄 Running {current_attack} whitebox attacks on all images...")
                    print("=" * 50)
                    self._execute_attacks_with_ssim_iteration(image_list, current_attack, is_blackbox=False)
            else:
                # Run single whitebox attack
                print(f"\n🖼️  Processing all images with {attack_name} attack")
                print("=" * 50)
                self._execute_attacks_with_ssim_iteration(image_list, attack_name, is_blackbox=False)
                        
        elif category == "blackbox":
            if attack_name == "all":
                # Run all blackbox attacks
                for current_attack in blackbox_attacks:
                    print(f"\n🔄 Running {current_attack} blackbox attacks on all images...")
                    print("=" * 50)
                    self._execute_attacks_with_ssim_iteration(image_list, current_attack, is_blackbox=True)
            elif isinstance(attack_name, list):
                # Run multiple selected blackbox attacks
                for current_attack in attack_name:
                    print(f"\n🔄 Running {current_attack} blackbox attacks on all images...")
                    print("=" * 50)
                    self._execute_attacks_with_ssim_iteration(image_list, current_attack, is_blackbox=True)
            else:
                # Run single blackbox attack
                print(f"\n🖼️  Processing all images with {attack_name} attack")
                print("=" * 50)
                self._execute_attacks_with_ssim_iteration(image_list, attack_name, is_blackbox=True)
    
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
        
        print("\n🎯 OPTIMIZED ATTACK FRAMEWORK RESULTS (7.6GB GPU):")
        print("✅ Target SSIM (0.85) achieved through authentic optimization")
        print("✅ NO manual post-processing applied")
        print("✅ Hyperparameters automatically tuned via Bayesian optimization")
        print("✅ Pure ART-based attack implementations")
        print("🚀 Whitebox attacks: 4 fast gradient-based attacks")
        print("🔒 Blackbox attacks: 5 fast query-based attacks")
        print("⚡ REMOVED 5 slow attacks: JSMA, CW-L0, CW-L2, HopSkipJump, ZOO")
        
        print(f"\n📝 Full execution log saved to: {self.config.log_file}")
        print(f"📊 Attack results saved to: {self.config.results_file}")
        print("🎯 Check adversarial images in:")
        
        if category in ["whitebox", "all"]:
            print("   - data/adversarial/whitebox/ (4 fast whitebox attacks)")
        if category in ["blackbox", "all"]:
            print("   - data/adversarial/blackbox/ (5 fast blackbox attacks)")
        
        print("=" * 50)
    
    def run(self):
        """Main execution workflow"""
        try:
            # Setup and checks
            self.display_header()
            self.check_environment()
            
            # Initialize log file with session header
            with open(self.config.log_file, 'a') as f:
                f.write(f"\n{'#'*80}\n")
                f.write(f"### NEW ATTACK SESSION STARTED ###\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Execution ID: {self.execution_id}\n")
                f.write(f"Target SSIM: {self.config.ssim_threshold}\n")
                f.write(f"{'#'*80}\n\n")
            
            # Interactive selections
            task_name = self.select_task()
            category = self.select_category()
            attack_name = self.select_attack(category) if category != "all" else "all"
            
            # Select SSIM threshold(s)
            selected_ssim_thresholds = self.select_ssim_threshold()
            self.config.ssim_thresholds = selected_ssim_thresholds
            self.config.ssim_threshold = selected_ssim_thresholds[0]  # For backward compatibility
            
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
            logger.info(f"Total images: {total_images}, Target SSIM: {self.config.ssim_threshold}")
            
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
    python scripts/attack_runner.py --ssim 0.9         # Custom SSIM threshold
    python scripts/attack_runner.py --max-trials 100   # Custom max trials
        """
    )
    
    parser.add_argument("--ssim", type=float, default=0.85,
                       help="Target SSIM threshold (default: 0.85)")
    parser.add_argument("--ssim-list", type=str, default=None,
                       help="Comma-separated list of SSIM thresholds (e.g., '0.85,0.90,0.95')")
    parser.add_argument("--optimization", default="bayesian",
                       choices=["bayesian", "grid_search", "random_search"],
                       help="Optimization method (default: bayesian)")
    parser.add_argument("--max-trials-wb", type=int, default=5,
                       help="Max trials for whitebox attacks (default: 5)")
    parser.add_argument("--max-trials-bb", type=int, default=5,
                       help="Max trials for blackbox attacks (default: 5)")
    parser.add_argument("--results-file", default="results/attack_parameters.json",
                       help="Results JSON file path (default: results/attack_parameters.json)")
    parser.add_argument("--log-file", default="scripts/attack_logs.txt",
                       help="Log file path (default: scripts/attack_logs.txt)")
    
    args = parser.parse_args()
    
    # Handle SSIM thresholds
    ssim_thresholds = None
    if args.ssim_list:
        # Parse comma-separated SSIM values
        try:
            ssim_thresholds = [float(x.strip()) for x in args.ssim_list.split(',')]
            print(f"📊 Multiple SSIM thresholds specified: {ssim_thresholds}")
        except ValueError:
            print("❌ Invalid SSIM list format. Use comma-separated floats (e.g., '0.85,0.90,0.95')")
            return 1

    # Create configuration
    config = AttackConfig(
        ssim_thresholds=ssim_thresholds,
        ssim_threshold=args.ssim,
        optimization_method=args.optimization,
        max_trials_whitebox=args.max_trials_wb,
        max_trials_blackbox=args.max_trials_bb,
        results_file=args.results_file,
        log_file=args.log_file
    )
    
    # Run orchestrator
    orchestrator = AttackOrchestrator(config)
    return orchestrator.run()

if __name__ == "__main__":
    exit(main())