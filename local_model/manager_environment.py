#!/usr/bin/env python3
"""
Environment Manager for Multi-VLM Pipeline
Handles different transformer versions for different VLM models
"""

import os
import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ModelEnvironment:
    name: str
    venv_path: str
    requirements_file: str
    python_exe: str
    description: str

class EnvironmentManager:
    """
    Manages multiple virtual environments for different VLM models
    that require incompatible transformer versions
    """
    
    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent.parent
        self.environments_dir = self.base_dir / "environments"
        self.requirements_dir = self.environments_dir / "requirements"
        
        # Create directories if they don't exist
        self.environments_dir.mkdir(exist_ok=True)
        self.requirements_dir.mkdir(exist_ok=True)
        
        # Define model environment mappings
        self.model_environments = self._initialize_environments()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _initialize_environments(self) -> Dict[str, ModelEnvironment]:
        """Initialize model environment configurations"""
        return {
            'florence2': ModelEnvironment(
                name='florence2',
                venv_path=str(self.base_dir / 'venv_florence'),
                requirements_file='requirements_step1.txt,requirements_step2.txt',  # Two-step installation
                python_exe='',  # Will be set after venv creation
                description='Florence-2 models (transformers==4.44.2 via 2-step install)'
            ),
            'default': ModelEnvironment(
                name='default',
                venv_path=str(self.base_dir / 'venv_MM'),
                requirements_file='requirements.txt',  # Latest transformers for all other VLMs
                python_exe='',
                description='All VLMs except Florence-2 (latest transformers)'
            )
        }
    
    def get_model_environment(self, model_name: str) -> str:
        """Map model name to environment name"""
        model_mapping = {
            # Florence models - Only these need special environment
            'florence2': 'florence2',
            'florence2_base': 'florence2',
            'florence2_large': 'florence2',
            'florence-2-base': 'florence2',
            'florence-2-large': 'florence2',
            
            # All other VLMs use default (latest transformers)
            'qwen2.5-vl-3b-instruct_4bit': 'default',
            'qwen2.5-vl-7b-instruct-4bit': 'default',
            'qwen2-vl-2b-instruct_4bit': 'default',
            'qwen25_3b': 'default',
            'qwen25_7b': 'default',
            'qwen2_2b': 'default',
            'gemma-3-4b-it_4bit': 'default',
            'paligemma-3b-mix-224_4bit': 'default',
            'deepseek-vl-1.3b-chat_4bit': 'default',
            'deepseek-vl-7b-chat_4bit': 'default',
            'smolvlm2-256m-video-instruct': 'default',
            'smolvlm2-500m-video-instruct': 'default',
            'smolvlm2-2.2b-instruct': 'default',
            'glm-edge-v-2b': 'default',
            'internvl3-1b': 'default',
            'internvl3-2b': 'default',
            'internvl2.5-4b': 'default',
            'moondream2-2b': 'default',
            'phi-3.5-vision-instruct-4bit': 'default',
            'internvl25_4b': 'default',
            'internvl3_1b': 'default', 
            'internvl3_2b': 'default',
            'deepseek1_1pt3b': 'default',
            'deepseek1_7b': 'default',
            'gemma3_4b': 'default',
            'glmedge_2b': 'default',
            'moondream2_2b': 'default',
            'paligemma_3b': 'default',
            'phi3pt5_vision_4b': 'default',
            'smolvlm2': 'default'
        }
        
        return model_mapping.get(model_name.lower(), 'default')
    
    def create_environment(self, env_name: str) -> bool:
        """Create virtual environment for specific model class"""
        if env_name not in self.model_environments:
            self.logger.error(f"Unknown environment: {env_name}")
            return False
        
        env_config = self.model_environments[env_name]
        
        try:
            self.logger.info(f"Creating environment: {env_config.description}")
            
            # Create virtual environment
            subprocess.run([
                'python3', '-m', 'venv', env_config.venv_path
            ], check=True)
            
            # Set python executable path
            env_config.python_exe = os.path.join(env_config.venv_path, 'bin', 'python')
            
            # Upgrade pip
            subprocess.run([
                env_config.python_exe, '-m', 'pip', 'install', '--upgrade', 'pip'
            ], check=True)
            
            # Install requirements - handle multiple files for Florence-2
            if ',' in env_config.requirements_file:
                # Florence-2 two-step installation
                req_files = env_config.requirements_file.split(',')
                for req_file in req_files:
                    req_path = self.base_dir / req_file.strip()
                    if req_path.exists():
                        self.logger.info(f"Installing requirements from {req_path}")
                        subprocess.run([
                            env_config.python_exe, '-m', 'pip', 'install', '-r', str(req_path)
                        ], check=True)
            else:
                # Single requirements file
                req_path = self.base_dir / env_config.requirements_file
                if req_path.exists():
                    self.logger.info(f"Installing requirements from {req_path}")
                    subprocess.run([
                        env_config.python_exe, '-m', 'pip', 'install', '-r', str(req_path)
                    ], check=True)
            
            self.logger.info(f"Successfully created environment: {env_name}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to create environment {env_name}: {e}")
            return False
    
    def run_model_script(self, model_name: str, script_path: str, args: List[str] = None) -> Tuple[str, str, int]:
        """
        Run a script in the appropriate environment for the given model
        
        Args:
            model_name: Name of the model
            script_path: Path to the Python script to run
            args: List of command line arguments
            
        Returns:
            Tuple of (stdout, stderr, return_code)
        """
        env_name = self.get_model_environment(model_name)
        env_config = self.model_environments[env_name]
        
        # Ensure environment exists
        if not os.path.exists(env_config.venv_path):
            self.logger.info(f"Environment {env_name} doesn't exist. Creating...")
            if not self.create_environment(env_name):
                return "", f"Failed to create environment {env_name}", 1
        
        # Set python executable if not set
        if not env_config.python_exe:
            env_config.python_exe = os.path.join(env_config.venv_path, 'bin', 'python')
        
        # Build command
        cmd = [env_config.python_exe, script_path]
        if args:
            cmd.extend(args)
        
        try:
            self.logger.info(f"Running {model_name} in {env_name} environment: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.base_dir)
            )
            
            return result.stdout, result.stderr, result.returncode
            
        except Exception as e:
            self.logger.error(f"Error running script: {e}")
            return "", str(e), 1
    
    def install_package(self, env_name: str, package: str) -> bool:
        """Install a package in specific environment"""
        if env_name not in self.model_environments:
            self.logger.error(f"Unknown environment: {env_name}")
            return False
        
        env_config = self.model_environments[env_name]
        
        if not env_config.python_exe:
            env_config.python_exe = os.path.join(env_config.venv_path, 'bin', 'python')
        
        try:
            subprocess.run([
                env_config.python_exe, '-m', 'pip', 'install', package
            ], check=True)
            
            self.logger.info(f"Successfully installed {package} in {env_name}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to install {package} in {env_name}: {e}")
            return False
    
    def environment_exists(self, env_name: str) -> bool:
        """Check if environment exists and is ready"""
        if env_name not in self.model_environments:
            return False
        
        env_config = self.model_environments[env_name]
        venv_exists = os.path.exists(env_config.venv_path)
        python_exe = os.path.join(env_config.venv_path, 'bin', 'python')
        python_exists = os.path.exists(python_exe)
        
        return venv_exists and python_exists
    
    def list_environments(self) -> Dict[str, Dict]:
        """List all environments and their status"""
        status = {}
        
        for name, env_config in self.model_environments.items():
            exists = os.path.exists(env_config.venv_path)
            python_exe = os.path.join(env_config.venv_path, 'bin', 'python')
            python_exists = os.path.exists(python_exe)
            
            status[name] = {
                'description': env_config.description,
                'venv_path': env_config.venv_path,
                'requirements_file': env_config.requirements_file,
                'exists': exists,
                'python_executable': python_exe if python_exists else 'Not found',
                'ready': exists and python_exists
            }
        
        return status
    
    def cleanup_environment(self, env_name: str) -> bool:
        """Remove virtual environment"""
        if env_name not in self.model_environments:
            self.logger.error(f"Unknown environment: {env_name}")
            return False
        
        env_config = self.model_environments[env_name]
        
        try:
            if os.path.exists(env_config.venv_path):
                import shutil
                shutil.rmtree(env_config.venv_path)
                self.logger.info(f"Removed environment: {env_name}")
            else:
                self.logger.info(f"Environment {env_name} doesn't exist")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove environment {env_name}: {e}")
            return False

    def check_and_setup_environments(self):
        """Check environment status and offer to create missing ones"""
        status = self.list_environments()
        
        missing_envs = [name for name, info in status.items() if not info['ready']]
        
        if missing_envs:
            print("\n" + "="*60)
            print("ENVIRONMENT SETUP")
            print("="*60)
            print("The following environments are missing or incomplete:")
            for env_name in missing_envs:
                print(f"  - {env_name}: {status[env_name]['description']}")
            
            print(f"\nEnvironments will be created automatically when needed.")
            print("This may take several minutes for first-time setup.")
            
            return status
        else:
            print("All environments are ready!")
            return status

    def ensure_environment_exists(self, env_name: str) -> bool:
        """Ensure environment exists, create if missing"""
        if self.environment_exists(env_name):
            return True
        
        self.logger.info(f"Environment {env_name} not found. Creating...")
        if self.create_environment(env_name):
            self.logger.info(f"Environment {env_name} created successfully!")
            return True
        else:
            self.logger.error(f"Failed to create environment {env_name}!")
            return False

    def run_model_with_env_check(self, model_name: str, script_path: str, args: list = None) -> tuple:
        """Run model script with automatic environment creation if needed"""
        env_name = self.get_model_environment(model_name)
        
        # Ensure environment exists
        if not self.ensure_environment_exists(env_name):
            return "", f"Failed to create/access environment {env_name}", 1
        
        # Run the script
        return self.run_model_script(model_name, script_path, args)

def main():
    """Test the environment manager"""
    manager = EnvironmentManager()
    
    print("Environment Manager Status:")
    print("=" * 50)
    
    status = manager.list_environments()
    for name, info in status.items():
        print(f"{name}:")
        print(f"  Description: {info['description']}")
        print(f"  Ready: {info['ready']}")
        print(f"  Path: {info['venv_path']}")
        print()
    
    # Check and setup environments
    manager.check_and_setup_environments()

if __name__ == "__main__":
    main()