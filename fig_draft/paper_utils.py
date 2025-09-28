#!/usr/bin/env python3
"""
VLM Robustness Paper Utility Script
Author: Research Team
Description: Compile LaTeX files and create distribution packages

This script provides comprehensive LaTeX compilation, validation, and packaging
utilities with better error handling and scalability than bash scripts.
"""

import argparse
import concurrent.futures
import json
import logging
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    MAGENTA = '\033[0;35m'
    CYAN = '\033[0;36m'
    WHITE = '\033[1;37m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class PaperManager:
    """Main class for managing LaTeX paper operations"""

    def __init__(self, script_dir: Optional[Path] = None):
        """Initialize the paper manager with directory paths"""
        self.script_dir = Path(script_dir) if script_dir else Path(__file__).parent
        self.project_root = self.script_dir.parent
        self.logs_dir = self.project_root / "logs"

        # Create logs directory
        self.logs_dir.mkdir(exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Configuration
        self.config = {
            "latex_options": ["-interaction=nonstopmode"],
            "required_files": ["main.tex", "references.bib", "figure1_final.png"],
            "section_files": [
                "sections/abstract.tex",
                "sections/introduction.tex",
                "sections/methodology.tex",
                "sections/results.tex",
                "sections/conclusion.tex"
            ],
            "aux_extensions": [
                "*.aux", "*.bbl", "*.blg", "*.log", "*.out",
                "*.toc", "*.lof", "*.lot", "*.fls",
                "*.fdb_latexmk", "*.synctex.gz"
            ],
            "exclude_from_zip": [
                "*.pdf", "figure0_ranking_all_LMM.png"
            ],
            "exclude_from_overleaf": [
                "*.pdf", "*.aux", "*.bbl", "*.blg", "*.log", "*.out",
                "*.toc", "*.lof", "*.lot", "*.fls", "*.fdb_latexmk",
                "*.synctex.gz", "paper_utils.py", "figure0_ranking_all_LMM.png"
            ]
        }

    def setup_logging(self) -> None:
        """Setup logging configuration"""
        log_file = self.logs_dir / f"paper_utils_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def print_status(self, message: str, color: str = Colors.BLUE) -> None:
        """Print colored status message"""
        print(f"{color}[INFO]{Colors.RESET} {message}")
        self.logger.info(message)

    def print_success(self, message: str) -> None:
        """Print success message"""
        print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} {message}")
        self.logger.info(f"SUCCESS: {message}")

    def print_warning(self, message: str) -> None:
        """Print warning message"""
        print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} {message}")
        self.logger.warning(message)

    def print_error(self, message: str) -> None:
        """Print error message"""
        print(f"{Colors.RED}[ERROR]{Colors.RESET} {message}")
        self.logger.error(message)

    def check_dependencies(self) -> bool:
        """Check if required tools are installed"""
        self.print_status("Checking dependencies...")

        required_tools = ["pdflatex", "bibtex", "zip", "pdfinfo"]
        missing_deps = []

        for tool in required_tools:
            if not shutil.which(tool):
                missing_deps.append(tool)

        if missing_deps:
            self.print_error(f"Missing dependencies: {', '.join(missing_deps)}")
            self.print_error("Please install: sudo apt-get install texlive-latex-base texlive-latex-extra texlive-latex-recommended zip poppler-utils")
            return False

        self.print_success("All dependencies available")
        return True

    def run_command(self, command: List[str], cwd: Optional[Path] = None,
                   capture_output: bool = True, timeout: int = 300) -> subprocess.CompletedProcess:
        """Run a shell command with error handling"""
        try:
            result = subprocess.run(
                command,
                cwd=cwd or self.script_dir,
                capture_output=capture_output,
                text=True,
                timeout=timeout
            )
            return result
        except subprocess.TimeoutExpired:
            self.print_error(f"Command timed out after {timeout}s: {' '.join(command)}")
            raise
        except Exception as e:
            self.print_error(f"Command failed: {' '.join(command)}, Error: {str(e)}")
            raise

    def clean_aux_files(self) -> bool:
        """Clean auxiliary LaTeX files"""
        self.print_status("Cleaning auxiliary files...")

        try:
            for pattern in self.config["aux_extensions"]:
                for file_path in self.script_dir.glob(pattern):
                    file_path.unlink()
                    self.logger.debug(f"Removed: {file_path}")

            self.print_success("Auxiliary files cleaned")
            return True
        except Exception as e:
            self.print_error(f"Failed to clean auxiliary files: {str(e)}")
            return False

    def validate_file_structure(self) -> Tuple[bool, Dict[str, bool]]:
        """Validate project file structure"""
        self.print_status("Validating file structure...")

        validation_results = {}
        all_valid = True

        # Check required files
        for file_name in self.config["required_files"]:
            file_path = self.script_dir / file_name
            exists = file_path.exists()
            validation_results[file_name] = exists

            if exists:
                self.print_success(f"✓ {file_name}")
            else:
                self.print_error(f"✗ {file_name} (MISSING)")
                all_valid = False

        # Check sections directory
        sections_dir = self.script_dir / "sections"
        if sections_dir.exists() and sections_dir.is_dir():
            self.print_success("✓ sections/ directory")
            validation_results["sections_dir"] = True

            section_count = len(list(sections_dir.glob("*.tex")))
            self.print_status(f"Found {section_count} section files")
        else:
            self.print_error("✗ sections/ directory (MISSING)")
            validation_results["sections_dir"] = False
            all_valid = False

        # Check figures directory
        figures_dir = self.script_dir / "figures"
        if figures_dir.exists() and figures_dir.is_dir():
            self.print_success("✓ figures/ directory")
            validation_results["figures_dir"] = True

            figure_count = len(list(figures_dir.glob("*.png")))
            self.print_status(f"Found {figure_count} figure files")

            # Log figure files
            figure_files = sorted(figures_dir.glob("*.png"))
            for fig in figure_files:
                self.logger.debug(f"Figure: {fig.name}")
        else:
            self.print_error("✗ figures/ directory (MISSING)")
            validation_results["figures_dir"] = False
            all_valid = False

        return all_valid, validation_results

    def compile_latex_document(self, max_retries: int = 2) -> bool:
        """Compile the main LaTeX document with bibliography"""
        self.print_status("Compiling main.tex...")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.logs_dir / f"main_compilation_{timestamp}.log"

        compilation_steps = [
            ("pdflatex (1st pass)", ["pdflatex"] + self.config["latex_options"] + ["main.tex"]),
            ("bibtex", ["bibtex", "main"]),
            ("pdflatex (2nd pass)", ["pdflatex"] + self.config["latex_options"] + ["main.tex"]),
            ("pdflatex (3rd pass)", ["pdflatex"] + self.config["latex_options"] + ["main.tex"])
        ]

        try:
            with open(log_file, 'w') as log:
                log.write(f"LaTeX Compilation Log - {datetime.now()}\n")
                log.write("=" * 50 + "\n\n")

                for step_name, command in compilation_steps:
                    self.print_status(f"Running {step_name}...")
                    log.write(f"\n--- {step_name.upper()} ---\n")

                    for attempt in range(max_retries + 1):
                        try:
                            result = self.run_command(command, timeout=300)

                            # Write output to log
                            log.write(f"STDOUT:\n{result.stdout}\n")
                            log.write(f"STDERR:\n{result.stderr}\n")
                            log.write(f"Return code: {result.returncode}\n\n")

                            if result.returncode == 0:
                                self.print_success(f"{step_name} completed")
                                break
                            elif step_name == "bibtex" and result.returncode != 0:
                                self.print_warning(f"{step_name} had warnings (this may be normal)")
                                break
                            else:
                                if attempt < max_retries:
                                    self.print_warning(f"{step_name} failed, retrying ({attempt + 1}/{max_retries})...")
                                    continue
                                else:
                                    self.print_error(f"{step_name} failed after {max_retries + 1} attempts")
                                    return False

                        except Exception as e:
                            if attempt < max_retries:
                                self.print_warning(f"{step_name} failed with exception, retrying: {str(e)}")
                                continue
                            else:
                                self.print_error(f"{step_name} failed with exception: {str(e)}")
                                return False

            # Check if PDF was generated
            pdf_path = self.script_dir / "main.pdf"
            if pdf_path.exists():
                # Get PDF info
                try:
                    result = self.run_command(["pdfinfo", str(pdf_path)])
                    if result.returncode == 0:
                        pages_line = [line for line in result.stdout.split('\n') if 'Pages:' in line]
                        pages = pages_line[0].split(':')[1].strip() if pages_line else "unknown"
                    else:
                        pages = "unknown"
                except:
                    pages = "unknown"

                pdf_size = f"{pdf_path.stat().st_size / (1024*1024):.1f}MB"
                self.print_success(f"PDF generated successfully: {pages} pages, {pdf_size}")

                # Backup PDF to logs directory
                backup_path = self.logs_dir / f"main_{timestamp}.pdf"
                shutil.copy2(pdf_path, backup_path)
                self.print_success("PDF backup saved to logs directory")

                return True
            else:
                self.print_error("PDF file not generated")
                return False

        except Exception as e:
            self.print_error(f"Compilation failed with exception: {str(e)}")
            return False

        finally:
            self.print_status(f"Compilation log saved to: {log_file}")

    def test_section_file(self, section_file: str) -> bool:
        """Test compilation of individual section file"""
        section_path = self.script_dir / section_file

        if not section_path.exists():
            self.print_warning(f"{section_file} - File not found")
            return False

        self.print_status(f"Testing {section_file}...")

        # Create minimal test document
        test_content = f"""\\documentclass{{article}}
\\usepackage{{amsmath,amssymb,amsfonts}}
\\usepackage{{graphicx}}
\\usepackage{{url}}
\\begin{{document}}
\\input{{{section_file}}}
\\end{{document}}"""

        test_file = self.script_dir / "test_section.tex"
        try:
            test_file.write_text(test_content)

            result = self.run_command(
                ["pdflatex"] + self.config["latex_options"] + ["test_section.tex"],
                timeout=60
            )

            if result.returncode == 0:
                self.print_success(f"{section_file} - OK")
                return True
            else:
                self.print_warning(f"{section_file} - Has issues (may be due to missing references)")
                return False

        except Exception as e:
            self.print_warning(f"{section_file} - Test failed: {str(e)}")
            return False

        finally:
            # Clean up test files
            for pattern in ["test_section.*"]:
                for file_path in self.script_dir.glob(pattern):
                    try:
                        file_path.unlink()
                    except:
                        pass

    def test_all_sections(self) -> Dict[str, bool]:
        """Test all section files in parallel"""
        self.print_status("Testing individual section files...")

        results = {}

        # Use ThreadPoolExecutor for parallel testing
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_section = {
                executor.submit(self.test_section_file, section): section
                for section in self.config["section_files"]
            }

            for future in concurrent.futures.as_completed(future_to_section):
                section = future_to_section[future]
                try:
                    results[section] = future.result()
                except Exception as e:
                    self.print_error(f"Section test failed for {section}: {str(e)}")
                    results[section] = False

        # Log results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        test_log = self.logs_dir / f"section_tests_{timestamp}.log"

        with open(test_log, 'w') as log:
            log.write(f"Section File Tests - {datetime.now()}\n")
            log.write("=" * 50 + "\n\n")

            for section, passed in results.items():
                status = "PASS" if passed else "ISSUES"
                log.write(f"{section}: {status}\n")

        self.print_status(f"Section test log saved to: {test_log}")
        return results

    def create_zip_package(self, include_pdf: bool = False, overleaf_only: bool = False) -> Optional[Path]:
        """Create ZIP package for Overleaf upload"""
        if overleaf_only:
            self.print_status("Creating lightweight ZIP package for Overleaf compilation...")
        else:
            self.print_status("Creating ZIP package for Overleaf...")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if overleaf_only:
            zip_name = f"VLM_Paper_Overleaf_{timestamp}.zip"
        else:
            zip_name = f"VLM_Robustness_Paper_{timestamp}.zip"
        zip_path = self.logs_dir / zip_name

        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add all files except excluded ones
                for file_path in self.script_dir.rglob('*'):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(self.script_dir.parent)

                        # Check if file should be excluded
                        should_exclude = False

                        # Exclude auxiliary files
                        for pattern in self.config["aux_extensions"]:
                            if file_path.match(pattern):
                                should_exclude = True
                                break

                        # Exclude specific files based on package type
                        if overleaf_only:
                            # For Overleaf, exclude more files for lighter package
                            for pattern in self.config["exclude_from_overleaf"]:
                                if file_path.match(pattern) or file_path.name == pattern:
                                    should_exclude = True
                                    break
                        elif not include_pdf:
                            for pattern in self.config["exclude_from_zip"]:
                                if file_path.match(pattern) or file_path.name == pattern:
                                    should_exclude = True
                                    break

                        if not should_exclude:
                            zipf.write(file_path, relative_path)
                            self.logger.debug(f"Added to ZIP: {relative_path}")

            zip_size = f"{zip_path.stat().st_size / (1024*1024):.1f}MB"
            self.print_success(f"ZIP package created: {zip_name} ({zip_size})")
            self.print_status(f"Location: {zip_path}")

            # Show contents (first 20 files)
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                file_list = zipf.namelist()
                self.print_status("Package contents (first 20 files):")
                for file_name in sorted(file_list)[:20]:
                    print(f"  {file_name}")
                if len(file_list) > 20:
                    print(f"  ... and {len(file_list) - 20} more files")

            # Create latest symlink
            if overleaf_only:
                latest_link = self.logs_dir / "VLM_Paper_Overleaf_latest.zip"
                symlink_name = "VLM_Paper_Overleaf_latest.zip"
            else:
                latest_link = self.logs_dir / "VLM_Robustness_Paper_latest.zip"
                symlink_name = "VLM_Robustness_Paper_latest.zip"

            if latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(zip_name)
            self.print_success(f"Latest package linked as: {symlink_name}")

            return zip_path

        except Exception as e:
            self.print_error(f"Failed to create ZIP package: {str(e)}")
            return None

    def run_full_pipeline(self) -> bool:
        """Run the complete validation, compilation, and packaging pipeline"""
        self.print_status("Running full pipeline...")

        pipeline_steps = [
            ("Dependency Check", self.check_dependencies),
            ("File Structure Validation", lambda: self.validate_file_structure()[0]),
            ("Clean Auxiliary Files", self.clean_aux_files),
            ("Compile Main Document", self.compile_latex_document),
            ("Test Section Files", lambda: all(self.test_all_sections().values())),
            ("Create ZIP Package", lambda: self.create_zip_package() is not None)
        ]

        for step_name, step_func in pipeline_steps:
            self.print_status(f"Pipeline: {step_name}...")
            try:
                if not step_func():
                    self.print_error(f"Pipeline failed at: {step_name}")
                    return False
                self.print_success(f"Pipeline: {step_name} completed")
            except Exception as e:
                self.print_error(f"Pipeline failed at {step_name}: {str(e)}")
                return False

        self.print_success("Full pipeline completed successfully!")
        return True


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="VLM Robustness Paper Utility Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python paper_utils.py compile      # Just compile the main document
  python paper_utils.py all          # Full validation and packaging
  python paper_utils.py package      # Create ZIP for Overleaf upload
  python paper_utils.py overleaf     # Create lightweight ZIP for Overleaf
  python paper_utils.py test         # Test individual section files
        """
    )

    parser.add_argument(
        'action',
        choices=['compile', 'test', 'validate', 'package', 'overleaf', 'clean', 'all', 'help'],
        default='help',
        nargs='?',
        help='Action to perform'
    )

    parser.add_argument(
        '--include-pdf',
        action='store_true',
        help='Include PDF files in ZIP package'
    )

    parser.add_argument(
        '--overleaf-only',
        action='store_true',
        help='Create lightweight package for Overleaf (excludes PDF and auxiliary files)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.action == 'help':
        parser.print_help()
        return 0

    # Initialize paper manager
    try:
        paper_manager = PaperManager()

        if args.verbose:
            paper_manager.logger.setLevel(logging.DEBUG)

        print("=" * 60)
        print(f"{Colors.BOLD}VLM Robustness Paper Utility Script{Colors.RESET}")
        print("=" * 60)
        print()

        # Execute requested action
        success = True

        if args.action == 'compile':
            success = (paper_manager.validate_file_structure()[0] and
                      paper_manager.clean_aux_files() and
                      paper_manager.compile_latex_document())

        elif args.action == 'test':
            results = paper_manager.test_all_sections()
            success = all(results.values())

        elif args.action == 'validate':
            success = paper_manager.validate_file_structure()[0]

        elif args.action == 'package':
            success = (paper_manager.validate_file_structure()[0] and
                      paper_manager.create_zip_package(args.include_pdf, args.overleaf_only) is not None)

        elif args.action == 'overleaf':
            success = (paper_manager.validate_file_structure()[0] and
                      paper_manager.create_zip_package(include_pdf=False, overleaf_only=True) is not None)

        elif args.action == 'clean':
            success = paper_manager.clean_aux_files()

        elif args.action == 'all':
            success = paper_manager.run_full_pipeline()

        print()
        print("=" * 60)
        if success:
            print(f"{Colors.GREEN}Operation completed successfully!{Colors.RESET}")
            print(f"Check logs in: {paper_manager.logs_dir}")
        else:
            print(f"{Colors.RED}Operation failed!{Colors.RESET}")
            print(f"Check logs in: {paper_manager.logs_dir}")
        print("=" * 60)

        return 0 if success else 1

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Operation cancelled by user{Colors.RESET}")
        return 1

    except Exception as e:
        print(f"{Colors.RED}Unexpected error: {str(e)}{Colors.RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())