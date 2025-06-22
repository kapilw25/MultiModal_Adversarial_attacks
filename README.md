# Evaluating Nano Vision-Language Models (VLMs) Against Cyber Security Attacks: Comprehensive Analysis Across White-Box (Finetuning) and Black-Box (Inference) Attack Scenarios

This repository contains tools for evaluating small (4-bit, 3 Billion parameter) vision-language models (VLMs) under various multi-modal adversarial attacks, focusing on their robustness and performance degradation.

## Overview

This evaluation framework includes:
- Testing infrastructure for lightweight VLMs (4-bit quantized, ~3B parameters)
- Multi-modal adversarial attack implementations to test model robustness
- Performance benchmarking on visual reasoning tasks under attack conditions
- Comparative analysis between original and adversarially perturbed inputs
- Evaluation across diverse visual content including charts, tables, and maps

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv_MM
   source venv_MM/bin/activate
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

Note: The evaluation scripts automatically download required NLTK resources (like WordNet) when needed.

## Evaluation Pipeline

The evaluation pipeline consists of three main components:

### 1. `eval_model.py`

This script generates evaluation results for a specific model on a specific task.

```bash
cd scripts
python eval_model.py
```

Interactive Usage:
- When you run the script, it will prompt you to:
  - Select the engine (1 for GPT-4o or 2 for Qwen25_VL_3B)
  - The script automatically handles the correct imports based on your selection
- The script uses a fixed task ('chart') and sample count (17)
- Results will be saved to `results/{engine}/eval_{engine}_{task}_{random_count}.json`
- The script can be configured to use either original or adversarial images by modifying the image path

### 2. `eval_vqa.py`

This script analyzes the evaluation results and calculates accuracy metrics.

```bash
cd scripts
python eval_vqa.py
```

Interactive Usage:
- When you run the script, it will prompt you to:
  - Select the engine (1 for GPT-4o or 2 for Qwen25_VL_3B)
  - Select the task (chart, table, dashboard, etc.)
- The script will find all evaluation files for the selected model and task
- It will evaluate each file and show a comparison of results
- For adversarial testing, it will show the accuracy drop due to the attack

Functions:
- `evaluator()` - Tests accuracy on 7 tasks (charts, tables, dashboards, flowcharts, relation graphs, floor plans, and visual puzzles)
- `evaluator_map()` - Tests accuracy on simulated maps
- `evaluate_all_files()` - Evaluates and compares all result files for a given model and task

### 3. `attack_model/v2_pgd_attack.py`

This script applies a Projected Gradient Descent (PGD) adversarial attack to images.

```bash
python attack_model/v2_pgd_attack.py --image_path data/test_extracted/chart/image.png --eps 0.03
```

Parameters:
- `--image_path`: Path to the input image
- `--eps`: Maximum perturbation (default: 8/255)
- `--eps_step`: Attack step size (default: 2/255)
- `--max_iter`: Maximum number of iterations (default: 10)

The script will generate an adversarial version of the image and save it to `data/test_extracted_adv/`.

#### Attack Implementation Notes

- The implementation uses a "black-box transfer attack" strategy, employing a pre-trained ResNet50 as a substitute model
- This approach is necessary because most VLMs are accessed through APIs or don't provide gradient access needed for direct attacks
- Transfer attacks rely on the principle that adversarial examples often transfer between different models
- For evaluating proprietary models, this transfer-based approach is practical and effective, as shown by the significant accuracy drop in Qwen25_VL_3B

#### Future Work

- Implement white-box attacks for models where architecture and gradients are accessible
- Evaluate finetuned models under adversarial conditions to measure robustness improvements
- Compare effectiveness of adversarial training techniques in improving VLM robustness
- Explore multi-modal adversarial attacks that target both vision and language components

## Adversarial Evaluation Workflow

To test model robustness against adversarial attacks:

1. Generate adversarial images:
   ```bash
   python attack_model/v2_pgd_attack.py --image_path data/test_extracted/chart/20231114102825506748.png --eps 0.03
   ```

2. Run evaluation on original images:
   ```bash
   # Edit eval_model.py to use original images:
   # img_path = 'data/test_extracted/' + data['image']
   python scripts/eval_model.py
   ```

3. Run evaluation on adversarial images:
   ```bash
   # Edit eval_model.py to use adversarial images:
   # img_path = 'data/test_extracted_adv/' + data['image']
   python scripts/eval_model.py
   ```

4. Compare results:
   ```bash
   python scripts/eval_vqa.py
   ```

## Recent Progress

- Added adversarial attack capabilities using PGD (Projected Gradient Descent)
- Enhanced evaluation scripts to compare performance on original vs. adversarial images
- Implemented automatic detection and comparison of multiple evaluation files
- Moved test files to a dedicated `unit_test` directory for better organization
- Fixed file path issues in evaluation scripts to use relative paths
- Implemented proper environment variable loading for API keys
- Added error handling for missing data
- Added automatic download of NLTK resources when needed
- Successfully ran evaluations on multiple models:
  - Qwen25_VL_3B: 82.35% accuracy on original images, 35.29% on adversarial images
  - GPT-4o: 64.71% accuracy on original images, 70.59% on adversarial images

## Directory Structure

```
Multi-modal-Self-instruct/
├── .env                    # Environment variables (API keys)
├── attack_model/           # Adversarial attack scripts
│   └── v2_pgd_attack.py    # PGD attack implementation
├── data/                   # Dataset files
│   ├── test_extracted/     # Original test images
│   └── test_extracted_adv/ # Adversarial test images
├── results/                # Evaluation results
│   ├── gpt4o/              # GPT-4o results
│   │   ├── eval_chart.json # Input evaluation data
│   │   ├── eval_gpt4o_chart_17.json      # Results on original images
│   │   └── eval_gpt4o_chart_17_adv.json  # Results on adversarial images
│   └── Qwen25_VL_3B/       # Qwen results
│       ├── eval_chart.json # Input evaluation data
│       ├── eval_Qwen25_VL_3B_chart_17.json      # Results on original images
│       └── eval_Qwen25_VL_3B_chart_17_adv.json  # Results on adversarial images
├── scripts/                # Evaluation scripts
│   ├── eval_model.py       # Script to generate model responses
│   ├── eval_vqa.py         # Script to calculate accuracy metrics
│   ├── llm_tools.py        # Utilities for OpenAI API calls
│   └── local_llm_tools.py  # Utilities for local model inference
├── unit_test/              # Test scripts
│   ├── test_local_llm_tools.py  # Tests for local model tools
│   └── test_qwen_model.py       # Tests for Qwen model
└── venv_MM/                # Virtual environment
```

## Usage

1. Set up your environment and API key as described in the Setup section
2. Run `eval_model.py` to generate model responses:
   ```bash
   cd scripts
   python eval_model.py
   # Follow the interactive prompts to select the model
   ```
3. Run `eval_vqa.py` to calculate accuracy metrics:
   ```bash
   cd scripts
   python eval_vqa.py
   # Follow the interactive prompts to select the model and task
   ```

Example workflow for adversarial testing:
```bash
# Generate adversarial image
python attack_model/v2_pgd_attack.py --image_path data/test_extracted/chart/20231114102825506748.png --eps 0.03

# Generate responses with GPT-4o on adversarial images
# (Make sure eval_model.py is set to use adversarial images)
python scripts/eval_model.py
# Select option 1 for GPT-4o

# Evaluate and compare results
python scripts/eval_vqa.py
# Select option 1 for GPT-4o
# Select option 1 for chart task
```

## Adversarial Robustness Results

Initial testing shows interesting differences in model robustness:

| Model | Original Accuracy | Adversarial Accuracy | Accuracy Change |
|-------|------------------|---------------------|----------------|
| GPT-4o | 64.71% | 70.59% | +5.88% |
| Qwen25_VL_3B | 82.35% | 35.29% | -47.06% |

These results suggest that while Qwen25_VL_3B performs better on clean images, GPT-4o shows significantly higher robustness against adversarial attacks.

## Notes

- The evaluation pipeline supports various visual reasoning tasks
- The interactive scripts eliminate the need to manually edit code when switching models
- The pipeline supports both OpenAI API models and local models
- Error handling has been improved to provide helpful messages when issues occur
- File overwrite protection prevents accidental data loss
- Adversarial testing can be extended to other image types and attack methods
