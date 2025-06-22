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

The evaluation pipeline consists of four main components:

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
- When you run the script, it will prompt you to select the engine (1 for GPT-4o or 2 for Qwen25_VL_3B)
- The script uses a fixed task ('chart') to match `eval_model.py`
- It will find all evaluation files for the selected model and task
- It will evaluate each file and show a comparison of results
- For adversarial testing, it will show the accuracy drop due to each attack type (PGD and FGSM)

Functions:
- `evaluator()` - Tests accuracy on chart tasks
- `evaluate_all_files()` - Evaluates and compares all result files for a given model and task

### 3. `attack_models/black_box_attacks/v2_pgd_attack.py`

This script applies a Projected Gradient Descent (PGD) adversarial attack to images.

```bash
python attack_models/black_box_attacks/v2_pgd_attack.py --image_path data/test_extracted/chart/image.png --eps 0.03
```

Parameters:
- `--image_path`: Path to the input image
- `--eps`: Maximum perturbation (default: 8/255)
- `--eps_step`: Attack step size (default: 2/255)
- `--max_iter`: Maximum number of iterations (default: 10)

The script will generate an adversarial version of the image and save it to `data/test_extracted_adv/`.

### 4. `attack_models/black_box_attacks/v3_fgsm_attack.py`

This script applies a Fast Gradient Sign Method (FGSM) adversarial attack to images.

```bash
python attack_models/black_box_attacks/v3_fgsm_attack.py --image_path data/test_extracted/chart/image.png --eps 0.03
```

Parameters:
- `--image_path`: Path to the input image
- `--eps`: Maximum perturbation (default: 8/255)

The script will generate an adversarial version of the image and save it to `data/test_extracted_adv_fgsm/`.

#### Attack Implementation Notes

- Both attacks use a "black-box transfer attack" strategy, employing a pre-trained ResNet50 as a substitute model
- PGD is an iterative attack that refines the adversarial perturbation over multiple steps
- FGSM is a single-step attack that creates adversarial examples in one go (faster but generally less effective)
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

1. Generate adversarial images using PGD:
   ```bash
   python attack_models/black_box_attacks/v2_pgd_attack.py --image_path data/test_extracted/chart/20231114102825506748.png --eps 0.03
   ```

2. Generate adversarial images using FGSM:
   ```bash
   python attack_models/black_box_attacks/v3_fgsm_attack.py --image_path data/test_extracted/chart/20231114102825506748.png --eps 0.03
   ```

3. Run evaluation on original images:
   ```bash
   # Edit eval_model.py to use original images:
   # img_path = 'data/test_extracted/' + data['image']
   python scripts/eval_model.py
   ```

4. Run evaluation on PGD adversarial images:
   ```bash
   # Edit eval_model.py to use PGD adversarial images:
   # img_path = 'data/test_extracted_adv/' + data['image']
   python scripts/eval_model.py
   ```

5. Run evaluation on FGSM adversarial images:
   ```bash
   # Edit eval_model.py to use FGSM adversarial images:
   # img_path = 'data/test_extracted_adv_fgsm/' + data['image']
   python scripts/eval_model.py
   ```

6. Compare results:
   ```bash
   python scripts/eval_vqa.py
   ```

## Recent Progress

- Added FGSM (Fast Gradient Sign Method) attack implementation for comparison with PGD
- Simplified evaluation scripts to focus on chart tasks for more direct comparisons
- Enhanced `eval_vqa.py` to automatically detect and compare results from different attack types
- Added adversarial attack capabilities using PGD (Projected Gradient Descent)
- Enhanced evaluation scripts to compare performance on original vs. adversarial images
- Implemented automatic detection and comparison of multiple evaluation files
- Moved test files to a dedicated `unit_test` directory for better organization
- Fixed file path issues in evaluation scripts to use relative paths
- Implemented proper environment variable loading for API keys
- Added error handling for missing data
- Added automatic download of NLTK resources when needed

## Directory Structure

```
Multi-modal-Self-instruct/
├── .env                    # Environment variables (API keys)
├── attack_models/          # Adversarial attack scripts
│   └── black_box_attacks/  # Black-box attack implementations
│       ├── v1_image_distortions.py  # Basic image distortions
│       ├── v2_pgd_attack.py         # PGD attack implementation
│       └── v3_fgsm_attack.py        # FGSM attack implementation
├── data/                   # Dataset files
│   ├── test_extracted/        # Original test images
│   ├── test_extracted_adv/    # PGD adversarial images
│   └── test_extracted_adv_fgsm/ # FGSM adversarial images
├── results/                # Evaluation results
│   ├── gpt4o/              # GPT-4o results
│   │   ├── eval_chart.json # Input evaluation data
│   │   ├── eval_gpt4o_chart_17.json         # Results on original images
│   │   ├── eval_gpt4o_chart_17_adv.json     # Results on PGD adversarial images
│   │   └── eval_gpt4o_chart_17_adv_fgsm.json # Results on FGSM adversarial images
│   └── Qwen25_VL_3B/       # Qwen results
│       ├── eval_chart.json # Input evaluation data
│       ├── eval_Qwen25_VL_3B_chart_17.json         # Results on original images
│       ├── eval_Qwen25_VL_3B_chart_17_adv.json     # Results on PGD adversarial images
│       └── eval_Qwen25_VL_3B_chart_17_adv_fgsm.json # Results on FGSM adversarial images
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
   # Follow the interactive prompts to select the model
   ```

Example workflow for comparing attack methods:
```bash
# Generate PGD adversarial image
python attack_models/black_box_attacks/v2_pgd_attack.py --image_path data/test_extracted/chart/20231114102825506748.png --eps 0.03

# Generate FGSM adversarial image
python attack_models/black_box_attacks/v3_fgsm_attack.py --image_path data/test_extracted/chart/20231114102825506748.png --eps 0.03

# Evaluate and compare results
python scripts/eval_vqa.py
# Select the model to evaluate
```

## Adversarial Robustness Results

Testing shows interesting differences in model robustness against different attack types:

### Comprehensive Results Table

| Model | Image Type | Accuracy | Accuracy Change |
|-------|------------|----------|----------------|
| GPT-4o | Original | 64.71% | - |
| GPT-4o | PGD Adversarial | 70.59% | +5.88% |
| GPT-4o | FGSM Adversarial | 64.71% | 0.00% |
| Qwen25_VL_3B | Original | 82.35% | - |
| Qwen25_VL_3B | PGD Adversarial | 35.29% | -47.06% |
| Qwen25_VL_3B | FGSM Adversarial | 41.18% | -41.18% |

### Raw Evaluation Output

#### GPT-4o Results:
```
=== ACCURACY COMPARISON ===
Adversarial (PGD) (eval_gpt4o_chart_17_adv.json): 70.59%
Adversarial (FGSM) (eval_gpt4o_chart_17_adv_fgsm.json): 64.71%
Original (eval_gpt4o_chart_17.json): 64.71%

Accuracy drop due to PGD attack: -5.88%
Accuracy drop due to FGSM attack: 0.00%
```

#### Qwen25_VL_3B Results:
```
=== ACCURACY COMPARISON ===
Original (eval_Qwen25_VL_3B_chart_17.json): 82.35%
Adversarial (PGD) (eval_Qwen25_VL_3B_chart_17_adv.json): 35.29%
Adversarial (FGSM) (eval_Qwen25_VL_3B_chart_17_adv_fgsm.json): 41.18%

Accuracy drop due to PGD attack: 47.06%
Accuracy drop due to FGSM attack: 41.18%
```

These results reveal several key insights:

1. **Baseline Performance**: Qwen25_VL_3B performs significantly better on original images (82.35% vs 64.71%)

2. **Adversarial Robustness**: 
   - GPT-4o shows remarkable robustness to both attack types, with PGD attacks actually improving performance
   - Qwen25_VL_3B shows significant vulnerability to both attack types, with accuracy dropping by over 40%

3. **Attack Effectiveness**:
   - For Qwen25_VL_3B, the PGD attack was more effective than FGSM (as expected)
   - For GPT-4o, neither attack was effective at reducing performance

4. **Comparative Analysis**:
   - While Qwen25_VL_3B outperforms GPT-4o on clean images, GPT-4o is substantially more robust to adversarial examples
   - The difference in robustness is dramatic - GPT-4o's performance is completely unaffected or even improved by attacks that cut Qwen25_VL_3B's performance in half

## Notes

- The evaluation pipeline now supports multiple attack types for comprehensive robustness testing
- The interactive scripts eliminate the need to manually edit code when switching models
- The pipeline supports both OpenAI API models and local models
- Error handling has been improved to provide helpful messages when issues occur
- File overwrite protection prevents accidental data loss
- Adversarial testing can be extended to other image types and attack methods
