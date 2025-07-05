# Evaluating Nano Vision-Language Models' (VLMs) Robustness Against Cyber Security Attacks

This repository contains tools for evaluating small (4-bit, 3 Billion parameter) vision-language models (VLMs) under various multi-modal adversarial attacks, focusing on their robustness and performance degradation.

# Attack Models Architecture

The repository is organized to focus on true black-box attacks:

```
attack_models/
└── true_black_box_attacks/    # Attacks that don't require gradient access
    ├── v0_attack_utils.py     # Shared utility functions with perceptual constraints
    └── v10_square_attack.py   # Square Attack implementation
```

## True Black-Box Attack Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                   TRUE BLACK-BOX ATTACK WORKFLOW                    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────┐
│ STEP 1: GENERATE ADVERSARIAL│
│         EXAMPLES           │
└───────────────┬─────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ attack_models/true_black_box_attacks/v10_square_attack.py           │
│                                                                     │
│ ┌───────────────┐    ┌───────────────┐    ┌───────────────────────┐ │
│ │ Load Original │ → │ Apply Square   │ → │ Apply Perceptual      │ │
│ │ Image         │    │ Attack        │    │ Constraints           │ │
│ └───────────────┘    └───────────────┘    └───────────────────────┘ │
│                                                                     │
│ Parameters:                                                         │
│ --image_path: Original image                                        │
│ --eps: Perturbation magnitude                                       │
│ --norm: Norm type (inf or 2)                                        │
│ --max_iter: Maximum iterations                                      │
│ --ssim_threshold: Structural similarity threshold                   │
│ --lpips_threshold: Perceptual similarity threshold                  │
│ --clip_threshold: Semantic similarity threshold                     │
└─────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ OUTPUT: data/test_BB_square/chart/20231114102825506748.png          │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 2: EVALUATE MODEL ON   │
│         ADVERSARIAL IMAGES  │
└───────────────┬─────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ scripts/eval_model.py                                               │
│                                                                     │
│ ┌───────────────┐    ┌───────────────┐    ┌───────────────────────┐ │
│ │ Select Model  │ → │ Load Images    │ → │ Generate Predictions   │ │
│ │ (Qwen25_VL_3B)│    │ (Adversarial) │    │ for Each Image        │ │
│ └───────────────┘    └───────────────┘    └───────────────────────┘ │
│                                                                     │
│ Model Loading Path:                                                 │
│ scripts/eval_model.py                                               │
│    ↓                                                                │
│ scripts/local_llm_tools.py                                          │
│    ↓                                                                │
│ local_model/model_classes.py → create_model()                       │
│    ↓                                                                │
│ local_model/qwen_model.py → QwenVLModelWrapper                      │
└─────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ OUTPUT: results/Qwen25_VL_3B/eval_Qwen25_VL_3B_chart_17_BB_square.json │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 3: CALCULATE ACCURACY  │
│         METRICS            │
└───────────────┬─────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ scripts/eval_vqa.py                                                 │
│                                                                     │
│ ┌───────────────┐    ┌───────────────┐    ┌───────────────────────┐ │
│ │ Load Results  │ → │ Calculate      │ → │ Display Accuracy       │ │
│ │ JSON Files    │    │ Accuracy      │    │ Comparison            │ │
│ └───────────────┘    └───────────────┘    └───────────────────────┘ │
│                                                                     │
│ Compares:                                                           │
│ - Original image accuracy (baseline)                                │
│ - Adversarial image accuracy                                        │
│ - Accuracy change (degradation/improvement)                         │
└─────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ FINAL OUTPUT: Accuracy metrics showing model robustness             │
│               against Square attack                                 │
│                                                                     │
│ For Qwen25_VL_3B with Square Attack:                                │
│ - Original accuracy: 82.35%                                         │
│ - Adversarial accuracy: 76.47%                                      │
│ - Change: -5.88% (Degradation)                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Overview

This evaluation framework includes:
- Testing infrastructure for lightweight VLMs (4-bit quantized, ~3B parameters)
- True black-box adversarial attack implementations to test model robustness
- Performance benchmarking on visual reasoning tasks under attack conditions
- Comparative analysis between original and adversarially perturbed inputs
- Evaluation across diverse visual content including charts, tables, and maps

### Adversarial Attack Types

The repository implements true black-box attacks that don't require any gradient information from the target model:

- **Square Attack**: A score-based black-box attack that perturbs square-shaped regions of the image to maximize the loss. Enhanced with perceptual constraints (SSIM, LPIPS, CLIP) to maintain visual quality.

### Future Black-Box Attack Implementations

The modular architecture of the `true_black_box_attacks` directory is designed to scale to additional black-box attacks:

- **HopSkipJump Attack**: A decision-based attack that requires only prediction labels, not probabilities or gradients.
- **Threshold Attack**: Uses binary search to find minimal perturbations that cross decision boundaries.
- **Pixel Attack**: Modifies individual pixels to cause misclassification with minimal changes.
- **Simple Black-box Adversarial Attack**: Uses random sampling to find adversarial perturbations.
- **Spatial Transformation Attack**: Applies geometric transformations instead of pixel-level perturbations.
- **Query-efficient Black-box Attack**: Optimizes the number of queries needed to find adversarial examples.
- **Zeroth Order Optimization (ZOO)**: Estimates gradients using finite differences for black-box optimization.
- **Decision-based/Boundary Attack**: Starts from an adversarial example and iteratively reduces perturbation while staying adversarial.
- **Geometric Decision-based Attack (GeoDA)**: Uses geometric insights to efficiently find adversarial examples.

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

The evaluation pipeline consists of two main scripts and the Square Attack implementation:

### Evaluation Scripts

#### 1. `eval_model.py`

Generates evaluation results for a specific model on a specific task.

```bash
cd scripts
python eval_model.py
```

- Interactive engine selection (GPT-4o or Qwen25_VL_3B or ALL)
- Interactive attack type selection
- Fixed task ('chart') and sample count (17)
- Results saved to `results/{engine}/eval_{engine}_{task}_{random_count}.json`
- Can be configured to use original or adversarial images

#### 2. `eval_vqa.py`

Analyzes evaluation results and calculates accuracy metrics.

```bash
cd scripts
python eval_vqa.py
```

- Interactive engine selection
- Fixed task ('chart')
- Automatically finds and evaluates all result files
- Shows accuracy comparison and changes due to different attack types

### Attack Implementations

| Attack Type | Attack | Script | Approach | Parameters | Output Directory |
|-------------|--------|--------|----------|------------|-----------------|
| **True Black-Box** | Square | `v10_square_attack.py` | Score-based with perceptual constraints | `--eps`, `--norm`, `--max_iter`, `--ssim_threshold`, `--lpips_threshold`, `--clip_threshold` | `test_BB_square/` |

## Workflow and Usage

### Complete Evaluation Workflow

1. **Generate adversarial images** using the Square Attack:
   ```bash
   # True Black-Box Attack
   python attack_models/true_black_box_attacks/v10_square_attack.py --image_path data/test_extracted/chart/image.png --eps 0.05 --norm inf --max_iter 100 --ssim_threshold 0.95 --lpips_threshold 0.05 --clip_threshold 0.9
   ```

2. **Run evaluation** on original and adversarial images:
   ```bash
   python scripts/eval_model.py
   # Follow the interactive prompts to select the model and attack type
   ```

3. **Compare results**:
   ```bash
   python scripts/eval_vqa.py
   # Follow the interactive prompts to select the model
   ```

## Directory Structure

```
Multi-modal-Self-instruct/
├── attack_models/          # Adversarial attack scripts
│   └── true_black_box_attacks/ # True black-box attacks (no gradient access)
│       ├── v0_attack_utils.py  # Utility functions with perceptual constraints
│       └── v10_square_attack.py # Square Attack implementation
├── data/                   # Dataset files
│   ├── test_extracted/        # Original test images
│   └── test_BB_square/        # Square attack adversarial images
├── results/                # Evaluation results
│   ├── gpt4o/              # GPT-4o results
│   └── Qwen25_VL_3B/       # Qwen results
├── scripts/                # Evaluation scripts
│   ├── eval_model.py       # Script to generate model responses
│   ├── eval_vqa.py         # Script to calculate accuracy metrics
│   ├── select_attack.py    # Script for attack selection
│   ├── llm_tools.py        # Utilities for OpenAI API calls
│   └── local_llm_tools.py  # Utilities for local model inference
└── unit_test/              # Test scripts
```

## Adversarial Robustness Results

Testing shows interesting differences in model robustness against the Square Attack:

### Results for Square Attack

#### Qwen25_VL_3B
| Attack | Accuracy | Change | Effect |
|--------|----------|--------|--------|
| Original | 82.35% | 0.00% | Baseline |
| Square | 76.47% | -5.88% | Degradation |

### Key Insights

- **Perceptual Constraints Matter**: The Square Attack's enhanced perceptual constraints (SSIM, LPIPS, CLIP) help maintain visual quality while still creating adversarial examples, resulting in less performance degradation but better visual quality.

- **Minimal Performance Impact**: The Square Attack causes only a minor degradation in Qwen25_VL_3B's performance (-5.88%), suggesting that perceptually constrained black-box attacks may be less effective but produce more visually imperceptible perturbations.

## Recent Progress and Future Work

### Recent Improvements
- Implemented the Square Attack with enhanced perceptual constraints using SSIM, LPIPS, and CLIP similarity metrics
- Refactored utility functions to make them more modular and reusable across different attack implementations
- Enhanced evaluation scripts to support the new attack organization
- Added detailed documentation for the attack method with mathematical formulations

### Future Directions
- Implement additional true black-box attacks (HopSkipJump, Threshold Attack, Pixel Attack, etc.)
- Develop more sophisticated perceptual constraints for adversarial examples
- Evaluate finetuned models under adversarial conditions to measure robustness improvements
- Compare effectiveness of adversarial training techniques in improving VLM robustness
- Explore multi-modal adversarial attacks that target both vision and language components
