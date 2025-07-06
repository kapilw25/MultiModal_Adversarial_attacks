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

- **Square Attack**: A score-based black-box attack that perturbs square-shaped regions of the image to maximize the loss. Enhanced with perceptual constraints (SSIM) to maintain visual quality while still degrading model performance.

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
| **True Black-Box** | Square | `v10_square_attack.py` | Score-based with perceptual constraints | `--eps`, `--norm`, `--max_iter`, `--ssim_threshold` | `test_BB_square/` |

## Workflow and Usage

### Complete Evaluation Workflow

1. **Generate adversarial images** using the Square Attack:
   ```bash
   # True Black-Box Attack
   python attack_models/true_black_box_attacks/v10_square_attack.py --image_path data/test_extracted/chart/image.png --eps 0.15 --norm inf --max_iter 200 --p_init 0.3 --ssim_threshold 0.85
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

Testing shows interesting differences in model robustness against various attacks:

### Results for Different Attacks on Qwen25_VL_3B

| Attack Type | Original Accuracy | Attack Accuracy | Change | Effect |
|-------------|-------------------|-----------------|--------|--------|
| Original | 82.35% | 82.35% | 0.00% | Baseline |
| PGD | 82.35% | 70.59% | -11.76% | Degradation |
| FGSM | 82.35% | 35.29% | -47.06% | Degradation |
| CW-L2 | 82.35% | 35.29% | -47.06% | Degradation |
| CW-L0 | 82.35% | 47.06% | -35.29% | Degradation |
| CW-L∞ | 82.35% | 29.41% | -52.94% | Degradation |
| L-BFGS | 82.35% | 35.29% | -47.06% | Degradation |
| JSMA | 82.35% | 82.35% | 0.00% | No Change |
| DeepFool | 82.35% | 47.06% | -35.29% | Degradation |
| Square | 82.35% | 76.47% | -5.88% | Degradation |

### Key Insights

- **SSIM Threshold Impact**: Our experiments show that targeting a specific SSIM value (0.85) creates more effective adversarial examples than simply meeting a minimum threshold. With SSIM=0.85, the Square Attack achieved a 5.88% performance degradation, while with SSIM=0.99, there was no degradation.

- **Visual Quality vs. Attack Effectiveness**: There's a clear trade-off between visual quality and attack effectiveness. Lower SSIM values (more visible perturbations) generally lead to greater performance degradation.

- **Numerical Calculation Errors**: The Square Attack with SSIM=0.85 primarily affected the model's ability to perform numerical calculations on chart data, with significant errors in computing averages, sums, differences, and ratios.

- **Attack Comparison**: While the Square Attack (5.88% degradation) is less effective than gradient-based attacks like CW-L∞ (52.94% degradation), it's still able to cause meaningful performance degradation without requiring any access to model gradients.

- **Perceptual Constraints**: Using exact SSIM targeting rather than minimum thresholds allows for more precise control over the visual quality vs. effectiveness trade-off in adversarial examples.

## Recent Progress and Future Work

### Recent Improvements
- Modified the Square Attack implementation to target exact SSIM values rather than just meeting minimum thresholds
- Implemented binary search with increased precision (20 steps) to achieve SSIM values very close to the target
- Removed safety margin adjustments to prevent over-constraining the adversarial examples
- Simplified the implementation by focusing on a single perceptual metric (SSIM) for better control

### Future Directions
- Implement additional true black-box attacks (HopSkipJump, Threshold Attack, Pixel Attack, etc.)
- Develop more sophisticated perceptual constraints for adversarial examples
- Evaluate finetuned models under adversarial conditions to measure robustness improvements
- Compare effectiveness of adversarial training techniques in improving VLM robustness
- Explore multi-modal adversarial attacks that target both vision and language components
- Investigate the relationship between SSIM values and model performance degradation across different attack types
