# Evaluating Nano Vision-Language Models (VLMs) Against Cyber Security Attacks: Comprehensive Analysis Across White-Box (Finetuning) and Black-Box (Inference) Attack Scenarios

This repository contains tools for evaluating small (4-bit, 3 Billion parameter) vision-language models (VLMs) under various multi-modal adversarial attacks, focusing on their robustness and performance degradation.

## Overview

This evaluation framework includes:
- Testing infrastructure for lightweight VLMs (4-bit quantized, ~3B parameters)
- Multi-modal adversarial attack implementations to test model robustness
- Performance benchmarking on visual reasoning tasks under attack conditions
- Comparative analysis between original and adversarially perturbed inputs
- Evaluation across diverse visual content including charts, tables, and maps

### Adversarial Attack Variants

Each attack variant has its own strengths and characteristics:

- **Projected Gradient Descent (PGD)**: An iterative attack that takes multiple small steps in the direction of the gradient and projects the perturbation back onto an epsilon-ball after each step. PGD creates stronger adversarial examples than single-step methods by exploring the loss landscape more thoroughly while maintaining a constraint on the maximum perturbation.

- **Fast Gradient Sign Method (FGSM)**: A single-step attack that generates adversarial examples by taking a step in the direction of the sign of the gradient of the loss function with respect to the input. FGSM is computationally efficient but typically produces less effective adversarial examples compared to iterative methods.

- **Carlini-Wagner L2 (CW-L2)**: Produces visually imperceptible perturbations with minimal overall distortion by optimizing for the L2 (Euclidean) distance between original and adversarial images. This variant typically creates high-quality adversarial examples that are difficult to detect visually.

- **Carlini-Wagner L0 (CW-L0)**: Changes the fewest pixels but may make more noticeable changes to those pixels. This variant focuses on minimizing the number of modified pixels rather than the magnitude of changes, resulting in sparse but potentially more visible perturbations.

- **Carlini-Wagner L∞ (CW-L∞)**: Distributes changes evenly across the image, limiting the maximum change to any pixel. This variant ensures that no single pixel is modified beyond a certain threshold, creating perturbations that are bounded in their maximum intensity.

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

The evaluation pipeline consists of two main scripts and five attack implementations:

### Evaluation Scripts

#### 1. `eval_model.py`

Generates evaluation results for a specific model on a specific task.

```bash
cd scripts
python eval_model.py
```

- Interactive engine selection (GPT-4o or Qwen25_VL_3B)
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

| Attack | Script | Approach | Parameters | Output Directory |
|--------|--------|----------|------------|-----------------|
| **PGD** | `v2_pgd_attack.py` | Multi-step with projection | `--eps`, `--eps_step`, `--max_iter` | `test_extracted_adv/` |
| **FGSM** | `v3_fgsm_attack.py` | Single-step gradient sign | `--eps` | `test_extracted_adv_fgsm/` |
| **CW-L2** | `v4_cw_l2_attack.py` | Optimization for L2 norm | `--confidence`, `--max_iter`, `--learning_rate` | `test_extracted_adv_cw_l2/` |
| **CW-L0** | `v5_cw_l0_attack.py` | Optimization for pixel count | `--confidence`, `--max_iter` | `test_extracted_adv_cw_l0/` |
| **CW-L∞** | `v6_cw_linf_attack.py` | Optimization for max perturbation | `--confidence`, `--binary_steps` | `test_extracted_adv_cw_linf/` |

All attacks use a "black-box transfer attack" strategy with a pre-trained ResNet50 as a substitute model, since most VLMs don't provide gradient access needed for direct attacks.

## Workflow and Usage

### Complete Evaluation Workflow

1. **Generate adversarial images** using one or more attack methods:
   ```bash
   # PGD attack
   python attack_models/black_box_attacks/v2_pgd_attack.py --image_path data/test_extracted/chart/image.png --eps 0.03
   
   # FGSM attack
   python attack_models/black_box_attacks/v3_fgsm_attack.py --image_path data/test_extracted/chart/image.png --eps 0.03
   
   # CW-L2 attack
   python attack_models/black_box_attacks/v4_cw_l2_attack.py --image_path data/test_extracted/chart/image.png --confidence 5 --max_iter 100
   
   # CW-L0 attack
   python attack_models/black_box_attacks/v5_cw_l0_attack.py --image_path data/test_extracted/chart/image.png --max_iter 50 --confidence 10
   
   # CW-L∞ attack
   python attack_models/black_box_attacks/v6_cw_linf_attack.py --image_path data/test_extracted/chart/image.png --confidence 5 --binary_steps 10
   ```

2. **Run evaluation** on original and adversarial images:
   ```bash
   # Edit eval_model.py to use the appropriate image path:
   # - Original: img_path = 'data/test_extracted/' + data['image']
   # - PGD: img_path = 'data/test_extracted_adv/' + data['image']
   # - FGSM: img_path = 'data/test_extracted_adv_fgsm/' + data['image']
   # - CW-L2: img_path = 'data/test_extracted_adv_cw_l2/' + data['image']
   # - CW-L0: img_path = 'data/test_extracted_adv_cw_l0/' + data['image']
   # - CW-L∞: img_path = 'data/test_extracted_adv_cw_linf/' + data['image']
   
   python scripts/eval_model.py
   # Follow the interactive prompts to select the model
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
│   └── black_box_attacks/  # Black-box attack implementations
│       ├── v2_pgd_attack.py         # PGD attack implementation
│       ├── v3_fgsm_attack.py        # FGSM attack implementation
│       ├── v4_cw_l2_attack.py       # CW-L2 attack implementation
│       ├── v5_cw_l0_attack.py       # CW-L0 attack implementation
│       └── v6_cw_linf_attack.py     # CW-L∞ attack implementation
├── data/                   # Dataset files
│   ├── test_extracted/        # Original test images
│   ├── test_extracted_adv/    # PGD adversarial images
│   ├── test_extracted_adv_fgsm/ # FGSM adversarial images
│   ├── test_extracted_adv_cw_l2/ # CW-L2 adversarial images
│   ├── test_extracted_adv_cw_l0/ # CW-L0 adversarial images
│   └── test_extracted_adv_cw_linf/ # CW-L∞ adversarial images
├── results/                # Evaluation results
│   ├── gpt4o/              # GPT-4o results
│   └── Qwen25_VL_3B/       # Qwen results
├── scripts/                # Evaluation scripts
│   ├── eval_model.py       # Script to generate model responses
│   ├── eval_vqa.py         # Script to calculate accuracy metrics
│   ├── llm_tools.py        # Utilities for OpenAI API calls
│   └── local_llm_tools.py  # Utilities for local model inference
└── unit_test/              # Test scripts
```

## Adversarial Robustness Results

Testing shows interesting differences in model robustness against different attack types:

### Comprehensive Results Table

| Model | Image Type | Accuracy | Accuracy Change |
|-------|------------|----------|----------------|
| GPT-4o | Original | 64.71% | - |
| GPT-4o | PGD Adversarial | 70.59% | +5.88% (improvement) |
| GPT-4o | FGSM Adversarial | 64.71% | 0.00% (no change) |
| GPT-4o | CW-L2 Adversarial | 76.47% | +11.76% (improvement) |
| GPT-4o | CW-L0 Adversarial | 58.82% | -5.88% (degradation) |
| GPT-4o | CW-L∞ Adversarial | 82.35% | +17.65% (improvement) |
| Qwen25_VL_3B | Original | 82.35% | - |
| Qwen25_VL_3B | PGD Adversarial | 35.29% | -47.06% (degradation) |
| Qwen25_VL_3B | FGSM Adversarial | 41.18% | -41.18% (degradation) |
| Qwen25_VL_3B | CW-L2 Adversarial | 35.29% | -47.06% (degradation) |
| Qwen25_VL_3B | CW-L0 Adversarial | 11.76% | -70.59% (degradation) |
| Qwen25_VL_3B | CW-L∞ Adversarial | 35.29% | -47.06% (degradation) |

### Key Insights

- **Contrasting Robustness Profiles**: While Qwen25_VL_3B outperforms GPT-4o on clean images (82.35% vs 64.71%), GPT-4o demonstrates exceptional robustness to adversarial attacks, with performance actually improving under most attack conditions.

- **Unexpected Performance Enhancement**: Most notably, GPT-4o's performance improves significantly with CW-L∞ (+17.65%) and CW-L2 (+11.76%) attacks, suggesting advanced adversarial training or architectural innovations in GPT-4o.

- **Attack Effectiveness Patterns**: 
  - For GPT-4o: CW-L∞ > CW-L2 > PGD > FGSM = Original > CW-L0
  - For Qwen25_VL_3B: Original > FGSM > PGD = CW-L2 = CW-L∞ > CW-L0

- **Differential Impact of CW-L0**: The CW-L0 attack is the only attack that degrades GPT-4o's performance (-5.88%) and is also the most effective against Qwen25_VL_3B (-70.59%). This suggests that sparse but significant pixel changes are particularly challenging for both models.

- **Implications for VLM Design**: These findings challenge conventional wisdom about adversarial attacks and suggest that certain architectural choices or training techniques may not only provide robustness but actually enhance performance under attack conditions.

## Recent Progress and Future Work

### Recent Improvements
- Added five complementary attack implementations (PGD, FGSM, CW-L2, CW-L0, CW-L∞) for comprehensive robustness testing
- Enhanced evaluation scripts with interactive model selection and automatic result comparison
- Improved accuracy reporting to clearly indicate improvements vs. degradations
- Streamlined workflow for testing multiple attack types against different models

### Future Directions
- Implement white-box attacks for models where architecture and gradients are accessible
- Evaluate finetuned models under adversarial conditions to measure robustness improvements
- Compare effectiveness of adversarial training techniques in improving VLM robustness
- Explore multi-modal adversarial attacks that target both vision and language components
