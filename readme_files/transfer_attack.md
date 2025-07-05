# Transfer Attack Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                      TRANSFER ATTACK WORKFLOW                       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────┐
│ STEP 1: GENERATE ADVERSARIAL│
│         EXAMPLES           │
└───────────────┬─────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ attack_models/transfer_attacks/v3_fgsm_attack.py                    │
│                                                                     │
│ ┌───────────────┐    ┌───────────────┐    ┌───────────────────────┐ │
│ │ Load Original │ → │ Apply FGSM     │ → │ Save Adversarial      │ │
│ │ Image         │    │ Attack        │    │ Image                 │ │
│ └───────────────┘    └───────────────┘    └───────────────────────┘ │
│                                                                     │
│ Parameters:                                                         │
│ --image_path: Original image                                        │
│ --eps: Perturbation magnitude                                       │
│ --targeted: Whether to perform targeted attack                      │
│ --target_class: Target class (for targeted attacks)                 │
└─────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ OUTPUT: data/test_BB_fgsm/chart/20231114102825506748.png            │
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
│ OUTPUT: results/Qwen25_VL_3B/eval_Qwen25_VL_3B_chart_17_BB_fgsm.json │
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
│               against FGSM attack                                   │
│                                                                     │
│ For Qwen25_VL_3B with FGSM:                                         │
│ - Original accuracy: 82.35%                                         │
│ - Adversarial accuracy: 41.18%                                      │
│ - Change: -41.18% (Degradation)                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Transfer Attack Overview

Transfer attacks use a substitute model's gradients to generate adversarial examples that can then be transferred to the target model. This approach is necessary when the target model doesn't provide gradient access, which is common with many commercial VLMs.

## Attack Implementations

The repository includes the following transfer attack implementations:

| Attack | Script | Approach | Parameters | Output Directory |
|--------|--------|----------|------------|-----------------|
| **PGD** | `v2_pgd_attack.py` | Multi-step with projection | `--eps`, `--eps_step`, `--max_iter` | `test_BB_pgd/` |
| **FGSM** | `v3_fgsm_attack.py` | Single-step gradient sign | `--eps`, `--targeted`, `--target_class` | `test_BB_fgsm/` |
| **CW-L2** | `v4_cw_l2_attack.py` | Optimization for L2 norm | `--confidence`, `--max_iter`, `--learning_rate` | `test_BB_cw_l2/` |
| **CW-L0** | `v5_cw_l0_attack.py` | Optimization for pixel count | `--confidence`, `--max_iter` | `test_BB_cw_l0/` |
| **CW-L∞** | `v6_cw_linf_attack.py` | Optimization for max perturbation | `--confidence`, `--binary_steps` | `test_BB_cw_linf/` |
| **L-BFGS** | `v7_lbfgs_attack.py` | Box-constrained optimization | `--target_class`, `--c_init`, `--max_iter` | `test_BB_lbfgs/` |
| **JSMA** | `v8_jsma_attack.py` | Saliency mapping | `--target_class`, `--max_iter`, `--theta`, `--use_logits` | `test_BB_jsma/` |
| **DeepFool** | `v9_deepfool_attack.py` | Iterative linear approximation | `--max_iter`, `--overshoot` | `test_BB_deepfool/` |

## Technical Details

### 1. Attack Generation (FGSM)

The Fast Gradient Sign Method (FGSM) attack is defined mathematically as:
- For untargeted attacks: x' = x + ε · sign(∇ₓJ(θ, x, y))
- For targeted attacks: x' = x - ε · sign(∇ₓJ(θ, x, t))

Where:
- x is the original input image
- x' is the adversarial example
- ε is the perturbation magnitude (controls how much each pixel can change)
- J is the loss function
- θ represents the model parameters
- y is the true label
- t is the target label
- ∇ₓJ represents the gradient of the loss with respect to the input x
- sign() is the sign function that returns -1, 0, or 1 depending on the sign of its input

### 2. Model Evaluation (Qwen25_VL_3B)

The Qwen25_VL_3B model is a 3 billion parameter vision-language model that processes both images and text. For efficient inference, it's configured with:
- 4-bit quantization using BitsAndBytesConfig
- NF4 quantization type
- Double quantization for further memory optimization
- Float16 compute dtype

### 3. Transfer Attack Strategy

This is a "transfer attack" strategy:
1. A pre-trained ResNet50 serves as a substitute model to generate adversarial examples
2. The attack is performed on this substitute model to create perturbed images
3. These adversarial examples are then transferred to test the target VLM (Qwen25_VL_3B)
4. This approach works because adversarial examples often transfer between different models

### 4. Evaluation Metrics

The evaluation compares:
- Accuracy on original images (baseline)
- Accuracy on adversarial images
- Percentage change in accuracy
- Classification of effect as "Degradation" (negative change) or "Improvement" (positive change)

## Adversarial Robustness Results

Testing shows interesting differences in model robustness against different transfer attack types:

### Comprehensive Results Table

#### GPT-4o
| Attack | Accuracy | Change | Effect |
|--------|----------|--------|--------|
| Original | 64.71% | 0.00% | Baseline |
| CW-L0 | 58.82% | -5.88% | Degradation |
| FGSM | 64.71% | 0.00% | No Change |
| PGD | 70.59% | +5.88% | Improvement |
| JSMA | 70.59% | +5.88% | Improvement |
| CW-L2 | 76.47% | +11.76% | Improvement |
| CW-L∞ | 82.35% | +17.65% | Improvement |
| L-BFGS | 82.35% | +17.65% | Improvement |
| DeepFool | 82.35% | +17.65% | Improvement |

#### Qwen25_VL_3B
| Attack | Accuracy | Change | Effect |
|--------|----------|--------|--------|
| Original | 82.35% | 0.00% | Baseline |
| CW-L0 | 11.76% | -70.59% | Degradation |
| CW-L∞ | 29.41% | -52.94% | Degradation |
| PGD | 35.29% | -47.06% | Degradation |
| CW-L2 | 35.29% | -47.06% | Degradation |
| L-BFGS | 35.29% | -47.06% | Degradation |
| JSMA | 35.29% | -47.06% | Degradation |
| FGSM | 41.18% | -41.18% | Degradation |
| DeepFool | 47.06% | -35.29% | Degradation |

### Key Insights

- **Contrasting Robustness Profiles**: While Qwen25_VL_3B outperforms GPT-4o on clean images (82.35% vs 64.71%), GPT-4o demonstrates exceptional robustness to adversarial attacks, with performance actually improving under most attack conditions.

- **Unexpected Performance Enhancement**: Most notably, GPT-4o's performance improves significantly with CW-L∞, L-BFGS, and DeepFool attacks (+17.65%), suggesting advanced adversarial training or architectural innovations in GPT-4o.

- **Attack Effectiveness Patterns**: 
  - For GPT-4o: CW-L∞ = L-BFGS = DeepFool > CW-L2 > PGD = JSMA > FGSM = Original > CW-L0
  - For Qwen25_VL_3B: Original > DeepFool > FGSM > PGD = CW-L2 = L-BFGS = JSMA > CW-L∞ > CW-L0

- **Differential Impact of CW-L0**: The CW-L0 attack is the only attack that consistently degrades GPT-4o's performance (-5.88%) and is also the most effective against Qwen25_VL_3B (-70.59%). This suggests that sparse but significant pixel changes are particularly challenging for both models.

## Usage Examples

Generate adversarial images using transfer attacks:

```bash
# PGD attack
python attack_models/transfer_attacks/v2_pgd_attack.py --image_path data/test_extracted/chart/image.png --eps 0.03

# FGSM attack
python attack_models/transfer_attacks/v3_fgsm_attack.py --image_path data/test_extracted/chart/image.png --eps 0.03

# CW-L2 attack
python attack_models/transfer_attacks/v4_cw_l2_attack.py --image_path data/test_extracted/chart/image.png --confidence 5 --max_iter 100

# CW-L0 attack
python attack_models/transfer_attacks/v5_cw_l0_attack.py --image_path data/test_extracted/chart/image.png --max_iter 50 --confidence 10

# CW-L∞ attack
python attack_models/transfer_attacks/v6_cw_linf_attack.py --image_path data/test_extracted/chart/image.png --confidence 5 --binary_steps 10

# L-BFGS attack
python attack_models/transfer_attacks/v7_lbfgs_attack.py --image_path data/test_extracted/chart/image.png --target_class 20 --c_init 0.1 --max_iter 10

# JSMA attack
python attack_models/transfer_attacks/v8_jsma_attack.py --image_path data/test_extracted/chart/image.png --target_class 20 --max_iter 100 --theta 1.0

# DeepFool attack
python attack_models/transfer_attacks/v9_deepfool_attack.py --image_path data/test_extracted/chart/image.png --max_iter 50 --overshoot 0.02
```
