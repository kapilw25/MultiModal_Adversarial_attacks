# Evaluating Nano Vision-Language Models' (VLMs) Robustness Against Cyber Security Attacks

This repository contains tools for evaluating small (4-bit, 3 Billion parameter) vision-language models (VLMs) under various multi-modal adversarial attacks, focusing on their robustness and performance degradation.

## System Design

The evaluation framework consists of four main modules:

### 1. VLM Inference Engine
Located in `local_model/`, this module handles model loading, quantization, and inference using a modular, plug-and-play architecture:

- `base_model.py`: Defines the abstract base class `BaseVLModel` that all VLM implementations must inherit from
- `model_classes.py`: Implements the factory pattern for creating model instances
- `model_utils.py`: Contains utility functions for memory management and model loading
- `qwen_model.py`: Contains `QwenVLModelWrapper` for Qwen2.5-VL-3B-Instruct
- `gemma_model.py`: Contains `GemmaVLModelWrapper` for Google's Gemma-3-4b-it
- Future models: GuardReasoner-VL-Eco-3B, NQLSG-Qwen2-VL-2B-v2-Base

#### 4-bit Quantization for Nano VLMs

The VLM Inference Engine supports model-specific 4-bit quantization strategies to optimize memory usage and inference speed:

- **Model-Specific Optimizations**:
  - **Qwen2.5-VL-3B-Instruct**: Uses `torch.float16` compute dtype with standard NF4 quantization
  - **Gemma-3-4b-it**: Uses `torch.bfloat16` compute dtype for better numerical stability, which is critical for Google models

- **Memory Management**:
  - Implemented memory monitoring decorators (`@memory_efficient`, `@time_inference`)
  - Model-specific memory optimizations (e.g., `low_cpu_mem_usage=True` for Gemma)
  - Explicit GPU memory fraction control for larger models

- **Performance Impact**:
  - 4-bit quantization reduces memory usage by ~65% compared to FP16
  - Inference speed varies by model (Gemma: ~2.5s/inference, Qwen: ~1.5s/inference)
  - Minimal accuracy degradation when using model-specific quantization parameters

#### Modular Design for Adding New VLMs

The VLM Inference Engine uses a clean, modular architecture that makes it easy to add new vision-language models:

1. **Abstract Base Class**: `BaseVLModel` in `base_model.py` defines a standard interface that all models must implement
2. **Factory Pattern**: `model_classes.py` provides a simple factory function that creates the appropriate model instance
3. **Model Implementation**: Each model has its own implementation file (e.g., `qwen_model.py`) that inherits from `BaseVLModel`

To add a new VLM (e.g., GuardReasoner-VL-Eco-3B):
1. Create a new file (e.g., `guard_reasoner_model.py`) with a class that inherits from `BaseVLModel`
2. Add a new condition in the `create_model` function in `model_classes.py`
3. Update the `get_model` function in `local_llm_tools.py` to support the new engine
4. Configure model-specific quantization parameters based on the model architecture

This design ensures that new models can be added with minimal changes to the existing codebase.

### 2. Transfer Attacks
Located in `attack_models/transfer_attacks/`, these attacks typically require access to model gradients but are implemented here using surrogate models:
- `v2_pgd_attack.py`: Projected Gradient Descent attack
- `v3_fgsm_attack.py`: Fast Gradient Sign Method attack
- `v4_cw_l2_attack.py`: Carlini & Wagner L2 attack
- `v5_cw_l0_attack.py`: Carlini & Wagner L0 attack
- `v6_cw_linf_attack.py`: Carlini & Wagner L∞ attack
- `v7_lbfgs_attack.py`: L-BFGS attack
- `v8_jsma_attack.py`: Jacobian-based Saliency Map Attack
- `v9_deepfool_attack.py`: DeepFool attack

### 3. True Black-Box Attacks
Located in `attack_models/true_black_box_attacks/`, these attacks don't require any gradient information:
- `v10_square_attack.py`: Square Attack with perceptual constraints
- `v11_hop_skip_jump_attack.py`: HopSkipJump Attack with perceptual constraints
- `v12_pixel_attack.py`: Pixel Attack with perceptual constraints
- `v13_simba_attack.py`: SimBA (Simple Black-box Adversarial) Attack with perceptual constraints
- `v14_spatial_transformation_attack.py`: Spatial Transformation Attack with perceptual constraints
- `v15_query_efficient_bb_attack.py`: Query-Efficient Black-box Attack with perceptual constraints
- `v0_attack_utils.py`: Shared utility functions for all black-box attacks

### 4. Evaluation Framework
Located in `scripts/`, this module handles model evaluation and result analysis:
- `eval_model.py`: Generates model responses for specific tasks
- `eval_vqa.py`: Analyzes results and calculates accuracy metrics
- `select_attack.py`: Handles attack selection and configuration
- `store_results_db.py`: Stores evaluation results in a SQLite database for efficient querying and analysis

## Attack Workflow

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                    ADVERSARIAL ATTACK WORKFLOW                                                                                    │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────┐          ┌────────────────────────────┐          ┌────────────────────────────┐          ┌────────────────────────────┐
│  STEP 1: GENERATE          │          │  STEP 2: INITIALIZE        │          │  STEP 3: EVALUATE          │          │  STEP 4: CALCULATE         │
│  ADVERSARIAL EXAMPLES      │ ──────► │  VLM INFERENCE ENGINE      │ ──────► │  MODEL ON ADV IMAGES       │ ──────► │  ACCURACY METRICS          │
└────────────┬───────────────┘          └────────────┬───────────────┘          └────────────┬───────────────┘          └────────────┬───────────────┘
             │                                       │                                       │                                       │
             ▼                                       ▼                                       ▼                                       ▼
┌────────────────────────────┐          ┌────────────────────────────┐          ┌────────────────────────────┐          ┌────────────────────────────┐
│ attack_models/[attack_type]│          │ local_model/               │          │ scripts/eval_model.py      │          │ scripts/eval_vqa.py        │
│ /[attack_script].py        │          │                            │          │                            │          │                            │
│                            │          │                            │          │                            │          │                            │
│ ┌──────────┐               │          │ ┌──────────┐               │          │ ┌──────────┐               │          │ ┌──────────┐               │
│ │Load      │               │          │ │Load      │               │          │ │Select    │               │          │ │Load      │               │
│ │Original  │               │          │ │Model     │               │          │ │Model     │               │          │ │Results   │               │
│ │Image     │               │          │ │Weights   │               │          │ │          │               │          │ │JSON Files│               │
│ └────┬─────┘               │          │ └────┬─────┘               │          │ └────┬─────┘               │          │ └────┬─────┘               │
│      │                     │          │      │                     │          │      │                     │          │      │                     │
│      ▼                     │          │      ▼                     │          │      ▼                     │          │      ▼                     │
│ ┌──────────┐               │          │ ┌──────────┐               │          │ ┌──────────┐               │          │ ┌──────────┐               │
│ │Apply     │               │          │ │Apply     │               │          │ │Load      │               │          │ │Calculate │               │
│ │Attack    │               │          │ │4-bit     │               │          │ │Images    │               │          │ │Accuracy  │               │
│ │Algorithm │               │          │ │Quantize  │               │          │ │(Adv)    │               │          │ │          │               │
│ └────┬─────┘               │          │ └────┬─────┘               │          │ └────┬─────┘               │          │ └────┬─────┘               │
│      │                     │          │      │                     │          │      │                     │          │      │                     │
│      ▼                     │          │      ▼                     │          │      ▼                     │          │      ▼                     │
│ ┌──────────┐               │          │ ┌──────────┐               │          │ ┌──────────┐               │          │ ┌──────────┐               │
│ │Apply     │               │          │ │Initialize│               │          │ │Generate  │               │          │ │Display   │               │
│ │Perceptual│               │          │ │Inference │               │          │ │Prediction│               │          │ │Accuracy  │               │
│ │Constraint│               │          │ │Pipeline  │               │          │ │          │               │          │ │Comparison│               │
│ └──────────┘               │          │ └──────────┘               │          │ └──────────┘               │          │ └──────────┘               │
└────────────────────────────┘          └────────────────────────────┘          └────────────────────────────┘          └────────────────────────────┘
        │                                        │                                        │                                        │
        ▼                                        ▼                                        ▼                                        ▼
┌────────────────────────────┐          ┌────────────────────────────┐          ┌────────────────────────────┐
│ OUTPUT:                    │          │ Model Loading Path:        │          │ OUTPUT:                    │
│ data/test_BB_[attack]/     │          │ model_classes.py →         │          │ results/Qwen25_VL_3B/      │
│ chart/[image_name].png     │          │ create_model() →           │          │ eval_Qwen25_VL_3B_chart_   │
│                            │          │ qwen_model.py →            │          │ 17_BB_[attack].json        │
│                            │          │ QwenVLModelWrapper         │          │                            │
└────────────────────────────┘          └────────────────────────────┘          └────────────────────────────┘
```

## Execution Commands

### 1. Setup

```bash
python -m venv venv_MM
source venv_MM/bin/activate
pip install -r requirements.txt
```

### 2. Generate Adversarial Examples

#### Square Attack (True Black-Box)
```bash
source venv_MM/bin/activate && python attack_models/true_black_box_attacks/v10_square_attack.py --image_path data/test_extracted/chart/20231114102825506748.png --eps 0.15 --norm inf --max_iter 200 --p_init 0.3 --ssim_threshold 0.85
```

#### HopSkipJump Attack (True Black-Box)
```bash
source venv_MM/bin/activate && python attack_models/true_black_box_attacks/v11_hop_skip_jump_attack.py --image_path data/test_extracted/chart/20231114102825506748.png --norm 2 --max_iter 20 --max_eval 500 --ssim_threshold 0.85
```

#### Pixel Attack (True Black-Box)
```bash
source venv_MM/bin/activate && python attack_models/true_black_box_attacks/v12_pixel_attack.py --image_path data/test_extracted/chart/20231114102825506748.png --th 10 --es 1 --max_iter 100 --ssim_threshold 0.85 --num_pixels 20
```

#### SimBA Attack (True Black-Box)
```bash
source venv_MM/bin/activate && python attack_models/true_black_box_attacks/v13_simba_attack.py --image_path data/test_extracted/chart/20231114102825506748.png --epsilon 0.15 --max_iter 1000 --freq_dim 32 --order diag --ssim_threshold 0.85
```

#### Spatial Transformation Attack (True Black-Box)
```bash
source venv_MM/bin/activate && python attack_models/true_black_box_attacks/v14_spatial_transformation_attack.py --image_path data/test_extracted/chart/20231114102825506748.png --max_translation 3 --max_rotation 10 --max_scaling 0.1 --ssim_threshold 0.85
```

#### Query-Efficient Black-box Attack (True Black-Box)
```bash
source venv_MM/bin/activate && python attack_models/true_black_box_attacks/v15_query_efficient_bb_attack.py --image_path data/test_extracted/chart/20231114102825506748.png --num_basis 20 --sigma 0.015625 --max_iter 100 --epsilon 0.1 --ssim_threshold 0.85
```

#### DeepFool Attack (Transfer-Based)
```bash
source venv_MM/bin/activate && python attack_models/transfer_attacks/v9_deepfool_attack.py --image_path data/test_extracted/chart/20231114102825506748.png --max_iter 50 --overshoot 0.02 --ssim_threshold 0.85
```

### 3. Evaluate Model Performance

```bash
# Run model evaluation on adversarial images
python scripts/eval_model.py

# Calculate accuracy metrics
python scripts/eval_vqa.py
```

## Attack Comparison Results

![Model Degradation Line Graph](/results/data_analysis/plots/model_degradation_line.png)

### VLM Robustness Against Different Attacks

This line graph shows how three different Vision-Language Models perform under various adversarial attacks, measuring their robustness by plotting accuracy change percentage against different attack types.

#### Key Observations:

- **GPT-4o (Blue)**: Most robust model, showing positive accuracy changes for many attacks, with peaks around +18% for Pixel and SimBA attacks.
- **Qwen-VL-3B (Orange)**: Most vulnerable model, showing significant negative accuracy changes (down to -53% for CW-L∞ and Pixel attacks).
- **Gemma-VL-4B (Green)**: Shows moderate robustness with performance fluctuating between slight improvements and moderate degradations.

#### Attack Effectiveness:
- **Most Damaging**: CW-L∞ and Pixel attacks for Qwen-VL-3B (-53%), CW-L0 for GPT-4o (-41%), Boundary for Gemma-VL-4B (-12%)
- **Least Effective**: JSMA, GeoDA, and Square attacks show minimal impact across models

The visualization demonstrates that different VLM architectures have varying levels of inherent robustness against adversarial attacks, with larger models generally showing more resilience than smaller ones.

### Attack Characteristics

The following tables provide qualitative descriptions of the attacks and their characteristics. For detailed numerical results including exact accuracy values and percentage changes, please refer to the `attack_comparison` table in the [Database Structure](#database-structure) section.

#### Transfer-Based Attacks

| Attack Name | Effectiveness | Approach | Implementation | 
|-------------|--------------|----------|----------------|
| CW-L∞ | Highest degradation | Optimizes perturbations under L∞ norm constraint | Uses a surrogate model to generate adversarial examples |
| FGSM | High degradation | Single-step gradient-based method that adds perturbation in the direction of the gradient sign | Fast and efficient but creates more visible perturbations |
| CW-L2 | High degradation | Optimizes perturbations under L2 norm constraint | Creates more localized perturbations than CW-L∞ |
| L-BFGS | High degradation | Uses L-BFGS optimization to find minimal perturbations | Computationally expensive but effective |
| DeepFool | Moderate degradation | Iteratively finds the nearest decision boundary | Creates smaller perturbations than FGSM |
| CW-L0 | Moderate degradation | Optimizes perturbations under L0 norm constraint (few pixels changed) | Creates sparse perturbations |
| PGD | Low degradation | Multi-step variant of FGSM with projection | More refined perturbations than FGSM |
| JSMA | No degradation | Modifies pixels based on saliency maps | Highly targeted but ineffective against VLMs |

#### True Black-Box Attacks

| Attack Name | Effectiveness | Approach | Implementation | Perceptual Quality | Error Types |
|-------------|--------------|----------|----------------|-------------------|------------|
| Pixel | High | Modifies a limited number of pixels (20 in our tests) | Uses evolutionary strategies (Differential Evolution) | SSIM=0.85 through binary search | Widespread errors across all question types |
| SimBA | High | Uses orthogonal perturbation vectors (DCT basis) | Query-efficient attack requiring only prediction scores | SSIM=0.85 through binary search | Affects both data reading and reasoning capabilities |
| ZOO | High | Zeroth-order optimization to estimate gradients | Coordinate-wise updates with Adam optimizer | SSIM=0.85 through binary search | Affects both data reading and reasoning capabilities |
| Boundary | High | Decision-based attack that walks along the decision boundary | Starts from a large perturbation and gradually reduces it | SSIM=0.85 through binary search | Affects both data reading and reasoning capabilities |
| HopSkipJump | Moderate | Decision-based attack that estimates gradients through queries | Creates global, diffuse perturbations | SSIM=0.85 through binary search | Widespread errors in data reading and calculations |
| Spatial | Moderate | Geometric transformations | Applies rotation, translation without pixel modifications | SSIM=0.85 through binary search | Affects spatial understanding and data relationships |
| Query-Efficient BB | Low | Adaptive, query-based perturbations | Estimates gradients using random sampling | SSIM=0.85 through binary search | Minimal impact on most question types |
| Square | Low | Randomly perturbs square regions | Creates localized, structured perturbations | SSIM=0.85 through binary search | Primarily affects complex calculations |
| GeoDA | Improvement | Subspace-based optimization using DCT basis | Efficient search in lower-dimensional subspaces | SSIM=0.85 through binary search | Improves model performance on certain tasks |

### Comparison of Black-Box Attacks

All black-box attacks were evaluated using the same SSIM threshold (0.85) for fair comparison:

| Attack Name | Effectiveness | Perturbation Pattern | Key Characteristics | Impact on VLM Performance |
|-------------|--------------|---------------------|---------------------|--------------------------|
| Pixel | High | Sparse, targeted modifications | Modifies only 20 specific pixels | Widespread errors across all question types |
| SimBA | High | Structured frequency perturbations | Uses DCT basis vectors for efficient queries | Affects both data reading and reasoning capabilities |
| ZOO | High | Gradient estimation | Uses zeroth-order optimization to estimate gradients | Affects both data reading and reasoning capabilities |
| Boundary | High | Decision boundary walking | Gradually reduces perturbation while staying adversarial | Affects both data reading and reasoning capabilities |
| HopSkipJump | Moderate | Global, diffuse perturbations | Estimates gradients through model queries | Widespread errors in data reading and calculations |
| Spatial | Moderate | Geometric transformations | Applies rotation, translation without pixel modifications | Affects spatial understanding and data relationships |
| Query-Efficient BB | Low | Adaptive, query-based perturbations | Estimates gradients using random sampling | Minimal impact on most question types |
| Square | Low | Localized, square-shaped patterns | Randomly perturbs square regions | Primarily affects complex calculations while preserving basic data reading |
| GeoDA | Improvement | Subspace optimization | Efficient search in lower-dimensional subspaces | Improves model performance on certain tasks |

This comparison demonstrates that the type and distribution of perturbation patterns significantly impact attack effectiveness, even when maintaining the same perceptual similarity constraints.

## Future Work

- Integrate additional Nano VLM models from Hugging Face:
  - yueliu1999/GuardReasoner-VL-Eco-3B
  - Lunzima/NQLSG-Qwen2-VL-2B-v2-Base
- Develop more sophisticated perceptual constraints for adversarial examples
- Evaluate finetuned models under adversarial conditions
- Compare effectiveness of adversarial training techniques
- Explore multi-modal adversarial attacks targeting both vision and language components
- Investigate the relationship between SSIM values and model performance degradation

## Database Structure

The evaluation results are stored in a SQLite database (`results/robustness.db`) designed for scalability and efficient querying. The database is structured to accommodate future expansion to multiple models, tasks, and evaluation scenarios.

### Schema Design

The main table `attack_comparison` has the following structure:

```
+----------------+----------+
| Column         | Type     |
+================+==========+
| id             | INTEGER  |
| task_name      | TEXT     |
| attack_type    | TEXT     |
| gpt4o_accuracy | REAL     |
| gpt4o_change   | REAL     |
| qwen_accuracy  | REAL     |
| qwen_change    | REAL     |
| gemma_accuracy | REAL     |
| gemma_change   | REAL     |
| timestamp      | TIMESTAMP|
+----------------+----------+
```

### Current Data

The table below shows the current data in the `attack_comparison` table:

| attack_type        |   gpt4o_accuracy |   gpt4o_change |   qwen_accuracy |   qwen_change |   gemma_accuracy |   gemma_change |
|:-------------------|-----------------:|---------------:|----------------:|--------------:|-----------------:|---------------:|
| Boundary           |            64.71 |           0.00 |           41.18 |        -41.18 |            29.41 |         -11.76 |
| CW-L0              |            23.53 |         -41.18 |           47.06 |        -35.29 |            41.18 |           0.00 |
| CW-L2              |            76.47 |          11.76 |           35.29 |        -47.06 |            41.18 |           0.00 |
| CW-L∞              |            47.06 |         -17.65 |           29.41 |        -52.94 |            41.18 |           0.00 |
| DeepFool           |            76.47 |          11.76 |           47.06 |        -35.29 |            47.06 |           5.88 |
| FGSM               |            58.82 |          -5.88 |           35.29 |        -47.06 |            47.06 |           5.88 |
| GeoDA              |            70.59 |           5.88 |           82.35 |          0.00 |            47.06 |           5.88 |
| HopSkipJump        |            76.47 |          11.76 |           47.06 |        -35.29 |            35.29 |          -5.88 |
| JSMA               |            76.47 |          11.76 |           82.35 |          0.00 |            41.18 |           0.00 |
| L-BFGS             |            70.59 |           5.88 |           35.29 |        -47.06 |            52.94 |          11.76 |
| Original           |            64.71 |           0.00 |           82.35 |          0.00 |            41.18 |           0.00 |
| PGD                |            64.71 |           0.00 |           70.59 |        -11.76 |            41.18 |           0.00 |
| Pixel              |            82.35 |          17.65 |           29.41 |        -52.94 |            35.29 |          -5.88 |
| Query-Efficient BB |            64.71 |           0.00 |           76.47 |         -5.88 |            41.18 |           0.00 |
| SimBA              |            82.35 |          17.65 |           41.18 |        -41.18 |            41.18 |           0.00 |
| Spatial            |            76.47 |          11.76 |           52.94 |        -29.41 |            52.94 |          11.76 |
| Square             |            70.59 |           5.88 |           76.47 |         -5.88 |            52.94 |          11.76 |
| ZOO                |            76.47 |          11.76 |           41.18 |        -41.18 |            42.86 |           1.68 |

### Scalability Features

The database is designed to scale to:
- 30+ VLM models (by adding additional model columns)
- Multiple tasks (chart, dashboard, etc.) via the `task_name` field
- All 17 attack types (already supported)
- 5000+ questions (via related tables if needed)

### Usage

To store evaluation results in the database:
```bash
python scripts/store_results_db.py
```

This creates or updates the SQLite database with the latest evaluation results, providing a centralized repository for all robustness metrics.
