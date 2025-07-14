# Evaluating Nano Vision-Language Models' (VLMs) Robustness Against Cyber Security Attacks

This repository contains tools for evaluating small (4-bit, 3 Billion parameter) vision-language models (VLMs) under various multi-modal adversarial attacks, focusing on their robustness and performance degradation.

## System Design

The evaluation framework consists of four main modules:

### 1. VLM Inference Engine
Located in `local_model/`, this module handles model loading, quantization, and inference using a modular, plug-and-play architecture:

- `base_model.py`: Defines the abstract base class `BaseVLModel` that all VLM implementations must inherit from
- `model_classes.py`: Implements the factory pattern for creating model instances
- `model_utils.py`: Contains utility functions for memory management and model loading
- `models/`: Directory containing individual model implementations:
  - `Qwen25_3B.py`, `Qwen25_7B.py`, `Qwen2_2B.py`: Qwen family models
  - `Gemma3_4B.py`: Google's Gemma model
  - `Paligemma_3B.py`: Google's Paligemma model
  - `DeepSeek1_1pt3B.py`, `DeepSeek1_7B.py`: DeepSeek VL models
  - `SmolVLM2_pt25B_pt5B_2pt2B.py`: Unified implementation for SmolVLM2 models (256M, 500M, 2.2B)

#### 4-bit Quantization for Nano VLMs

The VLM Inference Engine supports model-specific 4-bit quantization strategies to optimize memory usage and inference speed:

- **Model-Specific Optimizations**:
  - **Qwen Models**: Use `torch.float16` compute dtype with standard NF4 quantization
  - **Google Models**: Use `torch.bfloat16` compute dtype for better numerical stability
  - **SmolVLM2 Models**: Use `torch.float32` for smaller models (256M, 500M) and `torch.float16` for larger model (2.2B)

- **Memory Management**:
  - Implemented memory monitoring decorators (`@memory_efficient`, `@time_inference`)
  - Model-specific memory optimizations (e.g., `low_cpu_mem_usage=True`)
  - Explicit GPU memory fraction control based on model size

- **Performance Impact**:
  - 4-bit quantization reduces memory usage by ~65% compared to FP16
  - Inference speed varies by model family and size
  - Type-compatible inference ensures numerical stability across different architectures

#### Modular Design for Adding New VLMs

The VLM Inference Engine uses a clean, modular architecture that makes it easy to add new vision-language models:

1. **Abstract Base Class**: `BaseVLModel` in `base_model.py` defines a standard interface that all models must implement
2. **Factory Pattern**: `model_classes.py` provides a simple factory function that creates the appropriate model instance
3. **Model Implementation**: Each model has its own implementation file in the `models/` directory that inherits from `BaseVLModel`

This design ensures that new models can be added with minimal changes to the existing codebase.

## VLM Performance Comparison

| Model | Size | GPU Memory | Loading Time | Inference Time | Quantization Strategy | Response Quality |
|-------|------|------------|--------------|----------------|----------------------|------------------|
| Qwen2.5-VL-3B | 3B | 2.3 GB | 23.31s | 4.73s ⬆️ | float16, NF4 | Complete, accurate ✅ |
| Qwen2.5-VL-7B | 7B | 5.7 GB ⬆️ | 46.34s ⬆️ | 2.19s | float16, NF4 | Complete, accurate ✅ |
| Qwen2-VL-2B | 2B | 1.5 GB | 12.98s | 2.67s | float16, NF4 | Complete, detailed ✅ |
| Gemma-3-4b-it | 4B | 3.1 GB | 26.07s | 4.14s ⬆️ | bfloat16, NF4 | Complete, concise ✅ |
| PaliGemma-3B | 3B | 2.2 GB | 28.00s | 0.50s ⬇️ | bfloat16, NF4 | Brief, minimal ❌ |
| DeepSeek-VL-1.3B | 1.3B | 1.6 GB | 10.40s ⬇️ | 2.54s | Optimized 4-bit | Complete, accurate ✅ |
| DeepSeek-VL-7B | 7B | 4.8 GB ⬆️ | 39.84s | 3.39s | Extreme 4-bit | Incomplete response ❌ |
| SmolVLM2-256M | 256M | 1.0 GB ⬇️ | 2.89s ⬇️ | 1.64s | float32 | Incomplete response ❌ |
| SmolVLM2-500M | 500M | 1.9 GB | 5.24s ⬇️ | 1.30s ⬇️ | float32 | Brief, accurate ✅ |
| SmolVLM2-2.2B | 2.2B | 1.4 GB | 21.55s | 2.53s | float16, 4-bit | Repetitive content ❌ |
| GLM-Edge-V-2B | 2B | 3.7 GB | 1.80s ⬇️ | 26.12s ⬆️ | bfloat16 | Brief, accurate ✅ |
| InternVL3-1B | 1B | 1.7 GB | 171.97s ⬆️ | 3.50s | bfloat16, NF4 | Good, minor errors ⚠️ |
| InternVL3-2B | 2B | 2.1 GB | 375.84s ⬆️ | 5.07s | bfloat16, NF4 | Complete, accurate ✅ |
| InternVL2.5-4B | 4B | 3.3 GB | 657.13s ⬆️ | 3.76s | bfloat16, NF4 | Incomplete, inaccurate ❌ |

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

This line graph shows how different Vision-Language Models perform under various adversarial attacks, measuring their robustness by plotting accuracy change percentage against different attack types.

#### Key Observations:

- **GPT-4o (Blue)**: Most robust model, showing positive accuracy changes for many attacks
- **Qwen-VL-3B (Orange)**: Most vulnerable model, showing significant negative accuracy changes
- **Gemma-VL-4B (Green)**: Shows moderate robustness with performance fluctuations

#### Attack Effectiveness:
- **Most Damaging**: CW-L∞ and Pixel attacks for Qwen-VL-3B (-53%), CW-L0 for GPT-4o (-41%)
- **Least Effective**: JSMA, GeoDA, and Square attacks show minimal impact across models

### Attack Characteristics

The following tables provide qualitative descriptions of the attacks and their characteristics. For detailed numerical results, please refer to the `attack_comparison` table in the [Database Structure](#database-structure) section.

#### Transfer-Based Attacks

| Attack Name | Effectiveness | Approach | Implementation | 
|-------------|--------------|----------|----------------|
| CW-L∞ | Highest degradation | Optimizes perturbations under L∞ norm constraint | Uses a surrogate model to generate adversarial examples |
| FGSM | High degradation | Single-step gradient-based method that adds perturbation in the direction of the gradient sign | Fast and efficient but creates more visible perturbations |
| CW-L2 | High degradation | Optimizes perturbations under L2 norm constraint | Creates more localized perturbations than CW-L∞ |
| L-BFGS | High degradation | Uses L-BFGS optimization to find minimal perturbations | Computationally expensive but effective |

#### True Black-Box Attacks

| Attack Name | Effectiveness | Approach | Implementation | Perceptual Quality |
|-------------|--------------|----------|----------------|-------------------|
| Pixel | High | Modifies a limited number of pixels (20 in our tests) | Uses evolutionary strategies (Differential Evolution) | SSIM=0.85 through binary search |
| SimBA | High | Uses orthogonal perturbation vectors (DCT basis) | Query-efficient attack requiring only prediction scores | SSIM=0.85 through binary search |
| ZOO | High | Zeroth-order optimization to estimate gradients | Coordinate-wise updates with Adam optimizer | SSIM=0.85 through binary search |
| Boundary | High | Decision-based attack that walks along the decision boundary | Starts from a large perturbation and gradually reduces it | SSIM=0.85 through binary search |

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
