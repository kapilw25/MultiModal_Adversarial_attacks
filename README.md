# 4-bit Quantized VLMs Under Attack: Benchmarking Vision-Language Models' Robustness Against Multi-Modal Adversarial Threats

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
  - `Phi3pt5_vision_4B.py`: Microsoft's Phi-3.5-vision-instruct model

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
| Qwen2.5-VL-3B | 3B | 2.3 GB | 17.66s | 3.70s | float16, NF4 | Complete, accurate ✅ |
| Qwen2.5-VL-7B | 7B | 5.7 GB ⬆️ | 33.21s | 2.24s | float16, NF4 | Complete, accurate ✅ |
| Qwen2-VL-2B | 2B | 1.5 GB | 9.78s | 2.99s | float16, NF4 | Complete, detailed ✅ |
| Gemma-3-4b-it | 4B | 3.1 GB | 19.58s | 4.80s ⬆️ | bfloat16, NF4 | Complete, concise ✅ |
| PaliGemma-3B | 3B | 2.2 GB | 28.00s | 0.49s ⬇️ | bfloat16, NF4 | Brief, minimal ❌ |
| DeepSeek-VL-1.3B | 1.3B | 1.6 GB | 7.26s ⬇️ | 2.96s | Optimized 4-bit | Complete, accurate ✅ |
| DeepSeek-VL-7B | 7B | 4.8 GB ⬆️ | 29.62s | 3.54s | Extreme 4-bit | Incomplete response ❌ |
| SmolVLM2-256M | 256M | 1.0 GB ⬇️ | 2.28s ⬇️ | 1.76s | float32 | Incomplete response ❌ |
| SmolVLM2-500M | 500M | 1.9 GB | 3.90s ⬇️ | 1.37s ⬇️ | float32 | Brief, accurate ✅ |
| SmolVLM2-2.2B | 2.2B | 1.4 GB | 16.27s | 2.92s | float16, 4-bit | Repetitive content ❌ |
| Florence-2-base | 0.23B | 0.52 GB ⬇️ | 31.72s | 0.60s ⬇️ | float16 | Complete, accurate ✅ |
| Florence-2-large | 0.77B | 1.59 GB | 155.34s ⬆️ | 0.66s ⬇️ | float16 | Complete, detailed ✅ |
| Moondream2-2B | 1.93B | 3.7 GB | 6.57s ⬇️ | 2.64s | float16 | Concise, accurate ✅ |
| GLM-Edge-V-2B | 2B | 3.7 GB | 7.48s ⬇️ | 30.58s ⬆️ | bfloat16 | Brief, accurate ✅ |
| InternVL3-1B | 1B | 0.87 GB ⬇️ | 5.14s ⬇️ | 6.47s | bfloat16, NF4 | Good, minor errors ⚠️ |
| InternVL3-2B | 2B | 1.72 GB | 9.51s | 6.85s | bfloat16, NF4 | Complete, accurate ✅ |
| InternVL2.5-4B | 4B | 2.73 GB | 14.56s | 5.89s | bfloat16, NF4 | Complete, accurate ✅ |
| Phi-3.5-vision-instruct | 4.15B | 2.34 GB | 16.77s | 64.15s ⬆️ | bfloat16, NF4 | Complete, accurate ✅ |

### 2. Transfer Attacks & True Black-Box Attacks

The project implements 17 adversarial attacks against VLMs, all executable via a single bash script (`scripts/adversarial_attack_runner.sh`). The script runs 8 transfer attacks (PGD, FGSM, CW variants, L-BFGS, JSMA, DeepFool) that use surrogate models, and 9 true black-box attacks (Square, HopSkipJump, Pixel, SimBA, Spatial, Query-Efficient, ZOO, Boundary, GeoDA) that don't require gradient information. All attacks maintain SSIM ≥ 0.85 for perceptual similarity.

### 4. Evaluation Framework
Located in `scripts/`, this module handles model evaluation and result analysis:
- `model_inference_pipeline.py`: Generates model responses for specific tasks
- `vqa_metrics_evaluator.py`: Analyzes results and calculates accuracy metrics
- `adversarial_attack_config.py`: Handles attack selection and configuration
- `metrics_persistence_manager.py`: Stores evaluation results in a SQLite database for efficient querying and analysis

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

To run all attacks sequentially:
```bash
chmod +x scripts/adversarial_attack_runner.sh
./scripts/adversarial_attack_runner.sh
```

### 3. Evaluate Model Performance

```bash
# Run model evaluation on adversarial images
python scripts/model_inference_pipeline.py

# Calculate accuracy metrics
python scripts/vqa_metrics_evaluator.py
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
python scripts/metrics_persistence_manager.py
```

This creates or updates the SQLite database with the latest evaluation results, providing a centralized repository for all robustness metrics.

## Search Term Analysis

The following table provides a comprehensive analysis of key research terms used in this project, based on academic search patterns and research impact:

| Key Term | Search Frequency (2022-2024) | Research Impact | Top Venues | Search Patterns & Trends |
|----------|--------------------------|-----------------|-------------|------------------------|
| 4-bit Quantized | ~3,200 papers | High in ML optimization | NeurIPS, ICLR, ICML | - Paired with "model compression" <br> - Rising in LLM efficiency <br> - Industry adoption focus |
| VLMs/Vision-Language Models | ~18,000 papers | Very High | CVPR, ICCV, ACL | - Rapid growth since 2023 <br> - Industry research priority <br> - Multi-modal integration |
| Under Attack | ~15,000 papers | High in security | USENIX, CCS, S&P | - Common in adversarial ML <br> - Security conference focus <br> - Practical applications |
| Benchmarking | ~25,000 papers | High in systems | MLSys, OSDI, SOSP | - Evaluation frameworks <br> - Performance metrics <br> - Comparative analysis |
| Robustness | ~45,000 papers | Very High | All Top ML venues | - Most cited security term <br> - Cross-domain impact <br> - Theoretical & practical |
| Multi-Modal | ~30,000 papers | Extremely High | NeurIPS, ICLR, AAAI | - Fastest growing term <br> - Cross-domain research <br> - Industry applications |
| Adversarial | ~20,000 papers | Very High | Security conferences | - Attack & defense methods <br> - ML security focus <br> - Risk assessment |
| Threats | ~12,000 papers | High in security | Security journals | - Often with "adversarial" <br> - Risk analysis <br> - Defense strategies |
