# 4-bit Quantized VLMs Under Attack: Benchmarking Vision-Language Models' Robustness Against Multi-Modal Adversarial Threats

This repository contains tools for evaluating small (4-bit, 3 Billion parameter) vision-language models (VLMs) under various multi-modal adversarial attacks, focusing on their robustness and performance degradation.


## Execution Commands

### 1. Setup

```bash
# Create and activate virtual environment
python3 -m venv venv_MM
source venv_MM/bin/activate

# Step 1: Install base requirements (includes vLLM for batch inference)
# Note: --no-build-isolation uses system torch for packages that require it during build
pip install -r requirements.txt --no-build-isolation

# Step 2: Install flash-attn (prebuilt wheel for torch 2.9.0 + CUDA 12.8)
# Source: https://github.com/mjun0812/flash-attention-prebuild-wheels
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.17/flash_attn-2.8.3+cu128torch2.9-cp310-cp310-linux_x86_64.whl

# Step 3: Verify vLLM installation
python -c "from vllm import LLM; print('vLLM installed successfully')"
```

### 2. Run Pipeline

```bash
source venv_MM/bin/activate

# Step 1: Generate adversarial images (interactive menu)
python scripts/attack_runner.py

# Step 2: Run VLM inference on clean + adversarial images (vLLM batch)
python scripts/model_inference_vLLM.py

# Step 3: Evaluate model responses (automatic)
python scripts/model_evaluation.py

# Step 4: Calculate robustness metrics + generate plots (interactive menu)
python scripts/model_benchmark_robustness.py
# Note: Use --auto to skip menu and run full pipeline
```

> **Detailed documentation**: See [docs/PIPELINE.md](docs/PIPELINE.md) for script dependencies, I/O files, and interactive menus.


## System Design

The evaluation framework consists of four main scripts in `scripts/`:

| Script | Purpose |
|--------|---------|
| `attack_runner.py` | Generate adversarial images (whitebox + blackbox attacks) |
| `model_inference_vLLM.py` | Run VLM inference on clean + adversarial images (vLLM batch) |
| `model_evaluation.py` | Evaluate responses against ground truth |
| `model_benchmark_robustness.py` | Calculate robustness metrics + generate plots |

## Attack Workflow

```
┌────────────────────────────┐     ┌────────────────────────────┐     ┌────────────────────────────┐     ┌────────────────────────────┐
│  STEP 1: GENERATE          │     │  STEP 2: RUN VLM           │     │  STEP 3: EVALUATE          │     │  STEP 4: METRICS + PLOTS   │
│  ADVERSARIAL IMAGES        │────►│  INFERENCE                 │────►│  RESPONSES                 │────►│  (ROBUSTNESS ANALYSIS)     │
└────────────────────────────┘     └────────────────────────────┘     └────────────────────────────┘     └────────────────────────────┘
             │                                  │                                  │                                  │
             ▼                                  ▼                                  ▼                                  ▼
┌────────────────────────────┐     ┌────────────────────────────┐     ┌────────────────────────────┐     ┌────────────────────────────┐
│ scripts/attack_runner.py   │     │scripts/model_inference_vLLM│     │ scripts/model_evaluation.py│     │  model_benchmark_robustness │
│                            │     │                            │     │                            │     │                            │
│ • Load clean images        │     │ • Load VLM (4-bit quant)   │     │ • Compare response to      │     │ • Calculate baseline vs    │
│ • Apply attack algorithm   │     │ • Process clean images     │     │   ground truth             │     │   attack accuracy          │
│ • Control epsilon bounds   │     │ • Process adversarial imgs │     │ • Score correctness        │     │ • Compute degradation %    │
│ • Save adversarial images  │     │ • Save VLM responses       │     │ • Identify eval method     │     │ • Rank model robustness    │
│                            │     │                            │     │                            │     │ • Generate research plots  │
└────────────────────────────┘     └────────────────────────────┘     └────────────────────────────┘     └────────────────────────────┘
             │                                  │                                  │                                  │
             ▼                                  ▼                                  ▼                                  ▼
┌────────────────────────────┐     ┌────────────────────────────┐     ┌────────────────────────────┐     ┌────────────────────────────┐
│ OUTPUT:                    │     │ OUTPUT:                    │     │ OUTPUT:                    │     │ OUTPUT:                    │
│ • data/adversarial/**/*.png│     │ • inference_results table  │     │ • results_evaluation table │     │ • model_robustness_matrix  │
│ • attack_executions table  │     │                            │     │                            │     │ • aggregation views        │
│                            │     │                            │     │                            │     │ • results/research_plots/  │
└────────────────────────────┘     └────────────────────────────┘     └────────────────────────────┘     └────────────────────────────┘
```
