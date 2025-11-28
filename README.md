# 4-bit Quantized VLMs Under Attack: Benchmarking Vision-Language Models' Robustness Against Multi-Modal Adversarial Threats

This repository contains tools for evaluating small (4-bit, 3 Billion parameter) vision-language models (VLMs) under various multi-modal adversarial attacks, focusing on their robustness and performance degradation.


## Execution Commands

### 1. Setup

```bash
python3 -m venv venv_MM
source venv_MM/bin/activate
  # Step 1: Install base requirements (without flash-attn/tensorrt)
  pip install -r requirements.txt
  # Step 2: Install flash-attn separately with updated build tools
  pip install flash-attn>=2.5.0 --no-build-isolation
  # Step 3: Install TensorRT for GPU optimization (optional)
  pip install torch-tensorrt>=1.4.0 --extra-index-url https://download.pytorch.org/whl/cu121
  # step4: Install nvidia-modelopt for Quantization Support
  pip install nvidia-modelopt 
  # step 5: # Download spaCy English model for text processing
  python -m spacy download en_core_web_sm 
  # step6: 
  sudo apt-get install jq
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
