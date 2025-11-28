# Pipeline Documentation

Detailed documentation for each script in the adversarial attack evaluation pipeline.

## Pipeline Overview

```
attack_runner.py → model_inference_vLLM.py → model_evaluation.py → model_benchmark_robustness.py
     ↓                    ↓                     ↓                      ↓
 Adversarial          VLM Responses        Accuracy Scores       Robustness Metrics
   Images
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WRITE TO DATABASE (4 scripts)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  attack_runner.py ──────► attack_executions table                           │
│                                                                             │
│  model_inference_vLLM.py ► inference_results table                           │
│                                                                             │
│  model_evaluation.py ───► results_evaluation table (copy + evaluate)        │
│                                                                             │
│  model_benchmark_robustness.py ► model_robustness_matrix + views            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                     results/centralized_pipeline.db
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              VISUALIZATION (integrated into model_benchmark_robustness.py)  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  utils/model_visualizer.py ◄── called by model_benchmark_robustness.py     │
│                              ──► results/research_plots/*.png               │
│  (Interactive menu: metrics only, plots only, or both)                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Script | Direction | Table(s) |
|--------|-----------|----------|
| `attack_runner.py` | WRITE | `attack_executions` |
| `model_inference_vLLM.py` | WRITE | `inference_results` |
| `model_evaluation.py` | WRITE | `results_evaluation` |
| `model_benchmark_robustness.py` | WRITE | `model_robustness_matrix` + views |
| `utils/model_visualizer.py` | READ | All tables → PNG plots (called by model_benchmark_robustness.py) |

---

## 1. `scripts/attack_runner.py`

Generates adversarial images using whitebox and blackbox attacks.

### Dependencies

| Type     | Module                                   |
|----------|------------------------------------------|
| Internal | `scripts/utils/centralized_database.py`  |
| Internal | `attack_models/utils.py` (query_counter) |
| Internal | `attack_models/white_box_universal.py`   |
| Internal | `attack_models/black_box_universal.py`   |
| External | numpy, PIL, subprocess, logging, pathlib |

### Input Files

| File                         | Purpose                            |
|------------------------------|------------------------------------|
| `data/processed_images.json` | Lists images per task              |
| `data/clean/{task}/*.png`    | Original clean images              |
| `data/adversarial/`          | Checks existing adversarial images |

### Output Files

| File                                                        | Purpose                     |
|-------------------------------------------------------------|-----------------------------|
| `data/adversarial/whitebox/{attack}/{eps}/{task}/*.png`     | Whitebox adversarial images |
| `data/adversarial/blackbox/{attack}/{eps}/{task}/*.png`     | Blackbox adversarial images |
| `results/centralized_pipeline.db` → `attack_executions`     | Attack metadata             |

### Interactive Menu

```
🎯 Universal Attack Runner (White-Box + Black-Box) - 100% EPSILON CONTROL

[STEP 1: TASK SELECTION]
  [1] Chart interpretation (4 images)
  [2] Table data extraction (3 images)
  [3] Road map navigation (3 images)
  [4] Dashboard analysis (3 images)
  [5] Flowchart understanding (3 images)
  [6] Relation graph analysis (3 images)
  [7] Planar layout interpretation (3 images)
  [8] Visual puzzle solving (3 images)
  [9] ALL tasks (25 images total)

[STEP 2: ATTACK CATEGORY]
  [1] White-Box Attacks (5 attacks)
  [2] Black-Box Attacks (4 attacks)
  [3] All Attacks (9 attacks total)

[STEP 3: ATTACK TYPE] (if whitebox selected)
  [1] FGSM Attack
  [2] PGD Attack
  [3] AutoPGD Attack
  [4] AutoConjugateGradient Attack
  [5] BasicIterativeMethod Attack
  [6] ALL WHITEBOX ATTACKS
  → Supports comma-separated: "1,2,3"

[STEP 4: EPSILON LEVEL]
  [1] Minimal (ε = 4/255 ≈ 0.016)
  [2] Standard (ε = 8/255 ≈ 0.031)
  [3] Moderate (ε = 16/255 ≈ 0.063)
  [4] ALL levels
  [5] Custom epsilon (enter manually)

[STEP 5: REPLACEMENT MODE] (if existing images detected)
  [Y] COMPLETE REPLACEMENT - Delete all, start fresh
  [N] SELECTIVE OVERWRITE - Keep existing, overwrite selected
```

---

## 2. `scripts/model_inference_vLLM.py`

Runs VLM inference on clean and adversarial images using vLLM batch processing.

### Dependencies

| Type     | Module                                    |
|----------|-------------------------------------------|
| Internal | `scripts/utils/centralized_database.py`   |
| Internal | `scripts/utils/ground_truth_populator.py` |
| Internal | `scripts/utils/text_cleaner.py`           |
| External | vllm (LLM, SamplingParams)                |
| External | torch, tqdm, gc                           |

> **Note:** vLLM loads models directly from HuggingFace, bypassing `local_model/` directory.

### Input Files

| File                                                       | Purpose                 |
|------------------------------------------------------------|-------------------------|
| `results/centralized_pipeline.db` → `ground_truth_questions` | Questions per image     |
| `results/centralized_pipeline.db` → `attack_executions`      | Adversarial image paths |
| `data/clean/{task}/*.png`                                    | Clean images            |
| `data/adversarial/**/*.png`                                  | Adversarial images      |

### Output Files

| File                                                        | Purpose       |
|-------------------------------------------------------------|---------------|
| `results/centralized_pipeline.db` → `inference_results`     | VLM responses |

### Interactive Menu

```
================================================================================
🚀 SIMPLIFIED VLM INFERENCE ENGINE
================================================================================
Goal: Populate inference_results table with VLM answers
Source: ground_truth_questions + attack_executions tables

🔍 Checking ground truth data...

============================================================
🔧 ENGINE SELECTION
============================================================
Select VLM engine(s):
  [ 1] Qwen25_VL_3B
  [ 2] Qwen25_VL_7B
  [ 3] Qwen2_VL_2B
  [ 4] Gemma3_VL_4B
  [ 5] PaliGemma_VL_3B
  [ 6] DeepSeek1_VL_1pt3B
  [ 7] DeepSeek1_VL_7B
  [ 8] SmolVLM2_pt25B
  [ 9] SmolVLM2_pt5B
  [10] SmolVLM2_2pt2B
  [11] InternVL3_1B
  [12] InternVL3_2B
  [13] InternVL25_4B
  [14] Moondream2_2B
  [15] LLAVA_1pt5_7B
  [16] LLAVA_v1pt6_Mistral_7B
  [17] ALL ENGINES

Enter choice (1-17): _

📋 Configuration:
   🔧 Engines: [selected]
   🖼️  Processing: Both clean and adversarial images
```

---

## 3. `scripts/model_evaluation.py`

Evaluates VLM responses against ground truth answers.

### Dependencies

| Type     | Module                           |
|----------|----------------------------------|
| Internal | None (standalone)                |
| External | nltk, rouge, sqlite3, tqdm, re   |

### Input Files

| File                                                  | Purpose                   |
|-------------------------------------------------------|---------------------------|
| `results/centralized_pipeline.db` → `inference_results` | VLM responses to evaluate |

### Output Files

| File                                                         | Purpose                                                                  |
|--------------------------------------------------------------|--------------------------------------------------------------------------|
| `results/centralized_pipeline.db` → `results_evaluation`     | Evaluated results with `is_correct`, `confidence_score`, `evaluation_method` |

### Terminal Output

```
================================================================================
🧮 SIMPLIFIED MODEL EVALUATION ENGINE (Database-First)
================================================================================
Goal: Copy inference_results → results_evaluation table with evaluation columns
Columns added: is_correct, confidence_score, evaluation_method

✅ Found {count} records in inference_results table
🔄 Creating results_evaluation table from inference_results...
📊 Evaluating {total} records in results_evaluation table...

============================================================
📊 EVALUATION SUMMARY
============================================================
Overall: {correct}/{total} correct ({accuracy}%)

📱 Accuracy by Model:
   Qwen25_VL_3B: 45/100 (45.00%)
   ...

⚔️  Accuracy by Attack Type:
   original: 80/100 (80.00%)
   fgsm: 40/100 (40.00%)
   ...

❓ Accuracy by Question Type:
   SUMMARY: 30/50 (60.00%)
   ...

🔍 Evaluation Method Distribution:
   numerical_exact: 50 cases (90.0% success rate)
   string_containment: 30 cases (70.0% success rate)
   ...

💾 Database: results/centralized_pipeline.db → results_evaluation table
```

**NO INTERACTIVE MENU** - Runs automatically without user input.

---

## 4. `scripts/model_benchmark_robustness.py`

Calculates robustness and degradation metrics with interactive menu.

### Dependencies

| Type     | Module                              |
|----------|-------------------------------------|
| Internal | None (standalone)                   |
| External | numpy, scipy.stats, sqlite3, tqdm   |

### Input Files

| File                                                   | Purpose           |
|--------------------------------------------------------|-------------------|
| `results/centralized_pipeline.db` → `results_evaluation` | Evaluated results |

### Output Files

| File                                                                                               | Purpose            |
|----------------------------------------------------------------------------------------------------|--------------------|
| `results/centralized_pipeline.db` → `model_robustness_matrix`                                      | Robustness metrics |
| `results/centralized_pipeline.db` → Views: `model_comparison`, `task_robustness`, `attack_effectiveness` | Aggregated views   |

### Terminal Output

```
================================================================================
🚀 MULTI-DIMENSIONAL MODEL PERFORMANCE ANALYZER
================================================================================
Goal: Calculate robustness/degradation metrics from results_evaluation table
Output: model_robustness_matrix + aggregation views

✅ Found {count} records in results_evaluation table
✅ Created model_robustness_matrix table

🔍 Discovering available data in results_evaluation table...
   Available data: X models, Y tasks, Z attack types, W epsilon levels

📊 Calculating multi-dimensional robustness metrics...

================================================================================
📊 MULTI-DIMENSIONAL ROBUSTNESS ANALYSIS SUMMARY
================================================================================

🤖 MODEL ROBUSTNESS RANKING:
   Model_A: X.XX% avg degradation, Y.YY% worst case (Z scenarios)
   ...

⚔️ ATTACK EFFECTIVENESS RANKING:
   attack_type (epsilon X.XX): Y.YY% avg impact, Z/N severe cases
   ...

📋 TASK VULNERABILITY ANALYSIS:
   task_name: X.XX% average vulnerability across attacks
   ...

💾 Database: results/centralized_pipeline.db → model_robustness_matrix + views
```

### Interactive Menu

```
============================================================
  ROBUSTNESS ANALYSIS OPTIONS
============================================================

  [1] Calculate robustness metrics only
      → Creates model_robustness_matrix + aggregation views

  [2] Generate plots only
      → Uses existing metrics (if available)

  [3] Both (metrics + plots)
      → Full analysis pipeline

------------------------------------------------------------
Enter choice [1-3]: _
```

**Use `--auto` flag to skip menu and run full pipeline.**

---

## Database Schema

All scripts use a centralized SQLite database: `results/centralized_pipeline.db`

### Tables

| Table                    | Created By              | Purpose                          |
|--------------------------|-------------------------|----------------------------------|
| `attack_executions`      | `attack_runner.py`      | Attack metadata and file paths   |
| `ground_truth_questions` | `ground_truth_populator.py` | Questions and expected answers |
| `inference_results`      | `model_inference_vLLM.py` | Raw VLM responses              |
| `results_evaluation`     | `model_evaluation.py`   | Evaluated responses with scores  |
| `model_robustness_matrix`| `model_benchmark_robustness.py` | Robustness metrics         |

### Views

| View                  | Created By            | Purpose                              |
|-----------------------|-----------------------|--------------------------------------|
| `model_comparison`    | `model_benchmark_robustness.py`| Compare models by robustness    |
| `task_robustness`     | `model_benchmark_robustness.py`| Analyze task-level vulnerability|
| `attack_effectiveness`| `model_benchmark_robustness.py`| Rank attacks by effectiveness   |

---

## 5. `scripts/utils/model_visualizer.py`

Generates publication-ready visualization plots for research paper.

> **Note:** This is now a utility module called by `model_benchmark_robustness.py` (not a standalone script).
> Use interactive menu option [1] for metrics only, or [2] for plots only.

### Dependencies

| Type     | Module                              |
|----------|-------------------------------------|
| Internal | None (standalone)                   |
| External | pandas, matplotlib, seaborn, numpy  |

### Input Files

| File                                                   | Purpose              |
|--------------------------------------------------------|----------------------|
| `results/centralized_pipeline.db` → `inference_results`   | Performance metrics  |
| `results/centralized_pipeline.db` → `results_evaluation`   | Accuracy data        |
| `results/centralized_pipeline.db` → `attack_executions`    | Attack metadata     |

### Output Files

| File                                                    | Purpose                        |
|---------------------------------------------------------|--------------------------------|
| `results/research_plots/01_baseline_performance.png`    | Model baseline accuracy        |
| `results/research_plots/03_attack_effectiveness_heatmap.png` | Attack impact heatmap     |
| `results/research_plots/04_raw_accuracy_heatmap.png`    | Raw accuracy matrix            |
| `results/research_plots/05_model_family_robustness.png` | Family-level robustness        |
| `results/research_plots/06_size_category_robustness.png`| Size-based analysis            |
| `results/research_plots/08_architecture_vulnerability_analysis.png` | Vulnerability ranking |
| `results/research_plots/10_attack_transferability_analysis.png` | Attack effectiveness    |
| `results/research_plots/12_size_vs_memory.png`          | Size vs GPU memory             |
| `results/research_plots/13_top_models_radar.png`        | Multi-dimensional comparison   |
| `results/research_plots/15_attack_raw_metrics_comparison.png` | Attack metrics scatter  |

### Terminal Output

```
============================================================
VLM Data Analysis & Visualization
Database: centralized_pipeline.db
============================================================
Connected to database: results/centralized_pipeline.db
Loaded 16 models, X found in database
Generated color palettes: Y families, Z sizes

Loading robustness data...
Loaded robustness data: N records

Generating robustness plots...
Created: 01_baseline_performance.png
Created: 03_attack_effectiveness_heatmap.png
...

Plot generation complete!
Generated X plots in: results/research_plots
```

**NO INTERACTIVE MENU** - Runs automatically without user input.
