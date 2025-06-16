# Multi-modal Self-instruct Evaluation Pipeline

This repository contains tools for evaluating vision-language models (VLMs) on a collection of high-quality abstract images and corresponding question-answer pairs.

## Overview

The Multi-modal Self-instruct dataset includes:
- 11,193 high-quality abstract images with corresponding question-answer pairs
- 62,476 training instructions covering tables, charts, and road maps
- Evaluation pipeline for testing model performance on visual reasoning tasks

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

## Evaluation Pipeline

The evaluation pipeline consists of two main scripts:

### 1. `eval_model.py`

This script generates evaluation results for a specific model on a specific task.

```bash
cd scripts
python eval_model.py
```

Configuration:
- Set `engine` variable to specify the model (e.g., 'gpt4o')
- Set `task` variable to specify the task (e.g., 'chart')
- The script will load data from `../{engine}/eval_{task}.json`
- Results will be saved to `../{engine}/eval_{engine}_{task}_{random_count}.json`

### 2. `eval_vqa.py`

This script analyzes the evaluation results and calculates accuracy metrics.

```bash
cd scripts
python eval_vqa.py
```

Functions:
- `evaluator('../{engine}/eval_{engine}_{task}_{count}.json')` - Tests accuracy on 7 tasks (charts, tables, dashboards, flowcharts, relation graphs, floor plans, and visual puzzles)
- `evaluator_map('../{engine}/eval_{engine}_{task}_{count}.json')` - Tests accuracy on simulated maps

## Recent Progress

- Fixed file path issues in evaluation scripts to use relative paths
- Implemented proper environment variable loading for API keys
- Added error handling for missing data
- Successfully ran evaluation on GPT-4o model with chart task
- Achieved 100% accuracy on the test set

## Directory Structure

```
Multi-modal-Self-instruct/
├── .env                    # Environment variables (API keys)
├── data/                   # Dataset files
│   └── test_extracted/     # Test images
├── gpt4o/                  # Model-specific evaluation files
│   └── eval_chart.json     # Evaluation data for charts
├── scripts/                # Evaluation scripts
│   ├── eval_model.py       # Script to generate model responses
│   ├── eval_vqa.py         # Script to calculate accuracy metrics
│   └── llm_tools.py        # Utilities for API calls
└── venv_MM/                # Virtual environment
```

## Usage

1. Set up your environment and API key
2. Run `eval_model.py` to generate model responses
3. Run `eval_vqa.py` to calculate accuracy metrics
4. Modify the scripts to evaluate different models or tasks

## Notes

- The evaluation pipeline supports various visual reasoning tasks
- You can customize the evaluation by modifying the model and task parameters
- The pipeline uses the OpenAI API for model inference
