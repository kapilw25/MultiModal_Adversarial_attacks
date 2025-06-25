# Black Box Attack Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                      BLACK BOX ATTACK WORKFLOW                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────┐
│ STEP 1: GENERATE ADVERSARIAL│
│         EXAMPLES           │
└───────────────┬─────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ attack_models/black_box_attacks/v3_fgsm_attack.py                   │
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
│ OUTPUT: data/test_extracted_adv_fgsm/chart/20231114102825506748.png │
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
│ OUTPUT: results/Qwen25_VL_3B/eval_Qwen25_VL_3B_chart_17.json        │
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

### 3. Black Box Attack Strategy

This is a "black-box transfer attack" strategy:
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
