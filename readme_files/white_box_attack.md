# White Box Attack Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                      WHITE BOX ATTACK WORKFLOW                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────┐
│ STEP 1: GENERATE ADVERSARIAL│
│         EXAMPLES           │
└───────────────┬─────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ White Box Attack Methods (Direct Access to Model Internals)         │
│                                                                     │
│ ┌───────────────┐    ┌───────────────┐    ┌───────────────────────┐ │
│ │ Access Model  │ → │ Extract        │ → │ Generate Adversarial  │ │
│ │ Gradients     │    │ Internal Reps │    │ Examples              │ │
│ └───────────────┘    └───────────────┘    └───────────────────────┘ │
│                                                                     │
│ Attack Types:                                                       │
│ 1. Gradient-based Search (FGSM, PGD)                                │
│    - Uses gradient of loss w.r.t input                              │
│    - Formula: x' = x + ε · sign(∇ₓJ(θ, x, y))                      │
│                                                                     │
│ 2. Greedy Coordinate Gradient (GCG)                                 │
│    - Iteratively modifies inputs following gradient information     │
│    - Maximizes target objective with full access to model internals │
│                                                                     │
│ 3. Token Embedding Attacks                                          │
│    - Manipulates embedding space directly                           │
│    - Requires knowledge of internal token representations           │
└─────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ OUTPUT: Adversarial examples crafted with white-box knowledge       │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────┐
│ STEP 2: PREPARE ADVERSARIAL │
│         TRAINING DATA       │
└───────────────┬─────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ datasets = load_dataset("unsloth/LaTeX_OCR", split = "train")       │
│                                                                     │
│ ┌───────────────┐    ┌───────────────┐    ┌───────────────────────┐ │
│ │ Load Dataset  │ → │ Add White-Box │ → │ Create Conversation   │ │
│ │ with Images   │    │ Adv. Examples │    │ Dataset               │ │
│ └───────────────┘    └───────────────┘    └───────────────────────┘ │
│                                                                     │
│ Conversation Format:                                                │
│ - User: Text instruction + Image (original or adversarial)          │
│ - Assistant: Expected response (correct answer despite attack)      │
└─────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ OUTPUT: Formatted dataset with adversarial examples                 │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 3: LOAD AND CONFIGURE  │
│         MODEL FOR FINETUNING│
└───────────────┬─────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ attack_models/finetuning/qwen2_5_vl_7b_vision.py                    │
│                                                                     │
│ ┌───────────────┐    ┌───────────────┐    ┌───────────────────────┐ │
│ │ Load Model    │ → │ Configure      │ → │ Apply LoRA            │ │
│ │ in 4-bit      │    │ Quantization  │    │ Adapters              │ │
│ └───────────────┘    └───────────────┘    └───────────────────────┘ │
│                                                                     │
│ Model Configuration:                                                │
│ - Model: "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit"                 │
│ - 4-bit quantization for memory efficiency                          │
│ - Gradient checkpointing for longer contexts                        │
│                                                                     │
│ LoRA Configuration:                                                 │
│ - finetune_vision_layers = True                                     │
│ - finetune_language_layers = True                                   │
│ - r = 16 (rank for low-rank adaptation)                             │
│ - lora_alpha = 16                                                   │
└─────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 4: FINE-TUNE MODEL     │
│         WITH ADVERSARIAL    │
│         TRAINING           │
└───────────────┬─────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ trainer = SFTTrainer(...)                                           │
│ trainer_stats = trainer.train()                                     │
│                                                                     │
│ ┌───────────────┐    ┌───────────────┐    ┌───────────────────────┐ │
│ │ Configure     │ → │ Execute        │ → │ Monitor Training      │ │
│ │ SFT Trainer   │    │ Training      │    │ Metrics               │ │
│ └───────────────┘    └───────────────┘    └───────────────────────┘ │
│                                                                     │
│ Training Configuration:                                             │
│ - per_device_train_batch_size = 2                                   │
│ - gradient_accumulation_steps = 4                                   │
│ - learning_rate = 2e-4                                              │
│ - max_steps = 30 (can be increased for full training)               │
│ - Uses UnslothVisionDataCollator for vision data                    │
└─────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ OUTPUT: Fine-tuned model with LoRA adapters                         │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 5: SAVE FINE-TUNED    │
│         MODEL              │
└───────────────┬─────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ model.save_pretrained("lora_model")                                 │
│ tokenizer.save_pretrained("lora_model")                             │
│                                                                     │
│ ┌───────────────┐    ┌───────────────┐    ┌───────────────────────┐ │
│ │ Save LoRA     │ → │ Save          │ → │ Optional: Save as      │ │
│ │ Adapters      │    │ Tokenizer     │    │ 16-bit Full Model     │ │
│ └───────────────┘    └───────────────┘    └───────────────────────┘ │
│                                                                     │
│ Saving Options:                                                     │
│ - Local saving: save_pretrained("lora_model")                       │
│ - HuggingFace Hub: push_to_hub("your_name/lora_model")             │
│ - Full 16-bit model: save_pretrained_merged("unsloth_finetune")     │
└─────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 6: EVALUATE FINE-TUNED│
│         MODEL              │
└───────────────┬─────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ FastVisionModel.for_inference(model)                                │
│                                                                     │
│ ┌───────────────┐    ┌───────────────┐    ┌───────────────────────┐ │
│ │ Load Test     │ → │ Generate      │ → │ Compare with          │ │
│ │ Images        │    │ Predictions   │    │ Original Model        │ │
│ └───────────────┘    └───────────────┘    └───────────────────────┘ │
│                                                                     │
│ Evaluation Process:                                                 │
│ - Test on original images                                           │
│ - Test on adversarial images                                        │
│ - Compare performance before and after fine-tuning                  │
└─────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ FINAL OUTPUT: Robustness metrics showing model performance          │
│               before and after adversarial fine-tuning              │
│                                                                     │
│ Potential Outcomes:                                                 │
│ - Improved robustness against specific attack types                 │
│ - Maintained performance on clean images                            │
│ - Trade-offs between robustness and general performance             │
└─────────────────────────────────────────────────────────────────────┘
```

## Technical Details

### 1. White Box Attack Methods

**Gradient-based Search Attacks (FGSM, PGD):**
- Use the gradient of the loss with respect to the input to find adversarial examples
- Require direct access to model gradients, making them white-box by definition
- FGSM formula: x' = x + ε · sign(∇ₓJ(θ, x, y))
- PGD formula: xₜ₊₁' = Proj_ε(xₜ' + α · sign(∇ₓJ(θ, xₜ', y)))
- These attacks directly exploit the model's decision boundaries through gradient information

**Greedy Coordinate Gradient (GCG) Attacks:**
- Iteratively modify inputs by following gradient information to maximize a target objective
- Require full access to model internals, including forward and backward passes
- Work by identifying the most influential input dimensions and perturbing them optimally
- Can be more computationally expensive but often produce more effective adversarial examples

**Token Embedding Attacks:**
- Sophisticated white-box attacks that manipulate the embedding space directly
- Require knowledge of how the model represents tokens internally
- Can target specific neurons or attention patterns within the model
- Often more effective for text or multi-modal inputs where discrete tokens are used

### 2. White Box vs. Black Box Access

**White Box Access:**
- Complete access to model architecture and parameters
- Ability to modify model weights through fine-tuning
- Direct access to gradients and internal representations
- Can implement adversarial training by incorporating adversarial examples
- Enables more powerful and targeted attacks than black-box scenarios

**Black Box Access:**
- Limited to API-like interaction with the model
- Can only observe inputs and outputs
- Must rely on transferability of adversarial examples
- Cannot directly modify model parameters
- Typically requires more queries to the model to achieve effective attacks

### 3. Parameter-Efficient Fine-Tuning with LoRA

Low-Rank Adaptation (LoRA) is a technique that freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture:

- Original weight update: W = W₀ + ΔW
- LoRA parameterization: W = W₀ + BA
- Where B ∈ ℝᵐˣʳ, A ∈ ℝʳˣⁿ, and r << min(m,n)

This approach:
- Reduces trainable parameters by 99% compared to full fine-tuning
- Maintains most of the model's original capabilities
- Allows efficient adaptation to new tasks or domains
- Can be merged with original weights for deployment

### 4. Adversarial Training Process

Adversarial training involves:
1. Generating adversarial examples that fool the model using white-box access
2. Including these examples in the training data with correct labels
3. Fine-tuning the model to correctly handle these adversarial inputs

The mathematical objective can be formulated as:
- min_θ E(x,y)~D[max_δ∈S L(θ, x+δ, y)]

Where:
- θ represents the model parameters
- (x,y) are input-output pairs from distribution D
- δ represents the adversarial perturbation within constraint set S
- L is the loss function

### 5. Vision-Language Model Architecture

The Qwen2.5-VL-7B model combines:
- A vision encoder that processes images into embeddings
- A language model that processes text and vision embeddings
- Cross-attention mechanisms that connect visual and textual information

Fine-tuning can target:
- Vision layers only (visual perception)
- Language layers only (reasoning and generation)
- Attention mechanisms (cross-modal integration)
- MLP layers (feature processing)
- Any combination of the above
