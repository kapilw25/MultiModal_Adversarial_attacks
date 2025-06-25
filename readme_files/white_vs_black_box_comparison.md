# White-Box vs. Black-Box FGSM Attack Comparison

This document compares the implementation differences between white-box and black-box FGSM (Fast Gradient Sign Method) attacks on Vision-Language Models (VLMs).

## Comparison Table

| Step | White-Box FGSM Attack | Black-Box FGSM Attack |
|------|------------------------|------------------------|
| **1. Model Loading** | • Loads the actual target VLM model (Qwen2.5-VL-3B)<br>• Requires 16-bit precision to enable gradient computation<br>• Needs more GPU memory (6-7GB)<br>• May require CPU fallback for large models | • Loads a substitute model (ResNet50)<br>• Uses standard precision, no need for gradient access to target VLM<br>• Requires less GPU memory (~2-3GB)<br>• Can run on modest GPU hardware |
| **2. Input Processing** | • Uses the VLM's own processor<br>• Prepares inputs specifically for the VLM architecture<br>• Maintains the image tensor's gradient tracking<br>• Processes both image and text inputs<br>• Requires a question parameter | • Uses standard image preprocessing for ResNet50<br>• No question processing (classification model doesn't use text)<br>• Simpler preprocessing pipeline<br>• Processes only image inputs<br>• No question parameter needed |
| **3. Gradient Computation** | • Calculates loss using the VLM's own loss function<br>• Performs backpropagation through the actual VLM<br>• Gets precise gradients that reflect the VLM's decision boundaries<br>• Requires PyTorch's autograd functionality | • Calculates loss using the substitute model's classification loss<br>• Performs backpropagation through the substitute model<br>• Gets gradients that approximate but don't exactly match the VLM's<br>• Uses ART library's FastGradientMethod implementation |
| **4. Adversarial Example Generation** | • Uses gradients that directly reflect the VLM's vulnerabilities<br>• Creates perturbations optimized specifically for the target model<br>• Typically produces more effective adversarial examples<br>• Different questions can produce different adversarial examples | • Uses gradients from the substitute model<br>• Creates perturbations optimized for the substitute model<br>• Relies on transferability of adversarial examples between models<br>• Generally less effective due to the "transfer gap"<br>• Question-agnostic perturbations |
| **5. Output Storage** | • Saves to `data/test_extracted_adv_white_box_fgsm/` directory | • Saves to `data/test_extracted_adv_fgsm/` directory |

## Implementation Code Comparison

### White-Box FGSM Attack (Key Components)

```python
# Load the actual VLM model with gradient access
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.float16,
    device_map=device
)
model.train()  # Enable gradient computation

# Process inputs with gradient tracking
image_tensor.requires_grad = True
inputs = processor(
    text=[text],  # Text question is required
    images=image_tensor,
    padding=True,
    return_tensors="pt",
)

# Compute gradients directly from VLM
with torch.enable_grad():
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
loss.backward()

# Generate adversarial example using VLM gradients
image_grad = image_tensor.grad.sign()
perturbed_image = image_tensor + eps * image_grad
```

### Black-Box FGSM Attack (Key Components)

```python
# Import ART library implementation
from art.attacks.evasion import FastGradientMethod

# Load substitute model (ResNet50)
model = models.resnet50(pretrained=True)
model.to(device).eval()
classifier = PyTorchClassifier(
    model=model,
    clip_values=(0.0, 1.0),
    loss=torch.nn.CrossEntropyLoss(),
    input_shape=(3, 224, 224),
    nb_classes=1000
)

# Process image for substitute model (no text/question needed)
img_tensor = transform(image).unsqueeze(0).numpy()

# Create FGSM attack using substitute model
attack = FastGradientMethod(
    estimator=classifier,
    norm=np.inf,
    eps=eps,
    targeted=targeted,
    batch_size=1
)

# Generate adversarial example using substitute model gradients
adv_image = attack.generate(x=img_tensor)
```

## Advantages and Disadvantages

| Aspect | White-Box Attacks | Black-Box Attacks |
|--------|-------------------|-------------------|
| **Effectiveness** | ✅ More effective adversarial examples | ❌ Less effective due to transfer gap |
| **Target Precision** | ✅ Directly targets the model's actual vulnerabilities | ❌ Relies on adversarial transferability between models |
| **Transfer Gap** | ✅ No transfer gap between models | ❌ Subject to transfer gap between substitute and target models |
| **Model Access** | ❌ Requires direct access to model architecture and parameters | ✅ Works without direct access to target model |
| **Computational Requirements** | ❌ Higher computational and memory requirements | ✅ Lower computational and memory requirements |
| **Hardware Requirements** | ❌ May require high-end GPUs or CPU fallback | ✅ Works on modest GPU hardware |
| **API Compatibility** | ❌ Not feasible when only API access is available | ✅ Can be applied to models available only through APIs |
| **Perturbation Size** | ✅ Can achieve effects with smaller perturbations | ❌ May require stronger perturbations to achieve the same effect |
| **Implementation Complexity** | ❌ More complex implementation requiring model internals | ✅ Simpler implementation using substitute models |
| **Real-world Applicability** | ❌ Limited to scenarios with full model access | ✅ Applicable in most real-world threat scenarios |
| **Multi-modal Context** | ✅ Can incorporate text context (questions) | ❌ Cannot incorporate text context |
| **Library Dependencies** | ✅ Uses standard PyTorch functionality | ❌ Requires additional libraries like ART |

## Key Technical Differences

1. **Library Usage**: 
   - Black-box attack uses the Adversarial Robustness Toolbox (ART) library's `FastGradientMethod` implementation
   - White-box attack implements FGSM directly using PyTorch's autograd functionality

2. **Question Parameter**:
   - White-box attack requires a "question" parameter because it processes both image and text through the VLM
   - Black-box attack doesn't need a question parameter as it only processes the image through a classification model

3. **Memory Requirements**:
   - White-box attack is significantly more memory-intensive, often requiring 6-7GB of GPU memory
   - For large models, white-box attacks may require CPU fallback, which is much slower but more memory-efficient

4. **Context Sensitivity**:
   - White-box attacks can generate different adversarial examples based on the question context
   - Black-box attacks generate the same adversarial example regardless of how the VLM will be queried

5. **Implementation Approach**:
   - Black-box: "Generate once, attack anywhere" approach
   - White-box: "Tailored to specific model and context" approach

## Practical Implications

The choice between white-box and black-box attacks depends on the level of access to the target model:

- **With full model access:** White-box attacks are preferred for their effectiveness and precision
- **With API-only access:** Black-box attacks are the only feasible option
- **For research purposes:** Both approaches provide valuable insights into model robustness
- **With limited hardware:** Black-box attacks may be the only feasible option on modest hardware

In real-world scenarios, black-box attacks are often more realistic threats since attackers rarely have full access to model internals, while white-box attacks represent the upper bound of what's theoretically possible with complete model knowledge.

## Evaluation Pipeline Integration

Both attack types integrate into the same evaluation pipeline:
1. Generate adversarial images using either attack method
2. Run inference using `scripts/eval_model.py` on the adversarial images
3. Calculate accuracy metrics using `scripts/eval_vqa.py`

This allows for direct comparison of the effectiveness of both attack types against the same VLM models.
