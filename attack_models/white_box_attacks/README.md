# White Box Attacks on Vision-Language Models

This directory contains implementations of white box adversarial attacks on the Qwen2.5-VL-3B model. Unlike black box attacks, these attacks directly access the model's gradients to create more effective adversarial examples.

## Key Differences from Black Box Attacks

1. **Direct Gradient Access**: Uses the actual VLM's gradients rather than a substitute model
2. **No Transfer Gap**: Avoids the "transfer gap" present in black box attacks
3. **Higher Memory Requirements**: Requires loading the model in 16-bit (not 4-bit) to compute gradients
4. **Potentially Stronger Attacks**: Typically produces more effective adversarial examples

## Available Attacks

### FGSM (Fast Gradient Sign Method)

A single-step attack that perturbs an image by taking a step in the direction of the gradient sign.

```bash
python v3_fgsm_attack.py --image_path data/test_extracted/chart/image.png --eps 0.03 --question "Describe this chart in detail."
```

Parameters:
- `--image_path`: Path to the input image
- `--eps`: Perturbation magnitude (default: 0.03)
- `--question`: Question to ask the model (default: "Describe this chart in detail.")

## Implementation Details

### Utility Functions (`v0_attack_utils.py`)

- `load_image`: Loads and preprocesses an image
- `load_model`: Loads the Qwen2.5-VL-3B model with gradient access
- `save_image`: Saves the image to the specified path
- `process_vision_info`: Processes vision information for the model
- `get_output_path`: Generates output path for adversarial images
- `print_attack_info`: Prints information about the attack
- `cleanup_model`: Cleans up GPU resources

### Attack Implementation (`v3_fgsm_attack.py`)

The white box FGSM attack:
1. Loads the Qwen2.5-VL-3B model in 16-bit precision
2. Processes the input image and question
3. Computes the loss and gradients with respect to the image
4. Applies the FGSM perturbation: x' = x + ε · sign(∇ₓJ(θ, x, y))
5. Saves the adversarial image

## Memory Requirements

The Qwen2.5-VL-3B model requires approximately 6-7GB of GPU memory when loaded in 16-bit precision for gradient computation. Ensure your GPU has sufficient memory before running these attacks.

## Evaluation

To evaluate the model on white box adversarial examples:

1. Generate adversarial examples using the scripts in this directory
2. Run the evaluation script with the appropriate path:

```bash
python scripts/eval_model.py --engine Qwen25_VL_3B --attack white_box_fgsm --task chart
```

3. Compare results with the original model and black box attacks:

```bash
python scripts/eval_vqa.py --engine Qwen25_VL_3B
```
