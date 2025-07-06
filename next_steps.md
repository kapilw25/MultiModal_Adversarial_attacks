# Enhancing Evaluation and Attack Strategies

## Proposed Changes to eval_vqa.py

1. **Add numerical tracking variables** (line ~95)
   ```python
   numerical_questions = 0
   numerical_error_sum = 0.0
   numerical_relative_error_sum = 0.0
   ```

2. **Calculate numerical errors** (line ~165-180)
   ```python
   if is_number(pr) and is_number(gt):
       numerical_questions += 1
       pr_num = str_to_num(pr)
       gt_num = str_to_num(gt)
       abs_error = abs(pr_num - gt_num)
       numerical_error_sum += abs_error
       if gt_num != 0:
           rel_error = abs_error / abs(gt_num)
           numerical_relative_error_sum += rel_error
   ```

3. **Return numerical metrics** (line ~220-227)
   ```python
   numerical_accuracy = {}
   if numerical_questions > 0:
       numerical_accuracy = {
           'mean_absolute_error': numerical_error_sum / numerical_questions,
           'mean_relative_error': numerical_relative_error_sum / numerical_questions
       }
       print(f'{file_type} Mean Absolute Error: {numerical_accuracy["mean_absolute_error"]:.2f}')
       print(f'{file_type} Mean Relative Error: {numerical_accuracy["mean_relative_error"]*100:.2f}%')
   
   return ok_results, bad_results, accuracy if len(eval_file) - summary_cnt > 0 else 0, file_type, numerical_accuracy if numerical_questions > 0 else {}
   ```

4. **Update evaluate_all_files function** (line ~280-290)
   ```python
   numerical_metrics = {}
   for path in file_paths:
       file_name = os.path.basename(path)
       _, _, accuracy, file_type, num_metrics = evaluator(path)
       results[file_name] = accuracy
       file_types[file_name] = file_type
       numerical_metrics[file_name] = num_metrics
   ```

5. **Add metrics comparison** (line ~320-330)
   ```python
   if numerical_metrics.get(orig_file) and numerical_metrics.get(attack_file):
       orig_mae = numerical_metrics[orig_file].get('mean_absolute_error', 0)
       attack_mae = numerical_metrics[attack_file].get('mean_absolute_error', 0)
       mae_diff = attack_mae - orig_mae
       
       orig_mre = numerical_metrics[orig_file].get('mean_relative_error', 0) * 100
       attack_mre = numerical_metrics[attack_file].get('mean_relative_error', 0) * 100
       mre_diff = attack_mre - orig_mre
       
       print(f"  Numerical Mean Absolute Error: {orig_mae:.2f} → {attack_mae:.2f} ({'+' if mae_diff > 0 else ''}{mae_diff:.2f})")
       print(f"  Numerical Mean Relative Error: {orig_mre:.2f}% → {attack_mre:.2f}% ({'+' if mre_diff > 0 else ''}{mre_diff:.2f}%)")
   ```

## Balancing Imperceptibility and Effectiveness

### Strategies

1. **Adaptive Perturbation Magnitude**
   - Incrementally increase perturbation until finding performance degradation threshold
   - Stop when SSIM drops below threshold or target degradation is reached

2. **Semantic Region Targeting**
   - Focus stronger perturbations on critical regions (text, data points, axes)
   - Apply minimal changes elsewhere
   - Use OCR to target text elements with precision

3. **Perceptual Optimization**
   - Incorporate LPIPS alongside SSIM
   - Exploit human visual perception limitations
   - Use color space transformations less perceptible to humans

4. **Model-Specific Customization**
   - Develop separate parameters for different models
   - Identify transferable perturbations between models
   - Create hybrid attacks effective against multiple architectures

5. **Feature-Level Manipulation**
   - Modify chart elements rather than pixels
   - Create subtle inconsistencies between visual elements and text
   - Alter semantic relationships while preserving appearance

### Promising Black-box Attacks

1. **HopSkipJump Attack** (Chen et al., 2019)
   - Direction-based sampling minimizing L2 norm
   - Adaptive step sizes for efficient decision boundary finding

2. **Geometric Decision-based Attack (GeoDA)** (Rahmati et al., 2020)
   - Uses geometric information for minimal perturbations
   - Effective with high-dimensional inputs

3. **Square Attack** (Andriushchenko et al., 2020)
   - Random search with square-shaped updates
   - Query-efficient with good imperceptibility

4. **Simple Black-box Adversarial (SimBA)** (Guo et al., 2019)
   - Iteratively perturbs pixels or low-frequency components
   - Naturally creates imperceptible perturbations

### Implementation Plan

1. **Short-term**
   - Implement HopSkipJump and GeoDA with semantic importance maps
   - Evaluate against GPT-4o and Qwen25_VL_3B

2. **Mid-term**
   - Develop hybrid approaches combining semantic targeting with Square Attack/SimBA
   - Create comprehensive evaluation framework

3. **Long-term**
   - Build unified framework selecting optimal attack strategy automatically
   - Develop model-specific attack variants
   - Explore defenses to understand VLM vulnerabilities

### Evaluation Metrics
- SSIM and LPIPS similarity
- Human detection rate
- Model performance degradation
- Attack success rate
- Query efficiency
- Cross-model transferability
