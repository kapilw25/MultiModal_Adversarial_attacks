## Standard Practices in AI/ML Research

### Adjusting SSIM Threshold:
   • Lowering the SSIM threshold from 0.85 is a valid approach in adversarial ML research
   • Standard practice: Yes, researchers often use different perceptual constraints based on model sensitivity
   • Recommendation: Consider a model-specific approach where you use lower thresholds (e.g., 0.75) for models with low
base accuracy
   • Trade-off: Lower SSIM means more visible perturbations, which should be acknowledged in your methodology

### Modifying Evaluation Metrics:
   • Standard practice: Yes, using alternative metrics beyond accuracy is common when dealing with low-performance 
models
   • Better approaches include:
     • Confidence-weighted accuracy metrics
     • Area Under ROC Curve (AUC) instead of raw accuracy
     • Measuring changes in output distribution (KL divergence)
     • Analyzing qualitative changes in model responses

### Additional Standard Approaches:
   • Model-specific attack optimization: Tune attack hyperparameters per model
   • Targeted attacks: Focus on specific capabilities rather than general performance
   • Ensemble evaluation: Combine multiple metrics to get a more holistic view
   • Relative performance degradation: Use percentage change rather than absolute change

## Analysis of degradation in Performance of VLMs


### Gemma3_vl_4b Robustness:
   • Shows unusual positive degradation (+11.76%) under L-BFGS and Spatial attacks
   • Many attacks show no degradation (0%)
   • Original accuracy (41.18%) remains stable across most attacks

### Paligemma_vl_3b Performance:
   • Highly vulnerable to Spatial attack (-47.06% degradation)
   • Moderate vulnerability to CW-L2 (-5.88%)
   • Original accuracy (64.71%) is relatively high

### DeepSeek1_vl_1pt3b Performance:
   • Original accuracy is relatively low at 23.53%
   • Shows minimal degradation across most attacks (-5.88% at worst)
   • Some attacks actually show positive degradation (+5.88%)
   • Most attacks show no degradation (0%)

### DeepSeek1_vl_7b Performance:
   • Extremely low original accuracy at just 5.88%
   • Shows no degradation (0%) across all attacks
   • Consistently maintains the same low performance regardless of attack type

### SmolVLM2 Family:
• **SmolVLM2_pt25b**: Shows 0% accuracy across all attacks, including baseline
• **SmolVLM2_pt5b and SmolVLM2_2pt2b**: Consistent 5.88% accuracy with 0% degradation
• These extremely low and consistent values suggest these models may not be suitable for chart interpretation tasks

### Phi-3.5 Vision:
• **phi3pt5_vision_4b**: Maintains consistent 5.88% accuracy with 0% degradation
• Despite being a larger model (4B parameters), it performs poorly on chart tasks

### Florence2 Models:
• **florence2_pt23b**: Shows 11.76% accuracy with 0% degradation
• **florence2_pt77b**: Shows 5.88% accuracy with 0% degradation
• The smaller model actually performs better, but both show no impact from adversarial attacks

### Other Models:
• **moondream2_2b, glmedge_2b, internvl3_1b, internvl3_2b, internvl25_4b**: All show consistent 5.88% accuracy with 0% 
degradation
• This uniform performance suggests these models may be defaulting to the same answer across different inputs

## Why So Many Zeros?

The prevalence of zeros in the degradation metrics for DeepSeek models could be due to several factors:

1. Low Base Accuracy: With DeepSeek1_vl_7b starting at only 5.88% accuracy, there's little room for further degradation.
The model may be performing at near-random levels already.

2. Evaluation Methodology: The evaluation might be using metrics that don't capture subtle changes in performance when 
accuracy is already very low.

3. Model Architecture Resilience: It's possible that DeepSeek models have architectural features that make them less 
susceptible to certain types of perturbations.