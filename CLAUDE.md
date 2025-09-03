# Recommended Paper Structure & Plot Placement

● Section 3: Methodology

  - Sample attack visualizations
  - Attack category definitions (maybe a flow chart)

  Section 4: Baseline Performance

  - heatmap_raw_accuracy.png
  - top_models_radar.png
  - barchart_model_family_robustness.png (clean performance)

  Section 5: Robustness Analysis

  - heatmap_attack_effectiveness.png ⭐ Main result
  - heatmap_family_vs_attack_category.png
  - barchart_attack_category_effectiveness.png
  - linechart_model_degradation.png

  Section 6: Architecture & Scale Analysis

  - barchart_size_category_robustness.png
  - heatmap_size_vs_attack_category.png
  - size_memory_quality_bubble.png

  Section 7: Performance Implications

  - performance_heatmap.png
  - size_vs_memory.png (with discrepancy explanation)
  - metrics_by_size_category.png

  Priority Recommendations

  Immediate needs:
  1. Statistical significance testing plots
  2. Attack transferability analysis
  3. Clean baseline comparison (separate from robustness)

  Nice to have:
  1. Qualitative examples - side-by-side clean vs adversarial image responses
  2. Time-series analysis - if you have multiple evaluation runs
  3. Confidence interval plots - for robustness measurements

  Current Plots Analysis:

  01_baseline_performance.png: ✅ Self-explanatory - Clear baseline comparison showing Qwen25-VL-3B leads (74.1%) with color-coded families.

  02_model_degradation_line.png: ✅ Self-explanatory - Shows how 5 top models degrade under different attacks with clear trend lines.

  03_attack_effectiveness_heatmap.png: ✅ Self-explanatory - Comprehensive attack vs model heatmap showing vulnerability patterns.

  04_raw_accuracy_heatmap.png: ✅ Self-explanatory - Raw accuracy values across all attack-model combinations.

  05_model_family_robustness.png: ✅ Self-explanatory - Family-wise vulnerability comparison (InternVL most vulnerable at -8.9%).

  06_size_category_robustness.png: ✅ Self-explanatory - Shows 3-4B models are most vulnerable (-9.0%).

  Issues to Address:

  07_attack_category_effectiveness.png: ❌ Your observation is correct - This is redundant with Plot 10 and adds no new insights. Plot 10 provides the same information with better detail and sorting.

  08_architecture_vulnerability_analysis.png: ❌ Too cluttered - 3 different visualizations (radar + bubble + heatmap) make it difficult to read and interpret.

  10_attack_transferability_analysis.png: ✅ Much improved - Now sorted by effectiveness with clear Transfer vs Black-Box color coding.

  11_size_memory_quality_bubble.png: ❌ Top left still clustered - SmolVLM models are overlapping despite improvements.

  12_size_vs_memory.png: ❌ Missing trendline values - No numerical GB values on the trendline for reference.

  13_top_models_radar.png: ✅ Excellent - 8-dimensional comparison of 6 models, very informative.

