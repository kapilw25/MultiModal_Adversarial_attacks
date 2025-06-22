## Key Differences

1. Computational Complexity:
   • FGSM: Lowest (single gradient computation)
   • PGD: Moderate (multiple gradient computations, typically 10 iterations)
   • CW: Highest (complex optimization, typically 100+ iterations)

2. Attack Strength:
   • FGSM: Weakest (single-step approach limits effectiveness)
   • PGD: Strong (iterative approach finds better adversarial examples)
   • CW: Strongest (directly optimizes for minimal perturbation with successful attacks)

3. Perturbation Size:
   • FGSM: Largest perturbations (uses full epsilon budget in one step)
   • PGD: Moderate perturbations (constrained by epsilon)
   • CW: Smallest perturbations (explicitly minimizes perturbation size)

4. Visual Quality:
   • FGSM: Most visible changes (large perturbations)
   • PGD: Moderate visual changes
   • CW: Most subtle changes (optimized for minimal visual difference)

5. Optimization Approach:
   • FGSM: Direct gradient-based (non-iterative)
   • PGD: Projected gradient descent (iterative with constraints)
   • CW: Lagrangian optimization (complex optimization problem)

6. Use Cases:
   • FGSM: Quick testing, adversarial training
   • PGD: Standard benchmark for robustness
   • CW: Thorough security evaluation, finding minimal perturbations