## Key Differences

| Aspect | FGSM | PGD | JSMA | CW | L-BFGS | DeepFool |
|--------|------|-----|------|----|----|---------|
| **Full Name** | Fast Gradient Sign Method | Projected Gradient Descent | Jacobian-based Saliency Map Attack | Carlini-Wagner Attack | Limited-memory Broyden–Fletcher–Goldfarb–Shanno | DeepFool |
| **Computational Complexity** | Lowest (single gradient) | Moderate (10+ iterations) | High (Jacobian matrix) | Highest (complex optimization) | High (multiple optimizations) | Moderate (iterative approximation) |
| **Attack Strength** | Weakest | Strong | Strong for targeted | Strongest | Strong | Strong for untargeted |
| **Perturbation Size** | Largest | Moderate | Few pixels, large changes | Smallest | Small | Near-minimal |
| **Distance Metric** | L∞ norm | L∞ norm | L0 norm | L0, L2, L∞ variants | L2 norm | L2 norm (typically) |
| **Visual Quality** | Most visible | Moderate | Localized changes | Most subtle | Subtle | Very subtle |
| **Optimization Approach** | Direct gradient | Projected gradient descent | Saliency mapping | Lagrangian optimization | Box-constrained L-BFGS | Iterative linear approximation |
| **Use Cases** | Quick testing, training | Benchmark standard | Sparse pixel manipulation | Security evaluation | Early research, targeted attacks | Robustness measurement |

## FGSM Attack Definition

The Fast Gradient Sign Method (FGSM) is a simple one-step attack designed for efficiency rather than optimality. It was introduced by Goodfellow et al. in "Explaining and Harnessing Adversarial Examples" (2014).

### Mathematical Formulation

For untargeted attacks (causing any misclassification):
```
x' = x + ε · sign(∇ₓJ(θ, x, y))
```

For targeted attacks (causing misclassification to a specific target class t):
```
x' = x - ε · sign(∇ₓJ(θ, x, t))
```

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

### Key Characteristics

1. **L∞ Norm Optimization**: FGSM is optimized for the L∞ distance metric, which means it constrains the maximum change to any pixel.

2. **Efficiency**: Designed primarily for speed rather than producing minimal perturbations.

3. **Single-Step**: Uses a single step in the direction of the gradient sign, making it fast but less effective than iterative methods.

4. **Full Budget Usage**: Always uses the full perturbation budget (ε) for every pixel that has a non-zero gradient.

## Comparison: FGSM vs. PGD (Iterative Gradient Sign)

PGD (Projected Gradient Descent) is essentially an iterative version of FGSM, introduced by Kurakin et al. as "Iterative Gradient Sign" and further developed by Madry et al.

### Mathematical Formulation of PGD

```
x₀' = x
xₜ₊₁' = Proj_ε(xₜ' + α · sign(∇ₓJ(θ, xₜ', y)))
```

Where:
- Proj_ε is a projection function that ensures the perturbation remains within the ε-ball
- α is the step size (smaller than ε)
- t is the iteration number

### Key Differences Between FGSM and PGD

1. **Iterations**: 
   - FGSM: Single step (faster but less effective)
   - PGD: Multiple steps (slower but more effective)

2. **Exploration of Loss Landscape**:
   - FGSM: Limited exploration (one direction only)
   - PGD: Thorough exploration (can find better local maxima of the loss)

3. **Perturbation Quality**:
   - FGSM: Often creates larger, more visible perturbations
   - PGD: Can create more refined perturbations that better exploit model vulnerabilities

4. **Computational Cost**:
   - FGSM: O(1) gradient computations
   - PGD: O(n) gradient computations, where n is the number of iterations

5. **Effectiveness**:
   - FGSM: Generally less effective, especially against adversarially trained models
   - PGD: Considered a "universal first-order adversary" - if a model is robust to PGD, it's likely robust to other first-order attacks

6. **Use in Adversarial Training**:
   - FGSM: Used in "fast adversarial training" but can lead to "catastrophic overfitting"
   - PGD: Gold standard for adversarial training, but more computationally expensive

## L-BFGS Attack Definition

The Limited-memory Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) attack aims to find an adversarial example x' that is visually similar to the 
original image x (measured by L2 distance) while causing the classifier to predict an 
incorrect label. It formulates this as an optimization problem:

### Original Formulation
```
minimize ‖x - x'‖²₂
such that C(x') = l
x' ∈ [0, 1]ⁿ
```

Where:
• x is the original image
• x' is the adversarial example
• C(x') is the classifier's prediction on x'
• l is the target label (different from the original)
• [0, 1]ⁿ ensures pixel values remain valid

### Practical Implementation
Since the above constrained optimization is difficult to solve directly, it's reformulated as:

```
minimize c · ‖x - x'‖²₂ + loss_F,l(x')
such that x' ∈ [0, 1]ⁿ
```

Where:
• c is a constant that balances the two terms
• loss_F,l is a loss function (typically cross-entropy) that encourages the model to classify
x' as label l

The algorithm performs line search to find the optimal value of c that produces an 
adversarial example with minimal distortion. This involves solving the optimization problem 
repeatedly with different values of c, using bisection search or other one-dimensional 
optimization methods.

## Key Characteristics of L-BFGS Attack

1. **Target-specific**: It's designed to force the model to predict a specific incorrect label.

2. **Optimization-based**: Unlike faster gradient-based methods, it uses the L-BFGS optimization 
algorithm, which is more computationally intensive but can find adversarial examples with 
smaller perturbations.

3. **L2 norm**: It minimizes the L2 (Euclidean) distance between the original and adversarial 
images, which tends to spread small changes across many pixels.

4. **Box-constrained**: The adversarial example must have valid pixel values (typically in [0,1] 
range).

This attack demonstrated that neural networks have intriguing properties where visually 
imperceptible perturbations can cause misclassification, challenging the assumption that 
these models generalize well to slightly modified inputs.

## Jacobian-based Saliency Map Attack (JSMA)

The Jacobian-based Saliency Map Attack (JSMA) is an adversarial attack method introduced by Papernot et al. that is optimized for the L0 distance metric. Unlike attacks that focus on minimizing the overall perturbation magnitude (like L2 or L∞ norms), JSMA aims to modify the fewest possible pixels while successfully causing misclassification.

### Core Concept

JSMA is a greedy algorithm that iteratively selects and modifies pixels that have the highest impact on the classification outcome. It uses the gradient information to construct a "saliency map" that quantifies how influential each pixel is for achieving the target classification.

### Mathematical Formulation

The attack works by computing two key components for pairs of pixels (p,q):

1. **Alpha component (αpq)**: Measures how much changing pixels p and q will increase the likelihood of the target class t:
   ```
   αpq = ∑(i∈{p,q}) ∂Z(x)t/∂xi
   ```

2. **Beta component (βpq)**: Measures how much changing pixels p and q will decrease the likelihood of all other classes:
   ```
   βpq = (∑(i∈{p,q}) ∑(j) ∂Z(x)j/∂xi) - αpq
   ```

The algorithm then selects the pixel pair (p*,q*) that maximizes the product of these components, subject to constraints:
```
(p*,q*) = arg max(p,q) (-αpq · βpq) · (αpq > 0) · (βpq < 0)
```

Where:
- αpq > 0 ensures the target class becomes more likely
- βpq < 0 ensures other classes become less likely
- Maximizing -αpq · βpq finds the most effective pixel pair

### Variants

The paper describes two variants of JSMA:

1. **JSMA-Z**: Uses the logits (Z) - the output of the second-to-last layer before softmax - to calculate gradients. This is the primary version of the attack.

2. **JSMA-F**: Uses the final softmax outputs (F) instead of logits to calculate gradients. This variant was specifically used when attacking defensively distilled networks.

### Implementation Details

- The attack modifies one or two pixels at a time, iteratively increasing the likelihood of the target class.
- It continues until either:
  - The model classifies the image as the target class (success)
  - The number of modified pixels exceeds a predefined threshold (failure)

- For RGB images, the attack considers each color channel independently, so modifying all three channels of a single pixel counts as 3 in the L0 distance.

### Effectiveness and Limitations

- **Effectiveness**: JSMA can be highly effective at creating adversarial examples with minimal pixel changes.
- **Computational Cost**: The attack is computationally expensive as it requires calculating the Jacobian matrix and evaluating many pixel pairs.
- **Visibility**: While few pixels are changed, the magnitude of changes to those pixels can be large and potentially visible.
- **L0 Metric**: The L0 distance metric (counting changed pixels) may not always align with human perception of image similarity.

JSMA represents an important approach in adversarial attacks by demonstrating that neural networks can be fooled by changing just a few carefully selected pixels, highlighting a different dimension of vulnerability compared to attacks that spread small changes across many pixels.

## References

```
@misc{carlini2017evaluatingrobustnessneuralnetworks,
      title={Towards Evaluating the Robustness of Neural Networks}, 
      author={Nicholas Carlini and David Wagner},
      year={2017},
      eprint={1608.04644},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/1608.04644}, 
}

@article{papernot2016limitations,
      title={The Limitations of Deep Learning in Adversarial Settings}, 
      author={Nicolas Papernot and Patrick McDaniel and Somesh Jha and Matt Fredrikson and Z. Berkay Celik and Ananthram Swami},
      year={2016},
      journal={IEEE European Symposium on Security and Privacy (EuroS&P)},
      url={https://arxiv.org/abs/1511.07528},
}
```
## DeepFool Attack Definition

DeepFool is an adversarial attack method designed to find the minimal perturbation needed to cause a misclassification in deep neural networks. It was introduced by Moosavi-Dezfooli et al. in "DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks" (2016).

### Core Concept

DeepFool works by iteratively finding the closest decision boundary to the input sample and then pushing the sample across that boundary. It approximates the classifier as a linear model at each iteration and moves toward the closest decision boundary.

### Mathematical Formulation

For a binary classifier with decision function f, DeepFool finds the minimal perturbation r that satisfies:
```
sign(f(x + r)) ≠ sign(f(x))
```

For multi-class classifiers with k classes, at each iteration i, DeepFool:

1. Finds the closest hyperplane (decision boundary) by solving:
```
argmin_{k≠k₀} |f_k(x_i) - f_{k₀}(x_i)| / ||∇f_k(x_i) - ∇f_{k₀}(x_i)||₂
```
where k₀ is the current predicted class.

2. Computes the minimal perturbation to reach this hyperplane:
```
r_i = - (f_k(x_i) - f_{k₀}(x_i)) / ||∇f_k(x_i) - ∇f_{k₀}(x_i)||₂² · (∇f_k(x_i) - ∇f_{k₀}(x_i))
```

3. Updates the input: x_{i+1} = x_i + r_i

4. Repeats until the classification changes

### Algorithm Steps

1. **Initialization**: Start with the original image x₀ = x, and set i = 0
2. **Linearization**: At each iteration i, approximate the classifier locally as a linear function
3. **Find Minimal Perturbation**: Compute the minimal perturbation r_i to reach the closest decision boundary
4. **Update**: Set x_{i+1} = x_i + r_i
5. **Check**: If the classification changes, stop; otherwise, increment i and repeat
6. **Overshoot**: Apply a small overshoot factor (typically 1.02) to ensure crossing the boundary:
   ```
   x' = x + (1 + η) · ∑r_i
   ```
   where η is the overshoot parameter (e.g., 0.02)

### Key Properties

1. **Minimal Perturbation**: DeepFool aims to find the smallest perturbation needed for misclassification, measured by some norm (typically L2)
2. **Iterative Approach**: Unlike one-step methods like FGSM, DeepFool iteratively refines the perturbation
3. **Untargeted Attack**: DeepFool pushes samples across the nearest decision boundary without targeting a specific class
4. **Efficiency**: More efficient than optimization-based methods like C&W while still finding near-minimal perturbations
5. **Adaptability**: Can be adapted to different norms (L2, L∞, L1) by changing the distance metric

### Effectiveness and Limitations

- **Effectiveness**: DeepFool typically produces perturbations that are 2-10 times smaller than those from FGSM
- **Visual Quality**: The perturbations are often imperceptible to humans
- **Computational Cost**: Moderate - requires multiple iterations but converges quickly (typically 3-10 iterations)
- **Robustness Measurement**: Provides a good proxy for measuring model robustness
- **Limitation**: As an untargeted attack, it doesn't allow specifying a target class

DeepFool has become an important benchmark for evaluating neural network robustness and has contributed to the development of more effective adversarial training methods. Its ability to find near-minimal perturbations makes it particularly useful for understanding the fundamental vulnerabilities of deep learning models.
## References

```
@misc{carlini2017evaluatingrobustnessneuralnetworks,
      title={Towards Evaluating the Robustness of Neural Networks}, 
      author={Nicholas Carlini and David Wagner},
      year={2017},
      eprint={1608.04644},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/1608.04644}, 
}

@article{papernot2016limitations,
      title={The Limitations of Deep Learning in Adversarial Settings}, 
      author={Nicolas Papernot and Patrick McDaniel and Somesh Jha and Matt Fredrikson and Z. Berkay Celik and Ananthram Swami},
      year={2016},
      journal={IEEE European Symposium on Security and Privacy (EuroS&P)},
      url={https://arxiv.org/abs/1511.07528},
}

@article{moosavi2016deepfool,
      title={DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks}, 
      author={Seyed-Mohsen Moosavi-Dezfooli and Alhussein Fawzi and Pascal Frossard},
      year={2016},
      journal={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      url={https://arxiv.org/abs/1511.04599},
}
```
