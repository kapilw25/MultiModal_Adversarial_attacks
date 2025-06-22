# 6_Evaluating_and_Understanding_the_Robustness_of_Adversarial_Logit_Pairing

## Document Information

- **Source**: attack_models/research_papers/6_Evaluating_and_Understanding_the_Robustness_of_Adversarial_Logit_Pairing.pdf
- **Pages**: 6
- **Tables**: 7



## Page 1

Evaluating and Understanding the Robustness of
Adversarial Logit Pairing
Logan Engstrom∗
Andrew Ilyas∗
Anish Athalye∗
Massachusetts Institute of Technology
{engstrom,ailyas,aathalye}@mit.edu
Abstract
We evaluate the robustness of Adversarial Logit Pairing, a recently proposed de-
fense against adversarial examples. We ﬁnd that a network trained with Adversarial
Logit Pairing achieves 0.6% correct classiﬁcation rate under targeted adversarial
attack, the threat model in which the defense is considered. We provide a brief
overview of the defense and the threat models/claims considered, as well as a
discussion of the methodology and results of our attack. Our results offer insights
into the reasons underlying the vulnerability of ALP to adversarial attack, and are
of general interest in evaluating and understanding adversarial defenses.
1
Contributions
For summary, the contributions of this note are as follows:
1. Robustness: Under the white-box targeted attack threat model speciﬁed in Kannan et al.,
we upper bound the correct classiﬁcation rate of the defense to 0.6% (Table 1). We also
perform targeted and untargeted attacks and show that the attacker can reach success rates
of 98.6% and 99.9% respectively (Figures 1, 2).
2. Formulation: We analyze the ALP loss function and contrast it to that of Madry et al.,
pointing out several differences from the robust optimization objective (Section 4.1).
3. Loss landscape: We analyze the loss landscape induced by ALP by visualizing loss land-
scapes and adversarial attack trajectories (Section 4.2).
Furthermore, we suggest the experiments conducted in the analysis of ALP as another evaluation
method for adversarial defenses.
2
Introduction
Neural networks and machine learning models in general are known to be susceptible to adversarial
examples, or slightly perturbed inputs that induce speciﬁc and unintended behaviour [Szegedy et al.,
2013, Biggio et al., 2013]. Defenses against these adversarial attacks are of great signiﬁcance and
value. Unfortunately, many proposed defenses have had their claims invalidated by new attacks
within their corresponding threat models [Carlini and Wagner, 2016, He et al., 2017, Carlini and
Wagner, 2017a,b, Athalye et al., Uesato et al., 2018, Athalye and Carlini]. A notable defense has
been that of Madry et al., which proposes a “robust optimization”-based view of defense against
∗Equal contribution
arXiv:1807.10272v2  [stat.ML]  23 Nov 2018


**Table 1 from page 1**

| 0                                                                                   | 1                | 2                                                      |
|:------------------------------------------------------------------------------------|:-----------------|:-------------------------------------------------------|
|                                                                                     |                  | Abstract                                               |
| We evaluate the robustness of Adversarial Logit Pairing, a recently proposed de-    |                  |                                                        |
| fense against adversarial examples. We ﬁnd that a network trained with Adversarial  |                  |                                                        |
| Logit Pairing achieves 0.6% correct classiﬁcation rate under targeted adversarial   |                  |                                                        |
| attack,                                                                             | the threat model | in which the defense is considered. We provide a brief |
| overview of the defense and the threat models/claims considered, as well as a       |                  |                                                        |
| discussion of the methodology and results of our attack. Our results offer insights |                  |                                                        |
| into the reasons underlying the vulnerability of ALP to adversarial attack, and are |                  |                                                        |
| of general interest in evaluating and understanding adversarial defenses.           |                  |                                                        |



## Page 2

adversarial examples, in which the defender tries to ﬁnd parameters θ∗minimizing the following
objective:
min
θ
E(x,y)∼D

max
δ∈S L(θ, x + δ, y)

.
(1)
Here, L is a prespeciﬁed loss function, D is the labeled data distribution, and S is the set of admissible
adversarial perturbations (speciﬁed by a threat model). In practice, the defense is implemented through
adversarial training, where adversarial examples are generated during the training process and used
as inputs. The resulting classiﬁers have been empirically evaluated to offer increased robustness to
adversarial examples on the CIFAR-10 and MNIST datasets under small ℓ∞perturbations.
In Kannan et al., the authors claim that the defense of Madry et al. is ineffective when scaled to an
ImageNet [Deng et al., 2009] classiﬁer, and propose a new defense: Adversarial Logit Pairing (ALP).
In the ALP defense, a classiﬁer is trained with a training objective that enforces similarity between
the model’s logit activations on unperturbed and adversarial versions of the same image. The loss
additionally has a term meant to maintain accuracy on the original training set.
min
θ
E(x,y)∼D [L(θ, x, y) + λD (f(θ, x), f(θ, x + δ∗))]
where δ∗= arg max
δ∈S
L(θ, x + δ, y),
Here, D is a distance function, f is a function mapping parameters and inputs to logits (via the
given network), λ is a hyperparameter, and the rest of the notation is as in (1). This objective is
intended to promote “better internal representations of the data” [Kannan et al.] by providing an
extra regularization term. In the following sections, we show that ALP can be circumvented using
Projected Gradient Descent (PGD) based attacks.
2.1
Setup details
We analyze Adversarial Logit Pairing as implemented by the authors 2. We use the “models pre-
trained on ImageNet” from the code release to evaluate the claims of Kannan et al.. Via private
correspondence, the authors acknowledged our result but stated that the results in Kannan et al. were
generated with different, unreleased models not included in the ofﬁcial code release.
Our evaluation code is publicly available. 3.
3
Threat model and claims
Table 1: The claimed robustness of Adversarial Logit Pairing against targeted attacks on ImageNet, from
Kannan et al., compared to the lower bound on attacker success rate from this work. Attacker success rate in this
case represents the percentage of times an attacker successfully induces the adversarial target class, whereas
accuracy measures the percentage of times the classiﬁer outputs the correct class.
Source
Kannan et al.
this work
this work
Defense (ϵ = 16/255)
Claimed Accuracy
Defense Accuracy4
Attacker Success
Madry et al.
1.5%
–
–
Kannan et al.
27.9%5
0.6%
98.6%
ALP is claimed secure under a variety of white-box and black-box threat models; in this work, we
consider the white-box threat model, where an attacker has full access to the weights and parameters
2https://github.com/tensorflow/models/tree/master/research/adversarial_logit_pairing
3https://github.com/labsix/adversarial-logit-pairing-analysis
4We calculate this as in Kannan et al., i.e. correct classiﬁcation rate under targeted adversarial attack.
5As noted in §2.1, via private correspondence, the authors state that unreleased models were used to generate
the results in Kannan et al.. The authors are currently investigating these models; for the sake of comparison, we
give the claim from Kannan et al. here.
2


**Table 2 from page 2**

| 0                                                                                                          |
|:-----------------------------------------------------------------------------------------------------------|
| adversarial examples,                                                                                      |
| in which the defender tries to ﬁnd parameters θ∗ minimizing the following                                  |
| objective:                                                                                                 |
| (cid:20)                                                                                                   |
| (cid:21)                                                                                                   |
| min                                                                                                        |
| max                                                                                                        |
| L(θ, x + δ, y)                                                                                             |
| .                                                                                                          |
| (1)                                                                                                        |
| E(x,y)∼D                                                                                                   |
| θ                                                                                                          |
| δ∈S                                                                                                        |
| Here, L is a prespeciﬁed loss function, D is the labeled data distribution, and S is the set of admissible |
| adversarial perturbations (speciﬁed by a threat model). In practice, the defense is implemented through    |
| adversarial training, where adversarial examples are generated during the training process and used        |
| as inputs. The resulting classiﬁers have been empirically evaluated to offer increased robustness to       |
| adversarial examples on the CIFAR-10 and MNIST datasets under small (cid:96)∞ perturbations.               |
| In Kannan et al., the authors claim that the defense of Madry et al.                                       |
| is ineffective when scaled to an                                                                           |
| ImageNet [Deng et al., 2009] classiﬁer, and propose a new defense: Adversarial Logit Pairing (ALP).        |
| In the ALP defense, a classiﬁer is trained with a training objective that enforces similarity between      |
| the model’s logit activations on unperturbed and adversarial versions of the same image. The loss          |
| additionally has a term meant to maintain accuracy on the original training set.                           |



## Page 3

0
2
4
6
8
10
12
14
16
0
20
40
60
80
100
considered threat model
ϵ
Attack success rate (%)
Attack success rate (lower is better)
ALP
Baseline
Figure 1: Comparison of ALP-trained model
with baseline model under targeted adver-
sarial perturbations (with random labels)
bounded by varying ϵ from 0 to 16/255. Our
attack reaches 98.6% success rate (and 0.6%
correct classiﬁcation rate) at ϵ = 16/255.
0
2
4
6
8
10
12
14
16
0
20
40
60
80
100
considered threat model
ϵ
Accuracy (%)
Accuracy (higher is better)
ALP
Baseline
Figure 2: Comparison of ALP-trained model
with baseline model under untargeted ad-
versarial perturbations bounded by varying
ϵ from 0 to 16/255. The ALP-trained model
achieves 0.1% accuracy at ϵ = 16/255.
of the model being attacked. Speciﬁcally, we consider an Residual Network ALP-trained on the
ImageNet dataset, where ALP is claimed to achieve state-of-the-art accuracies in this setting under an
ℓ∞perturbation bound of 16/255, as shown in Table 1. The defense is originally evaluated against
targeted adversarial attacks, and thus Table 1 refers to the attacker success rate on targeted adversarial
attacks. For completeness, we also perform a brief analysis on untargeted attacks to show lack of
robustness (Figure 2), but do not consider this in the context of the proposed threat model or claims.
Adversary objective.
When evaluating attacks, an attack that can produce targeted adversarial
examples is stronger than an attack that can only produce untargeted adversarial examples. On the
other hand, a defense that is only robust against targeted adversarial examples (e.g. with random
target classes) is weaker than a defense that is robust against untargeted adversarial examples. The
ALP paper only attempts to show robustness to targeted adversarial examples.
4
Evaluation
4.1
Analyzing the defense objective
Adversarial Logit Pairing is proposed as an augmentation of adversarial training, which itself is
meant to approximate the robust optimization approach outlined in Equation 1. The paper claims
that by adding a “regularizer” to the adversarial training objective, better results on high-dimensional
datasets can be achieved. In this section we outline several conceptual differences between ALP and
the robust optimization perspective offered by Madry et al..
Training on natural vs. adversarial images.
A key part in the formulation of the robust optimiza-
tion objective is that minimization with respect to θ is done over the inputs that have been crafted by
the max player; θ is not minimized with respect to any “natural” x ∼D. In the ALP formulation,
on the other hand, regularization is applied to the loss on clean data L(θ, x, y). This fundamentally
changes the optimization objective from the defense of Madry et al..
Generating targeted adversarial examples.
A notable implementation decision given in Kannan
et al. is to generate targeted adversarial examples during the training process. This again deviates
from the robust optimization-inspired saddle point formulation for adversarial training, as the inner
maximization player no longer maximizes L(θ, x + δ, y), but rather minimizes L(θ, x + δ, yadv)
3


**Table 3 from page 3**

| 0                                                                                                          |
|:-----------------------------------------------------------------------------------------------------------|
| 0                                                                                                          |
| 0                                                                                                          |
| 0                                                                                                          |
| 2                                                                                                          |
| 4                                                                                                          |
| 6                                                                                                          |
| 8                                                                                                          |
| 10                                                                                                         |
| 12                                                                                                         |
| 14                                                                                                         |
| 16                                                                                                         |
| 0                                                                                                          |
| 2                                                                                                          |
| 4                                                                                                          |
| 6                                                                                                          |
| 8                                                                                                          |
| 10                                                                                                         |
| 12                                                                                                         |
| 14                                                                                                         |
| 16                                                                                                         |
| (cid:15)                                                                                                   |
| (cid:15)                                                                                                   |
| Figure 1: Comparison of ALP-trained model                                                                  |
| Figure 2: Comparison of ALP-trained model                                                                  |
| with baseline model under                                                                                  |
| targeted adver-                                                                                            |
| with baseline model under untargeted ad-                                                                   |
| sarial                                                                                                     |
| perturbations                                                                                              |
| (with                                                                                                      |
| random labels)                                                                                             |
| versarial perturbations bounded by varying                                                                 |
| bounded by varying (cid:15) from 0 to 16/255. Our                                                          |
| (cid:15) from 0 to 16/255. The ALP-trained model                                                           |
| attack reaches 98.6% success rate (and 0.6%                                                                |
| achieves 0.1% accuracy at (cid:15) = 16/255.                                                               |
| correct classiﬁcation rate) at (cid:15) = 16/255.                                                          |
| of the model being attacked. Speciﬁcally, we consider an Residual Network ALP-trained on the               |
| ImageNet dataset, where ALP is claimed to achieve state-of-the-art accuracies in this setting under an     |
| (cid:96)∞ perturbation bound of 16/255, as shown in Table 1. The defense is originally evaluated against   |
| targeted adversarial attacks, and thus Table 1 refers to the attacker success rate on targeted adversarial |
| attacks. For completeness, we also perform a brief analysis on untargeted attacks to show lack of          |
| robustness (Figure 2), but do not consider this in the context of the proposed threat model or claims.     |
| Adversary objective.                                                                                       |
| When evaluating attacks, an attack that can produce targeted adversarial                                   |
| examples is stronger than an attack that can only produce untargeted adversarial examples. On the          |
| other hand, a defense that                                                                                 |
| is only robust against targeted adversarial examples (e.g. with random                                     |
| target classes) is weaker than a defense that is robust against untargeted adversarial examples. The       |
| ALP paper only attempts to show robustness to targeted adversarial examples.                               |



## Page 4

for another class yadv. Note that although Athalye et al. recommends that attacks on ImageNet
classiﬁers be evaluated in the targeted threat model (which is noted in [Kannan et al.] in justifying this
implementation choice), this recommendation does not extend to adversarial training or empirically
showing that a defense is secure (a defense that is only robust to targeted attacks is weaker than one
robust to untargeted attacks).
4.2
Analyzing empirical robustness
Empirical evaluations give upper bounds for the robustness of a defense on test data. Evaluations
done with weak attacks can be seen as giving loose bounds, while evaluations done with stronger
attacks give tighter bounds of true adversarial risk [Uesato et al., 2018]. We ﬁnd that the robustness
of ALP as a defense to adversarial examples is signiﬁcantly lower than claimed in Kannan et al..
Attack procedure.
We originally used the evaluation code provided by the ALP authors and found
that setting the number of steps in the PGD attack to 100 from the default of 20 signiﬁcantly degrades
accuracy. For ease of use we reimplemented a standard PGD attack, which we ran for up to 1000
steps or until convergence. We evaluate both untargeted attacks and targeted attacks with random
targets, measuring model accuracy on the former and adversary success rate (percentage of data
points classiﬁed as the target class) for the latter.
Empirical robustness.
We establish tighter upper bounds on adversarial robustness for both the
ALP trained classiﬁer and the baseline (naturally trained) classiﬁer with our attack. Our results, with a
full curve of ϵ (allowed perturbation) vs attack success rate, are summarized in Figure 1. In the threat
model with ϵ = 16 our attack achieves a 98.6% success rate and reduces the accuracy (percentage of
correctly classiﬁed examples perturbed by the targeted attack) of the classiﬁer to 0.6%.
Figure 2 shows that untargeted attacks gives similar results: the ALP-trained model achieves 0.1%
accuracy at ϵ = 16/255.
Loss landscapes.
We plot loss landscapes around test data points in Figure 3. We vary the input
along a linear space deﬁned by the sign of the gradient and a random Rademacher vector, where
the x and y axes represent the magnitude of the perturbation added in each direction and the z axis
represents the loss. The plots provide evidence that ALP sometimes induces a “bumpier,” depressed
loss landscape tightly around the input points.
Attack convergence.
As suggested by analysis of the loss surface, the optimization landscape of
the ALP-trained network is less amenable to gradient descent. Examining, for a single data point,
the loss over steps of gradient descent in targeted (Figure 4) and untargeted (Figure 5) attacks, we
observe that the attack on the ALP-trained network takes more steps of gradient descent.
This was generally true over all data points. The attack on the ALP-trained network required more
steps of gradient descent to converge, but robustness had not increased (e.g. at ϵ = 16/255, both
networks have roughly 0% accuracy).
5
Conclusion
In this work, we perform an evaluation of the robustness of the Adversarial Logit Pairing defense
(ALP) as proposed in Kannan et al., and show that it is not robust under the considered threat model.
We then study the formulation, implementation, and loss landscape of ALP. The evaluation methods
we use are general and may help in enhancing evaluation standards for adversarial defenses.
Acknowledgements
We thank Harini Kannan, Alexey Kurakin, and Ian Goodfellow for releasing open-source code and
pre-trained models for Adversarial Logit Pairing.
4


**Table 4 from page 4**

| 0                                                                                                               |
|:----------------------------------------------------------------------------------------------------------------|
| recommends that attacks on ImageNet                                                                             |
| for another class yadv. Note that although Athalye et al.                                                       |
| classiﬁers be evaluated in the targeted threat model (which is noted in [Kannan et al.] in justifying this      |
| implementation choice), this recommendation does not extend to adversarial training or empirically              |
| showing that a defense is secure (a defense that is only robust to targeted attacks is weaker than one          |
| robust to untargeted attacks).                                                                                  |
| 4.2                                                                                                             |
| Analyzing empirical robustness                                                                                  |
| Empirical evaluations give upper bounds for the robustness of a defense on test data. Evaluations               |
| done with weak attacks can be seen as giving loose bounds, while evaluations done with stronger                 |
| attacks give tighter bounds of true adversarial risk [Uesato et al., 2018]. We ﬁnd that the robustness          |
| of ALP as a defense to adversarial examples is signiﬁcantly lower than claimed in Kannan et al..                |
| Attack procedure.                                                                                               |
| We originally used the evaluation code provided by the ALP authors and found                                    |
| that setting the number of steps in the PGD attack to 100 from the default of 20 signiﬁcantly degrades          |
| accuracy. For ease of use we reimplemented a standard PGD attack, which we ran for up to 1000                   |
| steps or until convergence. We evaluate both untargeted attacks and targeted attacks with random                |
| targets, measuring model accuracy on the former and adversary success rate (percentage of data                  |
| points classiﬁed as the target class) for the latter.                                                           |
| Empirical robustness.                                                                                           |
| We establish tighter upper bounds on adversarial robustness for both the                                        |
| ALP trained classiﬁer and the baseline (naturally trained) classiﬁer with our attack. Our results, with a       |
| full curve of (cid:15) (allowed perturbation) vs attack success rate, are summarized in Figure 1. In the threat |
| model with (cid:15) = 16 our attack achieves a 98.6% success rate and reduces the accuracy (percentage of       |
| correctly classiﬁed examples perturbed by the targeted attack) of the classiﬁer to 0.6%.                        |
| Figure 2 shows that untargeted attacks gives similar results:                                                   |
| the ALP-trained model achieves 0.1%                                                                             |
| accuracy at (cid:15) = 16/255.                                                                                  |
| Loss landscapes.                                                                                                |
| We plot loss landscapes around test data points in Figure 3. We vary the input                                  |
| along a linear space deﬁned by the sign of the gradient and a random Rademacher vector, where                   |
| the x and y axes represent the magnitude of the perturbation added in each direction and the z axis             |
| represents the loss. The plots provide evidence that ALP sometimes induces a “bumpier,” depressed               |
| loss landscape tightly around the input points.                                                                 |
| Attack convergence.                                                                                             |
| As suggested by analysis of the loss surface, the optimization landscape of                                     |
| the ALP-trained network is less amenable to gradient descent. Examining, for a single data point,               |
| the loss over steps of gradient descent in targeted (Figure 4) and untargeted (Figure 5) attacks, we            |
| observe that the attack on the ALP-trained network takes more steps of gradient descent.                        |
| This was generally true over all data points. The attack on the ALP-trained network required more               |
| steps of gradient descent to converge, but robustness had not increased (e.g. at (cid:15) = 16/255, both        |
| networks have roughly 0% accuracy).                                                                             |



## Page 5

0.05
0.00
0.05
0.05 0.00
0.05
0.08
1.69
3.29
4.90
6.51
Baseline
ImageNet #13742
0.05
0.00
0.05
0.05 0.00
0.05
0.00
2.43
4.85
7.28
9.70
ImageNet #16145
0.05
0.00
0.05
0.05 0.00
0.05
0.93
2.73
4.52
6.31
8.11
ImageNet #24230
0.05
0.00
0.05
0.05 0.00
0.05
0.35
1.02
1.68
2.35
3.02
ALP
ImageNet #13742
0.05
0.00
0.05
0.05 0.00
0.05
0.12
1.69
3.25
4.81
6.38
ImageNet #16145
0.05
0.00
0.05
0.05 0.00
0.05
3.69
4.34
4.98
5.63
6.27
ImageNet #24230
Figure 3: Comparison of loss landscapes of ALP-trained model and baseline model. Loss plots
are generated by varying the input to the models, starting from an original input image chosen
from the test set. We see that ALP sometimes induces decreased loss near the input locally, and
gives a “bumpier” optimization landscape. The z axis represents the loss. If ˆx is the original input,
then we plot the loss varying along the space determined by two vectors: r1 = sign(∇xf(ˆx)) and
r2 ∼Rademacher(0.5). We thus plot the following function: z = loss(x · r1 + y · r2). The classiﬁer
here takes inputs scaled to [0, 1].
0
50
100
150
200
250
0
2
4
6
8
10
PGD Step
−log P(target class)
ALP
Baseline
Figure 4: Comparison of targeted attack on
ALP-trained model versus baseline model on
a single data point, showing loss over PGD
steps. Vertical lines denote the step at which
the attack succeeded (in causing classiﬁcation
as the target class). The optimization process
requires more gradient descent steps on the
ALP model but still succeeds.
0
10
20
30
40
50
0
10
20
PGD Step
−log P(true class)
ALP
Baseline
Figure 5: Comparison of untargeted attack on
ALP-trained model versus baseline model on
a single data point, showing loss over PGD
steps. Vertical lines denote the step at which
the attack succeeded (in causing misclassiﬁca-
tion). The optimization process requires more
gradient descent steps on the ALP model but
still succeeds.
5


**Table 5 from page 5**

| 0                                                                                                      | 1         | 2    | 3    | 4    | 5                                        | 6    | 7    | 8    | 9         | 10   | 11           |
|:-------------------------------------------------------------------------------------------------------|:----------|:-----|:-----|:-----|:-----------------------------------------|:-----|:-----|:-----|:----------|:-----|:-------------|
| 1.68                                                                                                   |           |      |      | 3.25 |                                          |      |      | 4.98 |           |      |              |
| 1.02                                                                                                   |           |      | 0.05 | 1.69 |                                          |      | 0.05 | 4.34 |           |      | 0.05         |
| 0.35                                                                                                   |           |      | 0.00 | 0.12 |                                          |      | 0.00 | 3.69 |           |      | 0.00         |
|                                                                                                        |           |      | 0.05 |      |                                          |      | 0.05 |      |           |      | 0.05         |
|                                                                                                        | 0.05 0.00 |      |      |      | 0.05 0.00                                |      |      |      | 0.05 0.00 |      |              |
|                                                                                                        |           | 0.05 |      |      |                                          | 0.05 |      |      |           | 0.05 |              |
| Figure 3: Comparison of loss landscapes of ALP-trained model and baseline model. Loss plots            |           |      |      |      |                                          |      |      |      |           |      |              |
| are generated by varying the input                                                                     |           |      |      |      | to the models, starting from an original |      |      |      | input     |      | image chosen |
| from the test set. We see that ALP sometimes induces decreased loss near the input                     |           |      |      |      |                                          |      |      |      |           |      | locally, and |
| gives a “bumpier” optimization landscape. The z axis represents the loss. If ˆx is the original input, |           |      |      |      |                                          |      |      |      |           |      |              |
| then we plot the loss varying along the space determined by two vectors: r1 = sign(∇xf (ˆx)) and       |           |      |      |      |                                          |      |      |      |           |      |              |
| r2 ∼ Rademacher(0.5). We thus plot the following function: z = loss(x · r1 + y · r2). The classiﬁer    |           |      |      |      |                                          |      |      |      |           |      |              |
| here takes inputs scaled to [0, 1].                                                                    |           |      |      |      |                                          |      |      |      |           |      |              |

**Table 6 from page 5**

| 0                                              | 1                                              |
|:-----------------------------------------------|:-----------------------------------------------|
| 0                                              | 0                                              |
| 50                                             | 10                                             |
| 100                                            | 20                                             |
| 150                                            | 30                                             |
| 200                                            | 40                                             |
| 250                                            | 50                                             |
| PGD Step                                       | PGD Step                                       |
| Figure 4: Comparison of targeted attack on     | Figure 5: Comparison of untargeted attack on   |
| ALP-trained model versus baseline model on     | ALP-trained model versus baseline model on     |
| a single data point, showing loss over PGD     | a single data point, showing loss over PGD     |
| steps. Vertical lines denote the step at which | steps. Vertical lines denote the step at which |
| the attack succeeded (in causing classiﬁcation | the attack succeeded (in causing misclassiﬁca- |
| as the target class). The optimization process | tion). The optimization process requires more  |
| requires more gradient descent steps on the    | gradient descent steps on the ALP model but    |
| ALP model but still succeeds.                  | still succeeds.                                |



## Page 6

References
A. Athalye and N. Carlini. On the robustness of the CVPR 2018 white-box adversarial example defenses. arXiv
preprint. URL https://arxiv.org/abs/1804.03286.
A. Athalye, N. Carlini, and D. Wagner. Obfuscated gradients give a false sense of security: Circumventing
defenses to adversarial examples. In Proceedings of the 35th International Conference on Machine Learning,
ICML 2018. URL https://arxiv.org/abs/1802.00420.
B. Biggio, I. Corona, D. Maiorca, B. Nelson, N. Šrndi´c, P. Laskov, G. Giacinto, and F. Roli. Evasion attacks
against machine learning at test time. In Joint European Conference on Machine Learning and Knowledge
Discovery in Databases, pages 387–402. Springer, 2013.
N. Carlini and D. Wagner.
Defensive distillation is not robust to adversarial examples.
arXiv preprint
arXiv:1607.04311, 2016.
N. Carlini and D. Wagner. Adversarial examples are not easily detected: Bypassing ten detection methods.
AISec, 2017a.
N. Carlini and D. Wagner. Magnet and “efﬁcient defenses against adversarial attacks” are not robust to adversarial
examples. arXiv preprint arXiv:1711.08478, 2017b.
J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. Imagenet: A large-scale hierarchical image database.
In CVPR, pages 248–255. IEEE, 2009.
W. He, J. Wei, X. Chen, N. Carlini, and D. Song. Adversarial example defenses: Ensembles of weak defenses
are not strong. arXiv preprint arXiv:1706.04701, 2017.
H. Kannan, A. Kurakin, and I. Goodfellow. Adversarial logit pairing. arXiv preprint. URL https://arxiv.
org/abs/1803.06373.
A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu. Towards deep learning models resistant to
adversarial attacks. In International Conference on Learning Representations. URL https://arxiv.org/
abs/1706.06083.
C. Szegedy, W. Zaremba, I. Sutskever, J. Bruna, D. Erhan, I. Goodfellow, and R. Fergus. Intriguing properties of
neural networks. ICLR, 2013.
J. Uesato, B. O’Donoghue, A. van den Oord, and P. Kohli. Adversarial risk and the dangers of evaluating against
weak attacks. In Proceedings of the 35th International Conference on Machine Learning, ICML 2018, 2018.
URL https://arxiv.org/abs/1802.05666.
6


**Table 7 from page 6**

| 0                                                                                                                  |
|:-------------------------------------------------------------------------------------------------------------------|
| References                                                                                                         |
| A. Athalye and N. Carlini. On the robustness of the CVPR 2018 white-box adversarial example defenses. arXiv        |
| preprint. URL https://arxiv.org/abs/1804.03286.                                                                    |
| A. Athalye, N. Carlini, and D. Wagner. Obfuscated gradients give a false sense of security: Circumventing          |
| defenses to adversarial examples.                                                                                  |
| In Proceedings of the 35th International Conference on Machine Learning,                                           |
| ICML 2018. URL https://arxiv.org/abs/1802.00420.                                                                   |
| B. Biggio, I. Corona, D. Maiorca, B. Nelson, N. Šrndi´c, P. Laskov, G. Giacinto, and F. Roli. Evasion attacks      |
| against machine learning at test time.                                                                             |
| In Joint European Conference on Machine Learning and Knowledge                                                     |
| Discovery in Databases, pages 387–402. Springer, 2013.                                                             |
| arXiv preprint                                                                                                     |
| N. Carlini and D. Wagner.                                                                                          |
| Defensive distillation is not                                                                                      |
| robust                                                                                                             |
| to adversarial examples.                                                                                           |
| arXiv:1607.04311, 2016.                                                                                            |
| N. Carlini and D. Wagner. Adversarial examples are not easily detected: Bypassing ten detection methods.           |
| AISec, 2017a.                                                                                                      |
| N. Carlini and D. Wagner. Magnet and “efﬁcient defenses against adversarial attacks” are not robust to adversarial |
| examples. arXiv preprint arXiv:1711.08478, 2017b.                                                                  |
| J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. Imagenet: A large-scale hierarchical image database. |
| In CVPR, pages 248–255. IEEE, 2009.                                                                                |
| W. He, J. Wei, X. Chen, N. Carlini, and D. Song. Adversarial example defenses: Ensembles of weak defenses          |
| are not strong. arXiv preprint arXiv:1706.04701, 2017.                                                             |
| H. Kannan, A. Kurakin, and I. Goodfellow. Adversarial logit pairing. arXiv preprint. URL https://arxiv.            |
| org/abs/1803.06373.                                                                                                |
| A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu. Towards deep learning models resistant                 |
| to                                                                                                                 |
| adversarial attacks.                                                                                               |
| In International Conference on Learning Representations. URL https://arxiv.org/                                    |
| abs/1706.06083.                                                                                                    |
| C. Szegedy, W. Zaremba, I. Sutskever, J. Bruna, D. Erhan, I. Goodfellow, and R. Fergus.                            |
| Intriguing properties of                                                                                           |
| neural networks.                                                                                                   |
| ICLR, 2013.                                                                                                        |
| J. Uesato, B. O’Donoghue, A. van den Oord, and P. Kohli. Adversarial risk and the dangers of evaluating against    |
| weak attacks.                                                                                                      |
| In Proceedings of the 35th International Conference on Machine Learning, ICML 2018, 2018.                          |
| URL https://arxiv.org/abs/1802.05666.                                                                              |

