# 8_DeepFool_a_simple_and_accurate_method_to_fool_deep_neural_networks

## Document Information

- **Source**: attack_models/research_papers/8_DeepFool_a_simple_and_accurate_method_to_fool_deep_neural_networks.pdf
- **Pages**: 9
- **Tables**: 11


ß
## Page 1

DeepFool: a simple and accurate method to fool deep neural networks
Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Pascal Frossard
´Ecole Polytechnique F´ed´erale de Lausanne
{seyed.moosavi,alhussein.fawzi,pascal.frossard} at epfl.ch
Abstract
State-of-the-art deep neural networks have achieved im-
pressive results on many image classiﬁcation tasks. How-
ever, these same architectures have been shown to be un-
stable to small, well sought, perturbations of the images.
Despite the importance of this phenomenon, no effective
methods have been proposed to accurately compute the ro-
bustness of state-of-the-art deep classiﬁers to such pertur-
bations on large-scale datasets. In this paper, we ﬁll this
gap and propose the DeepFool algorithm to efﬁciently com-
pute perturbations that fool deep networks, and thus reli-
ably quantify the robustness of these classiﬁers. Extensive
experimental results show that our approach outperforms
recent methods in the task of computing adversarial pertur-
bations and making classiﬁers more robust.1
1. Introduction
Deep neural networks are powerful learning models that
achieve state-of-the-art pattern recognition performance in
many research areas such as bioinformatics [1, 16], speech
[12, 6], and computer vision [10, 8].
Though deep net-
works have exhibited very good performance in classiﬁca-
tion tasks, they have recently been shown to be particularly
unstable to adversarial perturbations of the data [18]. In
fact, very small and often imperceptible perturbations of the
data samples are sufﬁcient to fool state-of-the-art classiﬁers
and result in incorrect classiﬁcation. (e.g., Figure 1). For-
mally, for a given classiﬁer, we deﬁne an adversarial per-
turbation as the minimal perturbation r that is sufﬁcient to
change the estimated label ˆk(x):
∆(x; ˆk) := min
r
∥r∥2 subject to ˆk(x + r) ̸= ˆk(x),
(1)
where x is an image and ˆk(x) is the estimated label. We
call ∆(x; ˆk) the robustness of ˆk at point x. The robustness
of classiﬁer ˆk is then deﬁned as
1To encourage reproducible research, the code of DeepFool is made
available at http://github.com/lts4/deepfool
Figure 1:
An example of adversarial perturbations.
First row:
the original image x that is classiﬁed as
ˆk(x)=“whale”. Second row: the image x + r classiﬁed
as ˆk(x + r)=“turtle” and the corresponding perturbation r
computed by DeepFool. Third row: the image classiﬁed
as “turtle” and the corresponding perturbation computed
by the fast gradient sign method [4]. DeepFool leads to a
smaller perturbation.
arXiv:1511.04599v3  [cs.LG]  4 Jul 2016


**Table 1 from page 1**

| 0   |
|:----|
|     |

**Table 2 from page 1**

| 0   |
|:----|
|     |

**Table 3 from page 1**

| 0   |
|:----|
|     |



## Page 2

ρadv(ˆk) = Ex
∆(x; ˆk)
∥x∥2
,
(2)
where Ex is the expectation over the distribution of data.
The study of adversarial perturbations helps us understand
what features are used by a classiﬁer. The existence of such
examples is seemingly in contradiction with the generaliza-
tion ability of the learning algorithms. While deep networks
achieve state-of-the-art performance in image classiﬁcation
tasks, they are not robust at all to small adversarial pertur-
bations and tend to misclassify minimally perturbed data
that looks visually similar to clean samples. Though adver-
sarial attacks are speciﬁc to the classiﬁer, it seems that the
adversarial perturbations are generalizable across different
models [18]. This can actually become a real concern from
a security point of view.
An accurate method for ﬁnding the adversarial perturba-
tions is thus necessary to study and compare the robustness
of different classiﬁers to adversarial perturbations. It might
be the key to a better understanding of the limits of cur-
rent architectures and to design methods to increase robust-
ness. Despite the importance of the vulnerability of state-of-
the-art classiﬁers to adversarial instability, no well-founded
method has been proposed to compute adversarial perturba-
tions and we ﬁll this gap in this paper.
Our main contributions are the following:
• We propose a simple yet accurate method for comput-
ing and comparing the robustness of different classi-
ﬁers to adversarial perturbations.
• We perform an extensive experimental comparison,
and show that 1) our method computes adversarial per-
turbations more reliably and efﬁciently than existing
methods 2) augmenting training data with adversarial
examples signiﬁcantly increases the robustness to ad-
versarial perturbations.
• We show that using imprecise approaches for the com-
putation of adversarial perturbations could lead to dif-
ferent and sometimes misleading conclusions about the
robustness. Hence, our method provides a better un-
derstanding of this intriguing phenomenon and of its
inﬂuence factors.
We now review some of the relevant work.
The phe-
nomenon of adversarial instability was ﬁrst introduced and
studied in [18]. The authors estimated adversarial examples
by solving penalized optimization problems and presented
an analysis showing that the high complexity of neural net-
works might be a reason explaining the presence of adver-
sarial examples. Unfortunately, the optimization method
employed in [18] is time-consuming and therefore does not
scale to large datasets. In [14], the authors showed that con-
volutional networks are not invariant to some sort of trans-
formations based on the experiments done on Pascal3D+
annotations. Recently, Tsai et al. [19] provided a software
to misclassify a given image in a speciﬁed class, without
necessarily ﬁnding the smallest perturbation. Nguyen et al.
[13] generated synthetic unrecognizable images, which are
classiﬁed with high conﬁdence.
The authors of [3] also
studied a related problem of ﬁnding the minimal geomet-
ric transformation that fools image classiﬁers, and provided
quantitative measure of the robustness of classiﬁers to geo-
metric transformations. Closer to our work, the authors of
[4] introduced the “fast gradient sign” method, which com-
putes the adversarial perturbations for a given classiﬁer very
efﬁciently. Despite its efﬁciency, this method provides only
a coarse approximation of the optimal perturbation vectors.
In fact, it performs a unique gradient step, which often leads
to sub-optimal solutions. Then in an attempt to build more
robust classiﬁers to adversarial perturbations, [5] introduced
a smoothness penalty in the training procedure that allows
to boost the robustness of the classiﬁer. Notably, the method
in [18] was applied in order to generate adversarial pertur-
bations. We should ﬁnally mention that the phenomenon of
adversarial instability also led to theoretical work in [2] that
studied the problem of adversarial perturbations on some
families of classiﬁers, and provided upper bounds on the
robustness of these classiﬁers. A deeper understanding of
the phenomenon of adversarial instability for more complex
classiﬁers is however needed; the method proposed in this
work can be seen as a baseline to efﬁciently and accurately
generate adversarial perturbations in order to better under-
stand this phenomenon.
The rest of paper is organized as follows. In Section 2,
we introduce an efﬁcient algorithm to ﬁnd adversarial per-
turbations in a binary classiﬁer. The extension to the mul-
ticlass problem is provided in Section 3. In Section 4, we
propose extensive experiments that conﬁrm the accuracy of
our method and outline its beneﬁts in building more robust
classiﬁers.
2. DeepFool for binary classiﬁers
As a multiclass classiﬁer can be viewed as aggregation of
binary classiﬁers, we ﬁrst propose the algorithm for binary
classiﬁers. That is, we assume here ˆk(x) = sign(f(x)),
where f is an arbitrary scalar-valued image classiﬁcation
function f : Rn →R. We also denote by F ≜{x :
f(x) = 0} the level set at zero of f. We begin by analyzing
the case where f is an afﬁne classiﬁer f(x) = wT x + b,
and then derive the general algorithm, which can be applied
to any differentiable binary classiﬁer f.
In the case where the classiﬁer f is afﬁne, it can easily




## Page 3

F
f(x) < 0
f(x) > 0
r∗(x)
∆(x0; f)
x0
Figure 2: Adversarial examples for a linear binary classiﬁer.
be seen that the robustness of f at point x0, ∆(x0; f)2, is
equal to the distance from x0 to the separating afﬁne hyper-
plane F = {x : wT x + b = 0} (Figure 2). The minimal
perturbation to change the classiﬁer’s decision corresponds
to the orthogonal projection of x0 onto F. It is given by
the closed-form formula:
r∗(x0) := arg min ∥r∥2
(3)
subject to sign (f(x0 + r)) ̸= sign(f(x0))
= −f(x0)
∥w∥2
2
w.
Assuming now that f is a general binary differentiable clas-
siﬁer, we adopt an iterative procedure to estimate the robust-
ness ∆(x0; f). Speciﬁcally, at each iteration, f is linearized
around the current point xi and the minimal perturbation of
the linearized classiﬁer is computed as
arg min
ri
∥ri∥2 subject to f(xi) + ∇f(xi)T ri = 0.
(4)
The perturbation ri at iteration i of the algorithm is com-
puted using the closed form solution in Eq. (3), and the next
iterate xi+1 is updated. The algorithm stops when xi+1
changes sign of the classiﬁer. The DeepFool algorithm for
binary classiﬁers is summarized in Algorithm 1 and a geo-
metric illustration of the method is shown in Figure 3.
In practice, the above algorithm can often converge to a
point on the zero level set F. In order to reach the other side
of the classiﬁcation boundary, the ﬁnal perturbation vector
ˆr is multiplied by a constant 1 + η, with η ≪1. In our
experiments, we have used η = 0.02.
3. DeepFool for multiclass classiﬁers
We now extend the DeepFool method to the multiclass
case. The most common used scheme for multiclass clas-
siﬁers is one-vs-all. Hence, we also propose our method
2From now on, we refer to a classiﬁer either by f or its correspond-
ing discrete mapping ˆk. Therefore, ρadv(ˆk) = ρadv(f) and ∆(x; ˆk) =
∆(x; f).
Algorithm 1 DeepFool for binary classiﬁers
1: input: Image x, classiﬁer f.
2: output: Perturbation ˆr.
3: Initialize x0 ←x, i ←0.
4: while sign(f(xi)) = sign(f(x0)) do
5:
ri ←−
f(xi)
∥∇f(xi)∥2
2 ∇f(xi),
6:
xi+1 ←xi + ri,
7:
i ←i + 1.
8: end while
9: return ˆr = P
i ri.
x0
x1
F
Rn
Figure 3: Illustration of Algorithm 1 for n = 2.
As-
sume x0 ∈Rn. The green plane is the graph of x 7→
f(x0)+∇f(x0)T (x−x0), which is tangent to the classiﬁer
function (wire-framed graph) x 7→f(x). The orange line
indicates where f(x0) + ∇f(x0)T (x −x0) = 0. x1 is ob-
tained from x0 by projecting x0 on the orange hyperplane
of Rn.
based on this classiﬁcation scheme.
In this scheme, the
classiﬁer has c outputs where c is the number of classes.
Therefore, a classiﬁer can be deﬁned as f : Rn →Rc and
the classiﬁcation is done by the following mapping:
ˆk(x) = arg max
k
fk(x),
(5)
where fk(x) is the output of f(x) that corresponds to the
kth class. Similarly to the binary case, we ﬁrst present the
proposed approach for the linear case and then we general-
ize it to other classiﬁers.
3.1. Afﬁne multiclass classiﬁer
Let f(x) be an afﬁne classiﬁer, i.e., f(x) = W⊤x + b
for a given W and b. Since the mapping ˆk is the outcome of
a one-vs-all classiﬁcation scheme, the minimal perturbation
to fool the classiﬁer can be rewritten as follows
arg min
r
∥r∥2
s.t. ∃k : w⊤
k (x0 + r) + bk ≥w⊤
ˆk(x0)(x0 + r) + bˆk(x0),
(6)




## Page 4

x0
F1
F2
F3
Figure 4: For x0 belonging to class 4, let Fk = {x :
fk(x) −f4(x) = 0}. These hyperplanes are depicted in
solid lines and the boundary of P is shown in green dotted
line.
where wk is the kth column of W. Geometrically, the above
problem corresponds to the computation of the distance be-
tween x0 and the complement of the convex polyhedron P,
P =
c\
k=1
{x : fˆk(x0)(x) ≥fk(x)},
(7)
where x0 is located inside P. We denote this distance by
dist(x0, P c). The polyhedron P deﬁnes the region of the
space where f outputs the label ˆk(x0). This setting is de-
picted in Figure 4. The solution to the problem in Eq. (6)
can be computed in closed form as follows. Deﬁne ˆl(x0)
to be the closest hyperplane of the boundary of P (e.g.
ˆl(x0) = 3 in Figure 4). Formally, ˆl(x0) can be computed
as follows
ˆl(x0) = arg min
k̸=ˆk(x0)
fk(x0) −fˆk(x0)(x0)

∥wk −wˆk(x0)∥2
.
(8)
The minimum perturbation r∗(x0) is the vector that
projects x0 on the hyperplane indexed by ˆl(x0), i.e.,
r∗(x0) =
fˆl(x0)(x0) −fˆk(x0)(x0)

∥wˆl(x0) −wˆk(x0)∥2
2
(wˆl(x0) −wˆk(x0)).
(9)
In other words, we ﬁnd the closest projection of x0 on faces
of P.
3.2. General classiﬁer
We now extend the DeepFool algorithm to the general
case of multiclass differentiable classiﬁers.
For general
non-linear classiﬁers, the set P in Eq. (7) that describes the
region of the space where the classiﬁer outputs label ˆk(x0)
is no longer a polyhedron. Following the explained iterative
linearization procedure in the binary case, we approximate
x0
F1
F2
F3
Figure 5: For x0 belonging to class 4, let Fk = {x :
fk(x) −f4(x) = 0}. The linearized zero level sets are
shown in dashed lines and the boundary of the polyhedron
˜P0 in green.
the set P at iteration i by a polyhedron ˜Pi
˜Pi =
c\
k=1
n
x : fk(xi) −fˆk(x0)(xi)
(10)
+ ∇fk(xi)⊤x −∇fˆk(x0)(xi)⊤x ≤0
o
.
We then approximate, at iteration i, the distance between
xi and the complement of P, dist(xi, P c), by dist(xi, ˜P c
i ).
Speciﬁcally, at each iteration of the algorithm, the perturba-
tion vector that reaches the boundary of the polyhedron ˜Pi is
computed, and the current estimate updated. The method is
given in Algorithm 2. It should be noted that the proposed
algorithm operates in a greedy way and is not guaranteed
to converge to the optimal perturbation in (1). However,
we have observed in practice that our algorithm yields very
small perturbations which are believed to be good approxi-
mations of the minimal perturbation.
It should be noted that the optimization strategy of Deep-
Fool is strongly tied to existing optimization techniques. In
the binary case, it can be seen as Newton’s iterative algo-
rithm for ﬁnding roots of a nonlinear system of equations in
the underdetermined case [15]. This algorithm is known as
the normal ﬂow method. The convergence analysis of this
optimization technique can be found for example in [21].
Our algorithm in the binary case can alternatively be seen
as a gradient descent algorithm with an adaptive step size
that is automatically chosen at each iteration. The lineariza-
tion in Algorithm 2 is also similar to a sequential convex
programming where the constraints are linearized at each
step.
3.3. Extension to ℓp norm
In this paper, we have measured the perturbations using
the ℓ2 norm. Our framework is however not limited to this
choice, and the proposed algorithm can simply be adapted




## Page 5

Algorithm 2 DeepFool: multi-class case
1: input: Image x, classiﬁer f.
2: output: Perturbation ˆr.
3:
4: Initialize x0 ←x, i ←0.
5: while ˆk(xi) = ˆk(x0) do
6:
for k ̸= ˆk(x0) do
7:
w′
k ←∇fk(xi) −∇fˆk(x0)(xi)
8:
f ′
k ←fk(xi) −fˆk(x0)(xi)
9:
end for
10:
ˆl ←arg mink̸=ˆk(x0)
|f ′
k|
∥w′
k∥2
11:
ri ←
|f ′
ˆl|
∥w′
ˆl∥2
2 w′
ˆl
12:
xi+1 ←xi + ri
13:
i ←i + 1
14: end while
15: return ˆr = P
i ri
to ﬁnd minimal adversarial perturbations for any ℓp norm
(p ∈[1, ∞)). To do so, the update steps in line 10 and
11 in Algorithm 2 must be respectively substituted by the
following updates
ˆl ←arg min
k̸=ˆk(x0)
|f ′
k|
∥w′
k∥q
,
(11)
ri ←
|f ′
ˆl|
∥w′
ˆl∥q
q
|w′
ˆl|q−1 ⊙sign(w′
ˆl),
(12)
where ⊙is the pointwise product and q =
p
p−1.3 In par-
ticular, when p = ∞(i.e., the supremum norm ℓ∞), these
update steps become
ˆl ←arg min
k̸=ˆk(x0)
|f ′
k|
∥w′
k∥1
,
(13)
ri ←
|f ′
ˆl|
∥w′
ˆl∥1
sign(w′
ˆl).
(14)
4. Experimental results
4.1. Setup
We now test our DeepFool algorithm on deep convo-
lutional neural networks architectures applied to MNIST,
CIFAR-10, and ImageNet image classiﬁcation datasets. We
consider the following deep neural network architectures:
• MNIST: A two-layer fully connected network, and a
two-layer LeNet convoluational neural network archi-
tecture [9]. Both networks are trained with SGD with
momentum using the MatConvNet [20] package.
3To see this, one can apply Holder’s inequality to obtain a lower bound
on the ℓp norm of the perturbation.
• CIFAR-10: We trained a three-layer LeNet architec-
ture, as well as a Network In Network (NIN) architec-
ture [11].
• ILSVRC 2012: We used CaffeNet [7] and GoogLeNet
[17] pre-trained models.
In order to evaluate the robustness to adversarial pertur-
bations of a classiﬁer f, we compute the average robustness
ˆρadv(f), deﬁned by
ˆρadv(f) =
1
|D|
X
x∈D
∥ˆr(x)∥2
∥x∥2
,
(15)
where ˆr(x) is the estimated minimal perturbation obtained
using DeepFool, and D denotes the test set4.
We compare the proposed DeepFool approach to state-
of-the-art techniques to compute adversarial perturbations
in [18] and [4]. The method in [18] solves a series of pe-
nalized optimization problems to ﬁnd the minimal pertur-
bation, whereas [4] estimates the minimal perturbation by
taking the sign of the gradient
ˆr(x) = ϵ sign (∇xJ(θ, x, y)) ,
with J the cost used to train the neural network, θ is the
model parameters, and y is the label of x. The method is
called fast gradient sign method. In practice, in the absence
of general rules to choose the parameter ϵ, we chose the
smallest ϵ such that 90% of the data are misclassiﬁed after
perturbation.5
4.2. Results
We report in Table 1 the accuracy and average robustness
ˆρadv of each classiﬁer computed using different methods.
We also show the running time required for each method to
compute one adversarial sample. It can be seen that Deep-
Fool estimates smaller perturbations (hence closer to min-
imal perturbation deﬁned in (1)) than the ones computed
using the competitive approaches. For example, the aver-
age perturbation obtained using DeepFool is 5 times lower
than the one estimated with [4]. On the ILSVRC2012 chal-
lenge dataset, the average perturbation is one order of mag-
nitude smaller compared to the fast gradient method.
It
should be noted moreover that the proposed approach also
yields slightly smaller perturbation vectors than the method
in [18].
The proposed approach is hence more accurate
in detecting directions that can potentially fool neural net-
works. As a result, DeepFool can be used as a valuable
tool to accurately assess the robustness of classiﬁers. On
4For ILSVRC2012, we used the validation data.
5Using this method, we observed empirically that one cannot reach
100% misclassiﬁcation rate on some datasets. In fact, even by increas-
ing ϵ to be very large, this method can fail in misclassifying all samples.




## Page 6

Classiﬁer
Test error
ˆρadv [DeepFool]
time
ˆρadv [4]
time
ˆρadv [18]
time
LeNet (MNIST)
1%
2.0 × 10−1
110 ms
1.0
20 ms
2.5 × 10−1
> 4 s
FC500-150-10 (MNIST)
1.7%
1.1 × 10−1
50 ms
3.9 × 10−1
10 ms
1.2 × 10−1
> 2 s
NIN (CIFAR-10)
11.5%
2.3 × 10−2
1100 ms
1.2 × 10−1
180 ms
2.4 × 10−2
>50 s
LeNet (CIFAR-10)
22.6%
3.0 × 10−2
220 ms
1.3 × 10−1
50 ms
3.9 × 10−2
>7 s
CaffeNet (ILSVRC2012)
42.6%
2.7 × 10−3
510 ms*
3.5 × 10−2
50 ms*
-
-
GoogLeNet (ILSVRC2012)
31.3%
1.9 × 10−3
800 ms*
4.7 × 10−2
80 ms*
-
-
Table 1: The adversarial robustness of different classiﬁers on different datasets. The time required to compute one sample
for each method is given in the time columns. The times are computed on a Mid-2015 MacBook Pro without CUDA support.
The asterisk marks determines the values computed using a GTX 750 Ti GPU.
the complexity aspect, the proposed approach is substan-
tially faster than the standard method proposed in [18]. In
fact, while the approach [18] involves a costly minimization
of a series of objective functions, we observed empirically
that DeepFool converges in a few iterations (i.e., less than
3) to a perturbation vector that fools the classiﬁer. Hence,
the proposed approach reaches a more accurate perturba-
tion vector compared to state-of-the-art methods, while be-
ing computationally efﬁcient. This makes it readily suitable
to be used as a baseline method to estimate the robustness
of very deep neural networks on large-scale datasets. In that
context, we provide the ﬁrst quantitative evaluation of the
robustness of state-of-the-art classiﬁers on the large-scale
ImageNet dataset. It can be seen that despite their very good
test accuracy, these methods are extremely unstable to ad-
versarial perturbations: a perturbation that is 1000 smaller
in magnitude than the original image is sufﬁcient to fool
state-of-the-art deep neural networks.
We illustrate in Figure 1 perturbed images generated by
the fast gradient sign and DeepFool. It can be observed
that the proposed method generates adversarial perturba-
tions which are hardly perceptible, while the fast gradient
sign method outputs a perturbation image with higher norm.
It should be noted that, when perturbations are mea-
sured using the ℓ∞norm, the above conclusions remain un-
changed: DeepFool yields adversarial perturbations that are
smaller (hence closer to the optimum) compared to other
methods for computing adversarial examples. Table 2 re-
ports the ℓ∞robustness to adversarial perturbations mea-
sured by ˆρ∞
adv(f) =
1
|D|
P
x∈D
∥ˆr(x)∥∞
∥x∥∞, where ˆr(x) is
computed respectively using DeepFool (with p = ∞, see
Section 3.3), and the Fast gradient sign method for MNIST
and CIFAR-10 tasks.
Fine-tuning using adversarial examples In this sec-
tion, we ﬁne-tune the networks of Table 1 on adversarial
examples to build more robust classiﬁers for the MNIST
Classiﬁer
DeepFool
Fast gradient sign
LeNet (MNIST)
0.10
0.26
FC500-150-10 (MNIST)
0.04
0.11
NIN (CIFAR-10)
0.008
0.024
LeNet (CIFAR-10)
0.015
0.028
Table 2: Values of ˆρ∞
adv for four different networks based on
DeepFool (smallest l∞perturbation) and fast gradient sign
method with 90% of misclassiﬁcation.
and CIFAR-10 tasks. Speciﬁcally, for each network, we
performed two experiments: (i) Fine-tuning the network on
DeepFool’s adversarial examples, (ii) Fine-tuning the net-
work on the fast gradient sign adversarial examples. We
ﬁne-tune the networks by performing 5 additional epochs,
with a 50% decreased learning rate only on the perturbed
training set. For each experiment, the same training data
was used through all 5 extra epochs. For the sake of com-
pleteness, we also performed 5 extra epochs on the origi-
nal data. The evolution of ˆρadv for the different ﬁne-tuning
strategies is shown in Figures 6a to 6d, where the robust-
ness ˆρadv is estimated using DeepFool, since this is the most
accurate method, as shown in Table 1. Observe that ﬁne-
tuning with DeepFool adversarial examples signiﬁcantly in-
creases the robustness of the networks to adversarial pertur-
bations even after one extra epoch. For example, the ro-
bustness of the networks on MNIST is improved by 50%
and NIN’s robustness is increased by about 40%. On the
other hand, quite surprisingly, the method in [4] can lead
to a decreased robustness to adversarial perturbations of
the network. We hypothesize that this behavior is due to
the fact that perturbations estimated using the fast gradient
sign method are much larger than minimal adversarial per-


**Table 4 from page 6**

| 0                      | 1          | 2               | 3       | 4          | 5      | 6          | 7     |
|:-----------------------|:-----------|:----------------|:--------|:-----------|:-------|:-----------|:------|
| Classiﬁer              | Test error | ρadv [DeepFool] | time    | ρadv [4]   | time   | ρadv [18]  | time  |
| LeNet (MNIST)          | 1%         | 2.0 × 10−1      | 110 ms  | 1.0        | 20 ms  | 2.5 × 10−1 | > 4 s |
| FC500-150-10 (MNIST)   | 1.7%       | 1.1 × 10−1      | 50 ms   | 3.9 × 10−1 | 10 ms  | 1.2 × 10−1 | > 2 s |
| NIN (CIFAR-10)         | 11.5%      | 2.3 × 10−2      | 1100 ms | 1.2 × 10−1 | 180 ms | 2.4 × 10−2 | >50 s |
| LeNet (CIFAR-10)       | 22.6%      | 3.0 × 10−2      | 220 ms  | 1.3 × 10−1 | 50 ms  | 3.9 × 10−2 | >7 s  |
| CaffeNet (ILSVRC2012)  | 42.6%      | 2.7 × 10−3      | 510 ms* | 3.5 × 10−2 | 50 ms* | -          | -     |
| GoogLeNet (ILSVRC2012) | 31.3%      | 1.9 × 10−3      | 800 ms* | 4.7 × 10−2 | 80 ms* | -          | -     |

**Table 5 from page 6**

| 0                    | 1        | 2                  |
|:---------------------|:---------|:-------------------|
| Classiﬁer            | DeepFool | Fast gradient sign |
| LeNet (MNIST)        | 0.10     | 0.26               |
| FC500-150-10 (MNIST) | 0.04     | 0.11               |
| NIN (CIFAR-10)       | 0.008    | 0.024              |
| LeNet (CIFAR-10)     | 0.015    | 0.028              |



## Page 7

0
1
2
3
4
5
Number of extra epochs
0.12
0.14
0.16
0.18
0.2
0.22
0.24
0.26
0.28
ˆρadv
DeepFool
Fast gradient sign
Clean
(a) Effect of ﬁne-tuning on adversarial examples com-
puted by two different methods for LeNet on MNIST.
0
1
2
3
4
5
Number of extra epochs
0.06
0.07
0.08
0.09
0.1
0.11
0.12
0.13
0.14
0.15
0.16
ˆρadv
DeepFool
Fast gradient sign
Clean
(b) Effect of ﬁne-tuning on adversarial examples com-
puted by two different methods for a fully-connected
network on MNIST.
0
1
2
3
4
5
Number of extra epochs
0.022
0.024
0.026
0.028
0.03
0.032
0.034
0.036
0.038
0.04
0.042
ˆρadv
DeepFool
Fast gradient sign
Clean
(c) Effect of ﬁne-tuning on adversarial examples com-
puted by two different methods for NIN on CIFAR-10.
0
1
2
3
4
5
Number of extra epochs
0.025
0.03
0.035
0.04
ˆρadv
DeepFool
Fast gradient sign
Clean
(d) Effect of ﬁne-tuning on adversarial examples com-
puted by two different methods for LeNet on CIFAR-10.
Figure 6
turbations. Fine-tuning the network with overly perturbed
images decreases the robustness of the networks to adver-
sarial perturbations.
To verify this hypothesis, we com-
pare in Figure 7 the adversarial robustness of a network that
is ﬁne-tuned with the adversarial examples obtained using
DeepFool, where norms of perturbations have been deliber-
ately multiplied by α = 1, 2, 3. Interestingly, we see that
by magnifying the norms of the adversarial perturbations,
the robustness of the ﬁne-tuned network is decreased. This
might explain why overly perturbed images decrease the ro-
bustness of MNIST networks: these perturbations can re-
ally change the class of the digits, hence ﬁne-tuning based
on these examples can lead to a drop of the robustness (for
an illustration, see Figure 8). This lends credence to our
hypothesis, and further shows the importance of designing
accurate methods to compute minimal perturbations.
Table 3 lists the accuracies of the ﬁne-tuned networks. It
can be seen that ﬁne-tuning with DeepFool can improve the
accuracy of the networks. Conversely, ﬁne-tuning with the
approach in [4] has led to a decrease of the test accuracy in
all our experiments. This conﬁrms the explanation that the
fast gradient sign method outputs overly perturbed images
0
1
2
3
4
5
Number of extra epochs
0.05
0.1
0.15
0.2
0.25
0.3
ˆρadv
α = 1
α = 2
α = 3
Figure 7: Fine-tuning based on magniﬁed DeepFool’s ad-
versarial perturbations.
that lead to images that are unlikely to occur in the test data.
Hence, it decreases the performance of the method as it acts
as a regularizer that does not represent the distribution of
the original data. This effect is analogous to geometric data
augmentation schemes, where large transformations of the
original samples have a counter-productive effect on gener-


**Table 6 from page 7**

| 0   | 1   | 2   | 3     | 4                  |
|:----|:----|:----|:------|:-------------------|
|     |     |     | Clean | DeepFool           |
|     |     |     |       | Fast gradient sign |

**Table 7 from page 7**

| 0   | 1                  |
|:----|:-------------------|
|     | DeepFool           |
|     | Fast gradient sign |
|     | Clean              |

**Table 8 from page 7**

| 0                  | 1   | 2   |
|:-------------------|:----|:----|
| DeepFool           |     |     |
| Fast gradient sign |     |     |
| Clean              |     |     |

**Table 9 from page 7**

| 0   | 1   | 2   | 3   | 4     |
|:----|:----|:----|:----|:------|
|     |     |     |     | α = 1 |
|     |     |     |     | α = 2 |
|     |     |     |     | α = 3 |



## Page 8

α = 1
α = 2
α = 3
α = 4
Figure 8: From “1” to “7” : original image classiﬁed as “1”
and the DeepFool perturbed images classiﬁed as “7” using
different values of α.
Classiﬁer
DeepFool
Fast gradient sign
Clean
LeNet (MNIST)
0.8%
4.4%
1%
FC500-150-10 (MNIST)
1.5%
4.9%
1.7%
NIN (CIFAR-10)
11.2%
21.2%
11.5%
LeNet (CIFAR-10)
20.0%
28.6%
22.6%
Table 3: The test error of networks after the ﬁne-tuning on
adversarial examples (after ﬁve epochs). Each columns cor-
respond to a different type of augmented perturbation.
alization.6
To emphasize the importance of a correct estimation of
the minimal perturbation, we now show that using approxi-
mate methods can lead to wrong conclusions regarding the
adversarial robustness of networks. We ﬁne-tune the NIN
classiﬁer on the fast gradient sign adversarial examples. We
follow the procedure described earlier but this time, we de-
creased the learning rate by 90%. We have evaluated the ad-
versarial robustness of this network at different extra epochs
using DeepFool and the fast gradient sign method. As one
can see in Figure 9, the red plot exaggerates the effect of
training on the adversarial examples. Moreover, it is not
sensitive enough to demonstrate the loss of robustness at the
ﬁrst extra epoch. These observations conﬁrm that using an
accurate tool to measure the robustness of classiﬁers is cru-
cial to derive conclusions about the robustness of networks.
5. Conclusion
In this work, we proposed an algorithm, DeepFool, to
compute adversarial examples that fool state-of-the-art clas-
siﬁers. It is based on an iterative linearization of the clas-
siﬁer to generate minimal perturbations that are sufﬁcient
to change classiﬁcation labels. We provided extensive ex-
perimental evidence on three datasets and eight classiﬁers,
showing the superiority of the proposed method over state-
of-the-art methods to compute adversarial perturbations, as
well as the efﬁciency of the proposed approach. Due to
6While the authors of [4] reported an increased generalization perfor-
mance on the MNIST task (from 0.94% to 0.84%) using adversarial reg-
ularization, it should be noted that the their experimental setup is signiﬁ-
cantly different as [4] trained the network based on a modiﬁed cost func-
tion, while we performed straightforward ﬁne-tuning.
0
1
2
3
4
5
Number of extra epochs
0.8
0.9
1
1.1
1.2
1.3
1.4
1.5
Normalized robustness
DeepFool
Fast gradient sign
Figure 9: How the adversarial robustness is judged by dif-
ferent methods. The values are normalized by the corre-
sponding ˆρadvs of the original network.
its accurate estimation of the adversarial perturbations, the
proposed DeepFool algorithm provides an efﬁcient and ac-
curate way to evaluate the robustness of classiﬁers and to
enhance their performance by proper ﬁne-tuning. The pro-
posed approach can therefore be used as a reliable tool to
accurately estimate the minimal perturbation vectors, and
build more robust classiﬁers.
Acknowledgements
This work has been partly supported by the Hasler
Foundation, Switzerland, in the framework of the CORA
project.
References
[1] D. Chicco, P. Sadowski, and P. Baldi.
Deep autoencoder
neural networks for gene ontology annotation predictions. In
ACM Conference on Bioinformatics, Computational Biology,
and Health Informatics, pages 533–540, 2014.
[2] A. Fawzi, O. Fawzi, and P. Frossard.
Analysis of clas-
siﬁers’ robustness to adversarial perturbations.
CoRR,
abs/1502.02590, 2015.
[3] A. Fawzi and P. Frossard. Manitest: Are classiﬁers really
invariant?
In British Machine Vision Conference (BMVC),
pages 106.1–106.13, 2015.
[4] I. J. Goodfellow, J. Shlens, and C. Szegedy. Explaining and
harnessing adversarial examples. In International Confer-
ence on Learning Representations, 2015.
[5] S. Gu and L. Rigazio. Towards deep neural network architec-
tures robust to adversarial examples. CoRR, abs/1412.5068,
2014.
[6] G. E. Hinton, L. Deng, D. Yu, G. E. Dahl, A. Mohamed,
N. Jaitly, A. Senior, V. Vanhoucke, P. Nguyen, T. N. Sainath,
and B. Kingsbury. Deep neural networks for acoustic model-
ing in speech recognition: The shared views of four research
groups. IEEE Signal Process. Mag., 29(6):82–97, 2012.
[7] Y. Jia, E. Shelhamer, J. Donahue, S. Karayev, J. Long, R. Gir-
shick, S. Guadarrama, and T. Darrell.
Caffe: Convolu-
tional architecture for fast feature embedding. In ACM Inter-


**Table 10 from page 8**

| 0   | 1   | 2   | 3                  |
|:----|:----|:----|:-------------------|
|     |     |     | DeepFool           |
|     |     |     | Fast gradient sign |

**Table 11 from page 8**

| 0                    | 1        | 2                  | 3     |
|:---------------------|:---------|:-------------------|:------|
| Classiﬁer            | DeepFool | Fast gradient sign | Clean |
| LeNet (MNIST)        | 0.8%     | 4.4%               | 1%    |
| FC500-150-10 (MNIST) | 1.5%     | 4.9%               | 1.7%  |
| NIN (CIFAR-10)       | 11.2%    | 21.2%              | 11.5% |
| LeNet (CIFAR-10)     | 20.0%    | 28.6%              | 22.6% |



## Page 9

national Conference on Multimedia (MM), pages 675–678.
ACM, 2014.
[8] A. Krizhevsky, I. Sutskever, and G. E. Hinton.
Imagenet
classiﬁcation with deep convolutional neural networks. In
Advances in neural information processing systems (NIPS),
pages 1097–1105, 2012.
[9] Y. LeCun, P. Haffner, L. Bottou, and Y. Bengio.
Object
recognition with gradient-based learning. In Shape, contour
and grouping in computer vision, pages 319–345. 1999.
[10] Y. LeCun, K. Kavukcuoglu, C. Farabet, et al. Convolutional
networks and applications in vision. In IEEE International
Symposium on Circuits and Systems (ISCAS), pages 253–
256, 2010.
[11] M. Lin, Q. Chen, and S. Yan. Network in network. 2014.
[12] T. Mikolov, A. Deoras, D. Povey, L. Burget, and J. ˇCernock`y.
Strategies for training large scale neural network language
models. In IEEE Workshop on Automatic Speech Recogni-
tion and Understanding (ASRU), pages 196–201, 2011.
[13] A. Nguyen, J. Yosinski, and J. Clune. Deep neural networks
are easily fooled: High conﬁdence predictions for unrecog-
nizable images. In IEEE Conference on Computer Vision
and Pattern Recognition (CVPR), pages 427–436, 2015.
[14] B. Pepik, R. Benenson, T. Ritschel, and B. Schiele. What is
holding back convnets for detection? In Pattern Recognition,
pages 517–528. Springer, 2015.
[15] A. P. Ruszczy´nski.
Nonlinear optimization, volume 13.
Princeton university press, 2006.
[16] M. Spencer, J. Eickholt, and J. Cheng. A deep learning net-
work approach to ab initio protein secondary structure pre-
diction.
IEEE/ACM Trans. Comput. Biol. Bioinformatics,
12(1):103–112, 2015.
[17] C. Szegedy,
W. Liu,
Y. Jia,
P. Sermanet,
S. Reed,
D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich.
Going deeper with convolutions.
In IEEE Conference on
Computer Vision and Pattern Recognition (CVPR), pages 1–
9, 2015.
[18] C. Szegedy, W. Zaremba, I. Sutskever, J. Bruna, D. Erhan,
I. J. Goodfellow, and R. Fergus.
Intriguing properties of
neural networks. In International Conference on Learning
Representations (ICLR), 2014.
[19] C.-Y. Tsai and D. Cox.
Are deep learning algorithms
easily hackable?
http://coxlab.github.io/
ostrichinator.
[20] A. Vedaldi and K. Lenc. Matconvnet: Convolutional neural
networks for matlab. In ACM International Conference on
Multimedia (MM), pages 689–692, 2015.
[21] H. F. Walker and L. T. Watson. Least-change secant update
methods for underdetermined systems. SIAM Journal on nu-
merical analysis, 27(5):1227–1262, 1990.


