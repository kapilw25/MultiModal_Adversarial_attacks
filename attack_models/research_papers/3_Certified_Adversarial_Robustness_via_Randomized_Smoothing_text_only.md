# 3_Certified_Adversarial_Robustness_via_Randomized_Smoothing

## Document Information

- **Source**: attack_models/research_papers/3_Certified_Adversarial_Robustness_via_Randomized_Smoothing.pdf
- **Pages**: 36
- **Tables**: 40



## Page 1

Certiﬁed Adversarial Robustness via Randomized Smoothing
Jeremy Cohen 1 Elan Rosenfeld 1 J. Zico Kolter 1 2
Abstract
We show how to turn any classiﬁer that classi-
ﬁes well under Gaussian noise into a new classi-
ﬁer that is certiﬁably robust to adversarial per-
turbations under the ℓ2 norm.
This “random-
ized smoothing” technique has been proposed re-
cently in the literature, but existing guarantees are
loose. We prove a tight robustness guarantee in
ℓ2 norm for smoothing with Gaussian noise. We
use randomized smoothing to obtain an ImageNet
classiﬁer with e.g. a certiﬁed top-1 accuracy of
49% under adversarial perturbations with ℓ2 norm
less than 0.5 (=127/255). No certiﬁed defense
has been shown feasible on ImageNet except for
smoothing. On smaller-scale datasets where com-
peting approaches to certiﬁed ℓ2 robustness are
viable, smoothing delivers higher certiﬁed accu-
racies. Our strong empirical results suggest that
randomized smoothing is a promising direction
for future research into adversarially robust classi-
ﬁcation. Code and models are available at http:
//github.com/locuslab/smoothing.
1. Introduction
Modern image classiﬁers achieve high accuracy on i.i.d.
test sets but are not robust to small, adversarially-chosen
perturbations of their inputs (Szegedy et al., 2014; Biggio
et al., 2013). Given an image x correctly classiﬁed by, say,
a neural network, an adversary can usually engineer an ad-
versarial perturbation δ so small that x + δ looks just like
x to the human eye, yet the network classiﬁes x + δ as a
different, incorrect class. Many works have proposed heuris-
tic methods for training classiﬁers intended to be robust to
adversarial perturbations. However, most of these heuristics
have been subsequently shown to fail against suitably pow-
erful adversaries (Carlini & Wagner, 2017; Athalye et al.,
2018; Uesato et al., 2018). In response, a line of work on
1Carnegie Mellon University 2Bosch Center for AI. Correspon-
dence to: Jeremy Cohen <jeremycohen@cmu.edu>.
Proceedings of the 36 th International Conference on Machine
Learning, Long Beach, California, PMLR 97, 2019. Copyright
2019 by the author(s).
x
pA
pB
Figure 1. Evaluating the smoothed classiﬁer at an input x. Left:
the decision regions of the base classiﬁer f are drawn in differ-
ent colors. The dotted lines are the level sets of the distribution
N(x, σ2I). Right: the distribution f(N(x, σ2I)). As discussed
below, pA is a lower bound on the probability of the top class and
pB is an upper bound on the probability of each other class. Here,
g(x) is “blue.”
.
certiﬁable robustness studies classiﬁers whose prediction at
any point x is veriﬁably constant within some set around x
(Wong & Kolter, 2018; Raghunathan et al., 2018a, e.g.). In
most of these works, the robust classiﬁer takes the form of a
neural network. Unfortunately, all existing approaches for
certifying the robustness of neural networks have trouble
scaling to networks that are large and expressive enough to
solve problems like ImageNet.
One workaround is to look for robust classiﬁers that are not
neural networks. Recently, two papers (Lecuyer et al., 2019;
Li et al., 2018) showed that an operation we call randomized
smoothing1 can transform any arbitrary base classiﬁer f into
a new “smoothed classiﬁer” g that is certiﬁably robust in
ℓ2 norm. Let f be an arbitrary classiﬁer which maps inputs
Rd to classes Y. For any input x, the smoothed classiﬁer’s
prediction g(x) is deﬁned to be the class which f is most
likely to classify the random variable N(x, σ2I) as. That is,
g(x) returns the most probable prediction by f of random
Gaussian corruptions of x.
If the base classiﬁer f is most likely to classify N(x, σ2I)
as x’s correct class, then the smoothed classiﬁer g will be
1Smoothing was proposed under the name “PixelDP” (for dif-
ferential privacy). We use a different name since our improved
analysis does not involve differential privacy.
arXiv:1902.02918v2  [cs.LG]  15 Jun 2019


**Table 1 from page 1**

| 0                                                                   |
|:--------------------------------------------------------------------|
| Jeremy Cohen 1 Elan Rosenfeld 1                                     |
| J. Zico Kolter 1 2                                                  |
| Abstract                                                            |
| We show how to turn any classiﬁer                                   |
| that classi-                                                        |
| ﬁes well under Gaussian noise into a new classi-                    |
| pA                                                                  |
| ﬁer                                                                 |
| that                                                                |
| is certiﬁably robust                                                |
| to adversarial per-                                                 |
| turbations under                                                    |
| This “random-                                                       |
| the (cid:96)2 norm.                                                 |
| pB                                                                  |
| x                                                                   |
| ized smoothing” technique has been proposed re-                     |
| cently in the literature, but existing guarantees are               |
| loose. We prove a tight robustness guarantee in                     |
| (cid:96)2 norm for smoothing with Gaussian noise. We                |
| use randomized smoothing to obtain an ImageNet                      |
| classiﬁer with e.g. a certiﬁed top-1 accuracy of                    |
| 49% under adversarial perturbations with (cid:96)2 norm             |
| Figure 1. Evaluating the smoothed classiﬁer at an input x. Left:    |
| less than 0.5 (=127/255). No certiﬁed defense                       |
| the decision regions of the base classiﬁer f are drawn in differ-   |
| ent colors. The dotted lines are the level sets of the distribution |
| has been shown feasible on ImageNet except for                      |
| N (x, σ2I). Right:                                                  |
| the distribution f (N (x, σ2I)). As discussed                       |
| smoothing. On smaller-scale datasets where com-                     |
| below, pA is a lower bound on the probability of the top class and  |
| peting approaches to certiﬁed (cid:96)2 robustness are              |
| pB is an upper bound on the probability of each other class. Here,  |
| viable, smoothing delivers higher certiﬁed accu-                    |
| g(x) is “blue.”                                                     |
| racies. Our strong empirical results suggest that                   |
| .                                                                   |
| randomized smoothing is a promising direction                       |
| for future research into adversarially robust classi-               |
| certiﬁable robustness studies classiﬁers whose prediction at        |
| ﬁcation. Code and models are available at http:                     |
| any point x is veriﬁably constant within some set around x          |
| //github.com/locuslab/smoothing.                                    |
| (Wong & Kolter, 2018; Raghunathan et al., 2018a, e.g.). In          |
| most of these works, the robust classiﬁer takes the form of a       |
| neural network. Unfortunately, all existing approaches for          |
| 1. Introduction                                                     |
| certifying the robustness of neural networks have trouble           |
| scaling to networks that are large and expressive enough to         |
| Modern image classiﬁers achieve high accuracy on i.i.d.             |
| solve problems like ImageNet.                                       |
| test sets but are not robust                                        |
| to small, adversarially-chosen                                      |
| perturbations of their inputs (Szegedy et al., 2014; Biggio         |
| One workaround is to look for robust classiﬁers that are not        |
| et al., 2013). Given an image x correctly classiﬁed by, say,        |
| neural networks. Recently, two papers (Lecuyer et al., 2019;        |
| a neural network, an adversary can usually engineer an ad-          |
| Li et al., 2018) showed that an operation we call randomized        |
| versarial perturbation δ so small that x + δ looks just like        |
| smoothing1 can transform any arbitrary base classiﬁer f into        |
| x to the human eye, yet                                             |
| the network classiﬁes x + δ as a                                    |
| a new “smoothed classiﬁer” g that                                   |
| is certiﬁably robust                                                |
| in                                                                  |
| different, incorrect class. Many works have proposed heuris-        |
| (cid:96)2 norm. Let f be an arbitrary classiﬁer which maps inputs   |
| tic methods for training classiﬁers intended to be robust to        |
| Rd to classes Y. For any input x, the smoothed classiﬁer’s          |
| adversarial perturbations. However, most of these heuristics        |
| prediction g(x) is deﬁned to be the class which f is most           |
| have been subsequently shown to fail against suitably pow-          |
| likely to classify the random variable N (x, σ2I) as. That is,      |
| erful adversaries (Carlini & Wagner, 2017; Athalye et al.,          |
| g(x) returns the most probable prediction by f of random            |
| 2018; Uesato et al., 2018).                                         |
| In response, a line of work on                                      |
| Gaussian corruptions of x.                                          |
| If the base classiﬁer f is most likely to classify N (x, σ2I)       |
| 1Carnegie Mellon University 2Bosch Center for AI. Correspon-        |
| dence to: Jeremy Cohen <jeremycohen@cmu.edu>.                       |
| as x’s correct class, then the smoothed classiﬁer g will be         |
| Proceedings of                                                      |
| the 36 th International Conference on Machine                       |
| 1Smoothing was proposed under the name “PixelDP” (for dif-          |
| ferential privacy). We use a different name since our improved      |
| Learning, Long Beach, California, PMLR 97, 2019. Copyright          |
| analysis does not involve differential privacy.                     |
| 2019 by the author(s).                                              |



## Page 2

Certiﬁed Adversarial Robustness via Randomized Smoothing
correct at x. But the smoothed classiﬁer g will also possess
a desirable property that the base classiﬁer may lack: one
can verify that g’s prediction is constant within an ℓ2 ball
around any input x, simply by estimating the probabilities
with which f classiﬁes N(x, σ2I) as each class. The higher
the probability with which f classiﬁes N(x, σ2I) as the
most probable class, the larger the ℓ2 radius around x in
which g provably returns that class.
Lecuyer et al. (2019) proposed randomized smoothing as
a provable adversarial defense, and used it to train the ﬁrst
certiﬁably robust classiﬁer for ImageNet. Subsequently, Li
et al. (2018) proved a stronger robustness guarantee. How-
ever, both of these guarantees are loose, in the sense that
the smoothed classiﬁer g is provably always more robust
than the guarantee indicates. In this paper, we prove the
ﬁrst tight robustness guarantee for randomized smoothing.
Our analysis reveals that smoothing with Gaussian noise
naturally induces certiﬁable robustness under the ℓ2 norm.
We suspect that other, as-yet-unknown noise distributions
might induce robustness to other perturbation sets such as
general ℓp norm balls.
Randomized smoothing has one major drawback. If f is
a neural network, it is not possible to exactly compute the
probabilities with which f classiﬁes N(x, σ2I) as each
class. Therefore, it is not possible to exactly evaluate g’s
prediction at any input x, or to exactly compute the radius
in which this prediction is certiﬁably robust. Instead, we
present Monte Carlo algorithms for both tasks that are guar-
anteed to succeed with arbitrarily high probability.
Despite this drawback, randomized smoothing enjoys sev-
eral compelling advantages over other certiﬁably robust
classiﬁers proposed in the literature: it makes no assump-
tions about the base classiﬁer’s architecture, it is simple to
implement and understand, and, most importantly, it per-
mits the use of arbitrarily large neural networks as the base
classiﬁer. In contrast, other certiﬁed defenses do not cur-
Table 1. Approximate certiﬁed accuracy on ImageNet. Each row
shows a radius r, the best hyperparameter σ for that radius, the
approximate certiﬁed accuracy at radius r of the corresponding
smoothed classiﬁer, and the standard accuracy of the corresponding
smoothed classiﬁer. To give a sense of scale, a perturbation with
ℓ2 radius 1.0 could change one pixel by 255, ten pixels by 80, 100
pixels by 25, or 1000 pixels by 8. Random guessing on ImageNet
would attain 0.1% accuracy.
ℓ2 RADIUS
BEST σ
CERT. ACC (%)
STD. ACC(%)
0.5
0.25
49
67
1.0
0.50
37
57
2.0
0.50
19
57
3.0
1.00
12
44
Figure 2. The smoothed classiﬁer’s prediction at an input x (left)
is deﬁned as the most likely prediction by the base classiﬁer on
random Gaussian corruptions of x (right; σ = 0.5). Note that this
Gaussian noise is much larger in magnitude than the adversarial
perturbations to which g is provably robust. One interpretation of
randomized smoothing is that these large random perturbations
“drown out” small adversarial perturbations.
rently scale to large networks. Indeed, smoothing is the only
certiﬁed adversarial defense which has been shown feasible
on the full-resolution ImageNet classiﬁcation task.
We use randomized smoothing to train state-of-the-art certi-
ﬁably ℓ2-robust ImageNet classiﬁers; for example, one of
them achieves 49% provable top-1 accuracy under adver-
sarial perturbations with ℓ2 norm less than 127/255 (Table
1). We also demonstrate that on smaller-scale datasets like
CIFAR-10 and SHVN, where competing approaches to cer-
tiﬁed ℓ2 robustness are feasible, randomized smoothing can
deliver better certiﬁed accuracies, both because it enables
the use of larger networks and because it does not constrain
the expressivity of the base classiﬁer.
2. Related Work
Many works have proposed classiﬁers intended to be ro-
bust to adversarial perturbations. These approaches can
be broadly divided into empirical defenses, which empiri-
cally seem robust to known adversarial attacks, and certiﬁed
defenses, which are provably robust to certain kinds of ad-
versarial perturbations.
Empirical defenses
The most successful empirical de-
fense to date is adversarial training (Goodfellow et al.,
2015; Kurakin et al., 2017; Madry et al., 2018), in which
adversarial examples are found during training (often using
projected gradient descent) and added to the training set.
Unfortunately, it is typically impossible to tell whether a
prediction by an empirically robust classiﬁer is truly robust
to adversarial perturbations; the most that can be said is that
a speciﬁc attack was unable to ﬁnd any. In fact, many heuris-
tic defenses proposed in the literature were later “broken”
by stronger adversaries (Carlini & Wagner, 2017; Athalye
et al., 2018; Uesato et al., 2018; Athalye & Carlini, 2018).


**Table 2 from page 2**

| 0                                                                         | 1                                                                  |
|:--------------------------------------------------------------------------|:-------------------------------------------------------------------|
| Certiﬁed Adversarial Robustness via Randomized Smoothing                  |                                                                    |
| correct at x. But the smoothed classiﬁer g will also possess              |                                                                    |
| a desirable property that the base classiﬁer may lack: one                |                                                                    |
| can verify that g’s prediction is constant within an (cid:96)2 ball       |                                                                    |
| around any input x, simply by estimating the probabilities                |                                                                    |
| with which f classiﬁes N (x, σ2I) as each class. The higher               |                                                                    |
| the probability with which f classiﬁes N (x, σ2I) as the                  |                                                                    |
| most probable class,                                                      |                                                                    |
| the larger the (cid:96)2 radius around x in                               |                                                                    |
| which g provably returns that class.                                      |                                                                    |
| Lecuyer et al. (2019) proposed randomized smoothing as                    |                                                                    |
| a provable adversarial defense, and used it to train the ﬁrst             |                                                                    |
|                                                                           | Figure 2. The smoothed classiﬁer’s prediction at an input x (left) |
| certiﬁably robust classiﬁer for ImageNet. Subsequently, Li                |                                                                    |
|                                                                           | is deﬁned as the most                                              |
|                                                                           | likely prediction by the base classiﬁer on                         |
| et al. (2018) proved a stronger robustness guarantee. How-                |                                                                    |
|                                                                           | random Gaussian corruptions of x (right; σ = 0.5). Note that this  |
| ever, both of these guarantees are loose,                                 |                                                                    |
| in the sense that                                                         |                                                                    |
|                                                                           | Gaussian noise is much larger in magnitude than the adversarial    |
| the smoothed classiﬁer g is provably always more robust                   |                                                                    |
|                                                                           | perturbations to which g is provably robust. One interpretation of |
| than the guarantee indicates.                                             |                                                                    |
| In this paper, we prove the                                               |                                                                    |
|                                                                           | randomized smoothing is that                                       |
|                                                                           | these large random perturbations                                   |
| ﬁrst tight robustness guarantee for randomized smoothing.                 | “drown out” small adversarial perturbations.                       |
| Our analysis reveals that smoothing with Gaussian noise                   |                                                                    |
| naturally induces certiﬁable robustness under the (cid:96)2 norm.         |                                                                    |
| We suspect that other, as-yet-unknown noise distributions                 |                                                                    |
|                                                                           | rently scale to large networks. Indeed, smoothing is the only      |
| might induce robustness to other perturbation sets such as                |                                                                    |
|                                                                           | certiﬁed adversarial defense which has been shown feasible         |
| general (cid:96)p norm balls.                                             |                                                                    |
|                                                                           | on the full-resolution ImageNet classiﬁcation task.                |
| Randomized smoothing has one major drawback.                              |                                                                    |
| If f is                                                                   |                                                                    |
|                                                                           | We use randomized smoothing to train state-of-the-art certi-       |
| a neural network, it is not possible to exactly compute the               |                                                                    |
|                                                                           | ﬁably (cid:96)2-robust ImageNet classiﬁers; for example, one of    |
| probabilities with which f                                                |                                                                    |
| classiﬁes N (x, σ2I) as each                                              |                                                                    |
|                                                                           | them achieves 49% provable top-1 accuracy under adver-             |
| class. Therefore,                                                         |                                                                    |
| it                                                                        |                                                                    |
| is not possible to exactly evaluate g’s                                   |                                                                    |
|                                                                           | sarial perturbations with (cid:96)2 norm less than 127/255 (Table  |
| prediction at any input x, or to exactly compute the radius               |                                                                    |
|                                                                           | 1). We also demonstrate that on smaller-scale datasets like        |
| in which this prediction is certiﬁably robust.                            |                                                                    |
| Instead, we                                                               |                                                                    |
|                                                                           | CIFAR-10 and SHVN, where competing approaches to cer-              |
| present Monte Carlo algorithms for both tasks that are guar-              |                                                                    |
|                                                                           | tiﬁed (cid:96)2 robustness are feasible, randomized smoothing can  |
| anteed to succeed with arbitrarily high probability.                      |                                                                    |
|                                                                           | deliver better certiﬁed accuracies, both because it enables        |
|                                                                           | the use of larger networks and because it does not constrain       |
| Despite this drawback, randomized smoothing enjoys sev-                   |                                                                    |
|                                                                           | the expressivity of the base classiﬁer.                            |
| eral compelling advantages over other certiﬁably robust                   |                                                                    |
| classiﬁers proposed in the literature:                                    |                                                                    |
| it makes no assump-                                                       |                                                                    |
| tions about the base classiﬁer’s architecture, it is simple to            | 2. Related Work                                                    |
| implement and understand, and, most                                       |                                                                    |
| importantly,                                                              |                                                                    |
| it per-                                                                   |                                                                    |
|                                                                           | Many works have proposed classiﬁers intended to be ro-             |
| mits the use of arbitrarily large neural networks as the base             |                                                                    |
|                                                                           | bust                                                               |
|                                                                           | to adversarial perturbations.                                      |
|                                                                           | These approaches can                                               |
| classiﬁer.                                                                |                                                                    |
| In contrast, other certiﬁed defenses do not cur-                          |                                                                    |
|                                                                           | be broadly divided into empirical defenses, which empiri-          |
|                                                                           | cally seem robust to known adversarial attacks, and certiﬁed       |
|                                                                           | defenses, which are provably robust to certain kinds of ad-        |
| Table 1. Approximate certiﬁed accuracy on ImageNet. Each row              |                                                                    |
|                                                                           | versarial perturbations.                                           |
| shows a radius r,                                                         |                                                                    |
| the best hyperparameter σ for that radius,                                |                                                                    |
| the                                                                       |                                                                    |
| approximate certiﬁed accuracy at radius r of the corresponding            |                                                                    |
|                                                                           | Empirical defenses                                                 |
|                                                                           | The most successful empirical de-                                  |
| smoothed classiﬁer, and the standard accuracy of the corresponding        |                                                                    |
|                                                                           | fense to date is adversarial                                       |
|                                                                           | training (Goodfellow et al.,                                       |
| smoothed classiﬁer. To give a sense of scale, a perturbation with         |                                                                    |
|                                                                           | 2015; Kurakin et al., 2017; Madry et al., 2018),                   |
|                                                                           | in which                                                           |
| (cid:96)2 radius 1.0 could change one pixel by 255, ten pixels by 80, 100 |                                                                    |
|                                                                           | adversarial examples are found during training (often using        |
| pixels by 25, or 1000 pixels by 8. Random guessing on ImageNet            |                                                                    |
| would attain 0.1% accuracy.                                               | projected gradient descent) and added to the training set.         |
|                                                                           | Unfortunately,                                                     |
|                                                                           | it                                                                 |
|                                                                           | is typically impossible to tell whether a                          |
| BEST σ                                                                    | prediction by an empirically robust classiﬁer is truly robust      |
| CERT. ACC (%)                                                             |                                                                    |
| STD. ACC(%)                                                               |                                                                    |
| (cid:96)2 RADIUS                                                          |                                                                    |
|                                                                           | to adversarial perturbations; the most that can be said is that    |
| 0.5                                                                       |                                                                    |
| 0.25                                                                      |                                                                    |
| 49                                                                        |                                                                    |
| 67                                                                        |                                                                    |
|                                                                           | a speciﬁc attack was unable to ﬁnd any. In fact, many heuris-      |
| 1.0                                                                       |                                                                    |
| 0.50                                                                      |                                                                    |
| 37                                                                        |                                                                    |
| 57                                                                        |                                                                    |
|                                                                           | tic defenses proposed in the literature were later “broken”        |
| 2.0                                                                       |                                                                    |
| 0.50                                                                      |                                                                    |
| 19                                                                        |                                                                    |
| 57                                                                        |                                                                    |
| 3.0                                                                       | by stronger adversaries (Carlini & Wagner, 2017; Athalye           |
| 1.00                                                                      |                                                                    |
| 12                                                                        |                                                                    |
| 44                                                                        |                                                                    |
|                                                                           | et al., 2018; Uesato et al., 2018; Athalye & Carlini, 2018).       |



## Page 3

Certiﬁed Adversarial Robustness via Randomized Smoothing
Aiming to escape this cat-and-mouse game, a growing body
of work has focused on defenses with formal guarantees.
Certiﬁed defenses
A classiﬁer is said to be certiﬁably ro-
bust if for any input x, one can easily obtain a guarantee that
the classiﬁer’s prediction is constant within some set around
x, often an ℓ2 or ℓ∞ball. In most work in this area, the
certiﬁably robust classiﬁer is a neural network. Some works
propose algorithms for certifying the robustness of generi-
cally trained networks, while others (Wong & Kolter, 2018;
Raghunathan et al., 2018a) propose both a robust training
method and a complementary certiﬁcation mechanism.
Certiﬁcation methods are either exact (a.k.a “complete”) or
conservative (a.k.a “sound but incomplete”). In the context
of ℓp norm-bounded perturbations, exact methods take a
classiﬁer g, input x, and radius r, and report whether or
not there exists a perturbation δ within ∥δ∥≤r for which
g(x) ̸= g(x + δ). In contrast, conservative methods either
certify that no such perturbation exists or decline to make a
certiﬁcation; they may decline even when it is true that no
such perturbation exists. Exact methods are usually based
on Satisﬁability Modulo Theories (Katz et al., 2017; Carlini
et al., 2017; Ehlers, 2017; Huang et al., 2017) or mixed
integer linear programming (Cheng et al., 2017; Lomuscio
& Maganti, 2017; Dutta et al., 2017; Fischetti & Jo, 2018;
Bunel et al., 2018). Unfortunately, no exact methods have
been shown to scale beyond moderate-sized (100,000 acti-
vations) networks (Tjeng et al., 2019), and networks of that
size can only be veriﬁed when they are trained in a manner
that impairs their expressivity.
Conservative certiﬁcation is more scalable. Some conser-
vative methods bound the global Lipschitz constant of the
neural network (Gouk et al., 2018; Tsuzuku et al., 2018;
Anil et al., 2019; Cisse et al., 2017), but these approaches
tend to be very loose on expressive networks. Others mea-
sure the local smoothness of the network in the vicinity of a
particular input x. In theory, one could obtain a robustness
guarantee via an upper bound on the local Lipschitz con-
stant of the network (Hein & Andriushchenko, 2017), but
computing this quantity is intractable for general neural net-
works. Instead, a panoply of practical solutions have been
proposed in the literature (Wong & Kolter, 2018; Wang et al.,
2018a;b; Raghunathan et al., 2018a;b; Wong et al., 2018;
Dvijotham et al., 2018b;a; Croce et al., 2019; Gehr et al.,
2018; Mirman et al., 2018; Singh et al., 2018; Gowal et al.,
2018; Weng et al., 2018a; Zhang et al., 2018). Two themes
stand out. Some approaches cast veriﬁcation as an opti-
mization problem and import tools such as relaxation and
duality from the optimization literature to provide conserva-
tive guarantees (Wong & Kolter, 2018; Wong et al., 2018;
Raghunathan et al., 2018a;b; Dvijotham et al., 2018b;a).
Others step through the network layer by layer, maintaining
at each layer an outer approximation of the set of activations
reachable by a perturbed input (Mirman et al., 2018; Singh
et al., 2018; Gowal et al., 2018; Weng et al., 2018a; Zhang
et al., 2018). None of these local certiﬁcation methods have
been shown to be feasible on networks that are large and
expressive enough to solve modern machine learning prob-
lems like the ImageNet classiﬁcation task. Also, all either
assume speciﬁc network architectures (e.g. ReLU activa-
tions or a layered feedforward structure) or require extensive
customization for new network architectures.
Related work involving noise
Prior works have proposed
using a network’s robustness to Gaussian noise as a proxy
for its robustness to adversarial perturbations (Weng et al.,
2018b; Ford et al., 2019), and have suggested that Gaussian
data augmentation could supplement or replace adversar-
ial training (Zantedeschi et al., 2017; Kannan et al., 2018).
Smilkov et al. (2017) observed that averaging a classiﬁer’s
input gradients over Gaussian corruptions of an image yields
very interpretable saliency maps. The robustness of neural
networks to random noise has been analyzed both theo-
retically (Fawzi et al., 2016; Franceschi et al., 2018) and
empirically (Dodge & Karam, 2017). Finally, Webb et al.
(2019) proposed a statistical technique for estimating the
noise robustness of a classiﬁer more efﬁciently than naive
Monte Carlo simulation; we did not use this technique since
it appears to lack formal high-probability guarantees. While
these works hypothesized relationships between a neural net-
work’s robustness to random noise and the same network’s
robustness to adversarial perturbations, randomized smooth-
ing instead uses a classiﬁer’s robustness to random noise to
create a new classiﬁer robust to adversarial perturbations.
Randomized smoothing
Randomized smoothing has
been studied previously for adversarial robustness. Sev-
eral works (Liu et al., 2018; Cao & Gong, 2017) proposed
similar techniques as heuristic defenses, but did not prove
any guarantees. Lecuyer et al. (2019) used inequalities
from the differential privacy literature to prove an ℓ2 and
ℓ1 robustness guarantee for smoothing with Gaussian and
Laplace noise, respectively. Subsequently, Li et al. (2018)
used tools from information theory to prove a stronger ℓ2 ro-
bustness guarantee for Gaussian noise. However, all of these
robustness guarantees are loose. In contrast, we prove a tight
robustness guarantee in ℓ2 norm for randomized smoothing
with Gaussian noise.
3. Randomized smoothing
Consider a classiﬁcation problem from Rd to classes Y.
As discussed above, randomized smoothing is a method for
constructing a new, “smoothed” classiﬁer g from an arbitrary
base classiﬁer f. When queried at x, the smoothed classiﬁer
g returns whichever class the base classiﬁer f is most likely


**Table 3 from page 3**

| 0                                                                          | 1                                                                    |
|:---------------------------------------------------------------------------|:---------------------------------------------------------------------|
| Certiﬁed Adversarial Robustness via Randomized Smoothing                   |                                                                      |
| Aiming to escape this cat-and-mouse game, a growing body                   | reachable by a perturbed input (Mirman et al., 2018; Singh           |
| of work has focused on defenses with formal guarantees.                    | et al., 2018; Gowal et al., 2018; Weng et al., 2018a; Zhang          |
|                                                                            | et al., 2018). None of these local certiﬁcation methods have         |
|                                                                            | been shown to be feasible on networks that are large and             |
| Certiﬁed defenses                                                          |                                                                      |
| A classiﬁer is said to be certiﬁably ro-                                   |                                                                      |
|                                                                            | expressive enough to solve modern machine learning prob-             |
| bust if for any input x, one can easily obtain a guarantee that            |                                                                      |
|                                                                            | lems like the ImageNet classiﬁcation task. Also, all either          |
| the classiﬁer’s prediction is constant within some set around              |                                                                      |
|                                                                            | assume speciﬁc network architectures (e.g. ReLU activa-              |
| In most work in this area,                                                 |                                                                      |
| the                                                                        |                                                                      |
| x, often an (cid:96)2 or (cid:96)∞ ball.                                   |                                                                      |
|                                                                            | tions or a layered feedforward structure) or require extensive       |
| certiﬁably robust classiﬁer is a neural network. Some works                |                                                                      |
|                                                                            | customization for new network architectures.                         |
| propose algorithms for certifying the robustness of generi-                |                                                                      |
| cally trained networks, while others (Wong & Kolter, 2018;                 |                                                                      |
| Raghunathan et al., 2018a) propose both a robust training                  |                                                                      |
|                                                                            | Related work involving noise                                         |
|                                                                            | Prior works have proposed                                            |
| method and a complementary certiﬁcation mechanism.                         |                                                                      |
|                                                                            | using a network’s robustness to Gaussian noise as a proxy            |
|                                                                            | for its robustness to adversarial perturbations (Weng et al.,        |
| Certiﬁcation methods are either exact (a.k.a “complete”) or                |                                                                      |
|                                                                            | 2018b; Ford et al., 2019), and have suggested that Gaussian          |
| conservative (a.k.a “sound but incomplete”). In the context                |                                                                      |
|                                                                            | data augmentation could supplement or replace adversar-              |
| of (cid:96)p norm-bounded perturbations, exact methods take a              |                                                                      |
|                                                                            | ial training (Zantedeschi et al., 2017; Kannan et al., 2018).        |
| classiﬁer g,                                                               |                                                                      |
| input x, and radius r, and report whether or                               |                                                                      |
|                                                                            | Smilkov et al. (2017) observed that averaging a classiﬁer’s          |
| not there exists a perturbation δ within (cid:107)δ(cid:107) ≤ r for which |                                                                      |
|                                                                            | input gradients over Gaussian corruptions of an image yields         |
| g(x) (cid:54)= g(x + δ). In contrast, conservative methods either          |                                                                      |
|                                                                            | very interpretable saliency maps. The robustness of neural           |
| certify that no such perturbation exists or decline to make a              |                                                                      |
|                                                                            | networks to random noise has been analyzed both theo-                |
| certiﬁcation; they may decline even when it is true that no                |                                                                      |
|                                                                            | retically (Fawzi et al., 2016; Franceschi et al., 2018) and          |
| such perturbation exists. Exact methods are usually based                  |                                                                      |
|                                                                            | empirically (Dodge & Karam, 2017). Finally, Webb et al.              |
| on Satisﬁability Modulo Theories (Katz et al., 2017; Carlini               |                                                                      |
|                                                                            | (2019) proposed a statistical                                        |
|                                                                            | technique for estimating the                                         |
| et al., 2017; Ehlers, 2017; Huang et al., 2017) or mixed                   |                                                                      |
|                                                                            | noise robustness of a classiﬁer more efﬁciently than naive           |
| integer linear programming (Cheng et al., 2017; Lomuscio                   |                                                                      |
|                                                                            | Monte Carlo simulation; we did not use this technique since          |
| & Maganti, 2017; Dutta et al., 2017; Fischetti & Jo, 2018;                 |                                                                      |
|                                                                            | it appears to lack formal high-probability guarantees. While         |
| Bunel et al., 2018). Unfortunately, no exact methods have                  |                                                                      |
|                                                                            | these works hypothesized relationships between a neural net-         |
| been shown to scale beyond moderate-sized (100,000 acti-                   |                                                                      |
|                                                                            | work’s robustness to random noise and the same network’s             |
| vations) networks (Tjeng et al., 2019), and networks of that               |                                                                      |
|                                                                            | robustness to adversarial perturbations, randomized smooth-          |
| size can only be veriﬁed when they are trained in a manner                 |                                                                      |
|                                                                            | ing instead uses a classiﬁer’s robustness to random noise to         |
| that impairs their expressivity.                                           |                                                                      |
|                                                                            | create a new classiﬁer robust to adversarial perturbations.          |
| Conservative certiﬁcation is more scalable. Some conser-                   |                                                                      |
| vative methods bound the global Lipschitz constant of the                  |                                                                      |
| neural network (Gouk et al., 2018; Tsuzuku et al., 2018;                   | Randomized                                                           |
|                                                                            | smoothing                                                            |
|                                                                            | Randomized                                                           |
|                                                                            | smoothing                                                            |
|                                                                            | has                                                                  |
| Anil et al., 2019; Cisse et al., 2017), but these approaches               | been studied previously for adversarial                              |
|                                                                            | robustness.                                                          |
|                                                                            | Sev-                                                                 |
| tend to be very loose on expressive networks. Others mea-                  | eral works (Liu et al., 2018; Cao & Gong, 2017) proposed             |
| sure the local smoothness of the network in the vicinity of a              | similar techniques as heuristic defenses, but did not prove          |
| particular input x. In theory, one could obtain a robustness               | any guarantees.                                                      |
|                                                                            | Lecuyer et al.                                                       |
|                                                                            | (2019) used inequalities                                             |
| guarantee via an upper bound on the local Lipschitz con-                   | from the differential privacy literature to prove an (cid:96)2 and   |
| stant of the network (Hein & Andriushchenko, 2017), but                    | (cid:96)1 robustness guarantee for smoothing with Gaussian and       |
| computing this quantity is intractable for general neural net-             | Laplace noise, respectively. Subsequently, Li et al. (2018)          |
| works. Instead, a panoply of practical solutions have been                 | used tools from information theory to prove a stronger (cid:96)2 ro- |
| proposed in the literature (Wong & Kolter, 2018; Wang et al.,              | bustness guarantee for Gaussian noise. However, all of these         |
| 2018a;b; Raghunathan et al., 2018a;b; Wong et al., 2018;                   | robustness guarantees are loose. In contrast, we prove a tight       |
| Dvijotham et al., 2018b;a; Croce et al., 2019; Gehr et al.,                | robustness guarantee in (cid:96)2 norm for randomized smoothing      |
| 2018; Mirman et al., 2018; Singh et al., 2018; Gowal et al.,               | with Gaussian noise.                                                 |
| 2018; Weng et al., 2018a; Zhang et al., 2018). Two themes                  |                                                                      |
| stand out.                                                                 |                                                                      |
| Some approaches cast veriﬁcation as an opti-                               |                                                                      |
|                                                                            | 3. Randomized smoothing                                              |
| mization problem and import tools such as relaxation and                   |                                                                      |
| duality from the optimization literature to provide conserva-              | Consider a classiﬁcation problem from Rd                             |
|                                                                            | to classes Y.                                                        |
| tive guarantees (Wong & Kolter, 2018; Wong et al., 2018;                   | As discussed above, randomized smoothing is a method for             |
| Raghunathan et al., 2018a;b; Dvijotham et al., 2018b;a).                   | constructing a new, “smoothed” classiﬁer g from an arbitrary         |
| Others step through the network layer by layer, maintaining                | base classiﬁer f . When queried at x, the smoothed classiﬁer         |
| at each layer an outer approximation of the set of activations             | g returns whichever class the base classiﬁer f is most likely        |



## Page 4

Certiﬁed Adversarial Robustness via Randomized Smoothing
to return when x is perturbed by isotropic Gaussian noise:
g(x) = arg max
c∈Y
P(f(x + ε) = c)
(1)
where ε ∼N(0, σ2I)
An equivalent deﬁnition is that g(x) returns the class c
whose pre-image {x′ ∈Rd : f(x′) = c} has the largest
probability measure under the distribution N(x, σ2I). The
noise level σ is a hyperparameter of the smoothed classiﬁer
g which controls a robustness/accuracy tradeoff; it does not
change with the input x. We leave undeﬁned the behavior
of g when the argmax is not unique.
We will ﬁrst present our robustness guarantee for the
smoothed classiﬁer g. Then, since it is not possible to
exactly evaluate the prediction of g at x or to certify the ro-
bustness of g around x, we will give Monte Carlo algorithms
for both tasks that succeed with arbitrarily high probability.
3.1. Robustness guarantee
Suppose that when the base classiﬁer f classiﬁes N(x, σ2I),
the most probable class cA is returned with probability pA,
and the “runner-up” class is returned with probability pB.
Our main result is that smoothed classiﬁer g is robust around
x within the ℓ2 radius R = σ
2 (Φ−1(pA)−Φ−1(pB)), where
Φ−1 is the inverse of the standard Gaussian CDF. This result
also holds if we replace pA with a lower bound pA and we
replace pB with an upper bound pB.
Theorem 1. Let f : Rd →Y be any deterministic or
random function, and let ε ∼N(0, σ2I). Let g be deﬁned
as in (1). Suppose cA ∈Y and pA, pB ∈[0, 1] satisfy:
P(f(x + ε) = cA) ≥pA ≥pB ≥max
c̸=cA P(f(x + ε) = c)
(2)
Then g(x + δ) = cA for all ∥δ∥2 < R, where
R = σ
2 (Φ−1(pA) −Φ−1(pB))
(3)
We now make several observations about Theorem 12
• Theorem 1 assumes nothing about f. This is crucial
since it is unclear which well-behavedness assump-
tions, if any, are satisﬁed by modern deep architectures.
• The certiﬁed radius R is large when: (1) the noise level
σ is high, (2) the probability of the top class cA is high,
and (3) the probability of each other class is low.
2After the dissemination of this work, a more general result
was published in Levine et al. (2019); Salman et al. (2019): if h :
Rd →[0, 1] is a function and ˆh is the “smoothed” version ˆh(x) =
Eε∼N (0,σ2I)[h(x + ε)], then the function x 7→Φ−1(ˆh(x)) is
1/σ-Lipschitz. Theorem 1 can be proved by applying this result to
the functions fc(x) = 1[f(x) = c] for each class c.
• The certiﬁed radius R goes to ∞as pA →1 and
pB →0. This should sound reasonable: the Gaussian
distribution is supported on all of Rd, so the only way
that f(x + ε) = cA with probability 1 is if f = cA
almost everywhere.
Both Lecuyer et al. (2019) and Li et al. (2018) proved ℓ2
robustness guarantees for the same setting as Theorem 1, but
with different, smaller expressions for the certiﬁed radius.
However, our ℓ2 robustness guarantee is tight: if (2) is all
that is known about f, then it is impossible to certify an ℓ2
ball with radius larger than R. In fact, it is impossible to
certify any superset of the ℓ2 ball with radius R:
Theorem 2. Assume pA + pB ≤1. For any perturbation
δ with ∥δ∥2 > R, there exists a base classiﬁer f consistent
with the class probabilities (2) for which g(x + δ) ̸= cA.
Theorem 2 shows that Gaussian smoothing naturally in-
duces ℓ2 robustness: if we make no assumptions on the base
classiﬁer beyond the class probabilities (2), then the set of
perturbations to which a Gaussian-smoothed classiﬁer is
provably robust is exactly an ℓ2 ball.
The complete proofs of Theorems 1 and 2 are in Appendix
A. We now sketch the proofs in the special case when there
are only two classes.
Theorem 1 (binary case). Suppose pA ∈( 1
2, 1] satisﬁes
P(f(x + ε) = cA) ≥pA. Then g(x + δ) = cA for all
∥δ∥2 < σΦ−1(pA).
Proof sketch. Fix a perturbation δ ∈Rd. To guarantee
that g(x + δ) = cA, we need to show that f classiﬁes the
translated Gaussian N(x + δ, σ2I) as cA with probability
>
1
2. However, all we know about f is that f classiﬁes
N(x, σ2I) as cA with probability ≥pA. This raises the
question: out of all possible base classiﬁers f which classify
N(x, σ2I) as cA with probability ≥pA, which one f ∗
classiﬁes N(x+δ, σ2I) as cA with the smallest probability?
One can show using an argument similar to the Neyman-
Pearson lemma (Neyman & Pearson, 1933) that this “worst-
case” f ∗is a linear classiﬁer whose decision boundary is
normal to the perturbation δ (Figure 3):
f ∗(x′) =
(
cA
if δT (x′ −x) ≤σ∥δ∥2Φ−1(pA)
cB
otherwise
(4)
This “worst-case” f ∗classiﬁes N(x + δ, σ2I) as cA with
probability Φ

Φ−1(pA) −∥δ∥2
σ

. Therefore, to ensure that
even the “worst-case” f ∗classiﬁes N(x+δ, σ2I) as cA with
probability > 1
2, we solve for those δ for which
Φ

Φ−1(pA) −∥δ∥2
σ

> 1
2
which is equivalent to the condition ∥δ∥2 < σΦ−1(pA).


**Table 4 from page 4**

| 0                                                               | 1                                                                           |
|:----------------------------------------------------------------|:----------------------------------------------------------------------------|
| Certiﬁed Adversarial Robustness via Randomized Smoothing        |                                                                             |
| to return when x is perturbed by isotropic Gaussian noise:      | • The certiﬁed radius R goes to ∞ as pA → 1 and                             |
|                                                                 | the Gaussian                                                                |
|                                                                 | pB → 0. This should sound reasonable:                                       |
| g(x) = arg max                                                  |                                                                             |
| P(f (x + ε) = c)                                                |                                                                             |
| (1)                                                             |                                                                             |
|                                                                 | distribution is supported on all of Rd, so the only way                     |
| c∈Y                                                             |                                                                             |
|                                                                 | that f (x + ε) = cA with probability 1 is if f = cA                         |
| ε ∼ N (0, σ2I)                                                  |                                                                             |
| where                                                           |                                                                             |
|                                                                 | almost everywhere.                                                          |
| An equivalent deﬁnition is                                      |                                                                             |
| that g(x)                                                       |                                                                             |
| returns                                                         |                                                                             |
| the class c                                                     |                                                                             |
|                                                                 | Both Lecuyer et al. (2019) and Li et al. (2018) proved (cid:96)2            |
| whose pre-image {x(cid:48) ∈ Rd                                 |                                                                             |
| : f (x(cid:48)) = c} has the largest                            |                                                                             |
|                                                                 | robustness guarantees for the same setting as Theorem 1, but                |
| probability measure under the distribution N (x, σ2I). The      |                                                                             |
|                                                                 | with different, smaller expressions for the certiﬁed radius.                |
| noise level σ is a hyperparameter of the smoothed classiﬁer     |                                                                             |
|                                                                 | if (2) is all                                                               |
|                                                                 | However, our (cid:96)2 robustness guarantee is tight:                       |
| g which controls a robustness/accuracy tradeoff; it does not    |                                                                             |
|                                                                 | that is known about f , then it is impossible to certify an (cid:96)2       |
| change with the input x. We leave undeﬁned the behavior         |                                                                             |
|                                                                 | ball with radius larger than R.                                             |
|                                                                 | In fact,                                                                    |
|                                                                 | it                                                                          |
|                                                                 | is impossible to                                                            |
| of g when the argmax is not unique.                             |                                                                             |
|                                                                 | certify any superset of the (cid:96)2 ball with radius R:                   |
| We will ﬁrst present our                                        | Theorem 2. Assume pA + pB ≤ 1. For any perturbation                         |
| robustness guarantee                                            |                                                                             |
| for                                                             |                                                                             |
| the                                                             |                                                                             |
| smoothed classiﬁer g.                                           | δ with (cid:107)δ(cid:107)2 > R, there exists a base classiﬁer f consistent |
| Then,                                                           |                                                                             |
| since it                                                        |                                                                             |
| is not possible to                                              |                                                                             |
| exactly evaluate the prediction of g at x or to certify the ro- | with the class probabilities (2) for which g(x + δ) (cid:54)= cA.           |
| bustness of g around x, we will give Monte Carlo algorithms     |                                                                             |
|                                                                 | Theorem 2 shows that Gaussian smoothing naturally in-                       |
| for both tasks that succeed with arbitrarily high probability.  |                                                                             |
|                                                                 | if we make no assumptions on the base                                       |
|                                                                 | duces (cid:96)2 robustness:                                                 |
|                                                                 | classiﬁer beyond the class probabilities (2), then the set of               |
| 3.1. Robustness guarantee                                       |                                                                             |
|                                                                 | perturbations to which a Gaussian-smoothed classiﬁer is                     |
| Suppose that when the base classiﬁer f classiﬁes N (x, σ2I),    |                                                                             |
|                                                                 | provably robust is exactly an (cid:96)2 ball.                               |
| the most probable class cA is returned with probability pA,     |                                                                             |
|                                                                 | The complete proofs of Theorems 1 and 2 are in Appendix                     |
| and the “runner-up” class is returned with probability pB.      |                                                                             |
|                                                                 | A. We now sketch the proofs in the special case when there                  |
| Our main result is that smoothed classiﬁer g is robust around   |                                                                             |
|                                                                 | are only two classes.                                                       |
| x within the (cid:96)2 radius R = σ                             |                                                                             |
| 2 (Φ−1(pA)−Φ−1(pB)), where                                      |                                                                             |
| Φ−1 is the inverse of the standard Gaussian CDF. This result    | Suppose pA ∈ ( 1                                                            |
|                                                                 | Theorem 1 (binary case).                                                    |
|                                                                 | 2 , 1] satisﬁes                                                             |
| also holds if we replace pA with a lower bound pA and we        | P(f (x + ε) = cA) ≥ pA.                                                     |
|                                                                 | Then g(x + δ) = cA for all                                                  |
| replace pB with an upper bound pB.                              |                                                                             |
|                                                                 | (cid:107)δ(cid:107)2 < σΦ−1(pA).                                            |
| Let f                                                           |                                                                             |
| : Rd → Y be any deterministic or                                |                                                                             |
| Theorem 1.                                                      |                                                                             |
| random function, and let ε ∼ N (0, σ2I). Let g be deﬁned        |                                                                             |
|                                                                 | Proof sketch. Fix a perturbation δ ∈ Rd.                                    |
|                                                                 | To guarantee                                                                |
| as in (1). Suppose cA ∈ Y and pA, pB ∈ [0, 1] satisfy:          |                                                                             |
|                                                                 | that g(x + δ) = cA, we need to show that f classiﬁes the                    |
|                                                                 | translated Gaussian N (x + δ, σ2I) as cA with probability                   |
| P(f (x + ε) = c)                                                |                                                                             |
| (2)                                                             |                                                                             |
| P(f (x + ε) = cA) ≥ pA ≥ pB ≥ max                               |                                                                             |
| c                                                               | > 1                                                                         |
| (cid:54)=cA                                                     | 2 . However, all we know about f is that f classiﬁes                        |
|                                                                 | N (x, σ2I) as cA with probability ≥ pA. This raises the                     |
| Then g(x + δ) = cA for all (cid:107)δ(cid:107)2 < R, where      |                                                                             |
|                                                                 | question: out of all possible base classiﬁers f which classify              |
|                                                                 | N (x, σ2I) as cA with probability ≥ pA, which one f ∗                       |
| σ 2                                                             |                                                                             |
| R =                                                             |                                                                             |
| (3)                                                             |                                                                             |
| (Φ−1(pA) − Φ−1(pB))                                             |                                                                             |
|                                                                 | classiﬁes N (x+δ, σ2I) as cA with the smallest probability?                 |
|                                                                 | One can show using an argument similar to the Neyman-                       |
|                                                                 | Pearson lemma (Neyman & Pearson, 1933) that this “worst-                    |
| We now make several observations about Theorem 12               |                                                                             |
|                                                                 | case” f ∗ is a linear classiﬁer whose decision boundary is                  |
|                                                                 | normal to the perturbation δ (Figure 3):                                    |
| • Theorem 1 assumes nothing about f . This is crucial           |                                                                             |
|                                                                 | (cid:40)                                                                    |
| since it                                                        |                                                                             |
| is unclear which well-behavedness assump-                       |                                                                             |
|                                                                 | cA                                                                          |
|                                                                 | if δT (x(cid:48) − x) ≤ σ(cid:107)δ(cid:107)2Φ−1(pA)                        |
|                                                                 | f ∗(x(cid:48)) =                                                            |
|                                                                 | (4)                                                                         |
| tions, if any, are satisﬁed by modern deep architectures.       |                                                                             |
|                                                                 | otherwise                                                                   |
|                                                                 | cB                                                                          |
| • The certiﬁed radius R is large when: (1) the noise level      |                                                                             |
|                                                                 | This “worst-case” f ∗ classiﬁes N (x + δ, σ2I) as cA with                   |
| σ is high, (2) the probability of the top class cA is high,     | (cid:16)                                                                    |
|                                                                 | (cid:17)                                                                    |
|                                                                 | probability Φ                                                               |
|                                                                 | . Therefore, to ensure that                                                 |
|                                                                 | Φ−1(pA) − (cid:107)δ(cid:107)2                                              |
| and (3) the probability of each other class is low.             | σ                                                                           |
|                                                                 | even the “worst-case” f ∗ classiﬁes N (x+δ, σ2I) as cA with                 |
| 2After the dissemination of this work, a more general result    |                                                                             |
|                                                                 | probability > 1                                                             |
|                                                                 | 2 , we solve for those δ for which                                          |
| was published in Levine et al. (2019); Salman et al. (2019):    |                                                                             |
| if h :                                                          |                                                                             |



## Page 5

Certiﬁed Adversarial Robustness via Randomized Smoothing
x + δ
x
x + δ
x
Figure 3. Illustration of f ∗in two dimensions. The concentric
circles are the density contours of N(x, σ2I) and N(x + δ, σ2I).
Out of all base classiﬁers f which classify N(x, σ2I) as cA (blue)
with probability ≥pA, such as both classiﬁers depicted above,
the “worst-case” f ∗— the one which classiﬁes N(x + δ, σ2I) as
cA with minimal probability — is depicted on the right: a linear
classiﬁer with decision boundary normal to the perturbation δ.
Theorem 2 is a simple consequence: for any δ with ∥δ∥2 >
R, the base classiﬁer f ∗deﬁned in (4) is consistent with (2);
yet if f ∗is the base classiﬁer, then g(x + δ) = cB.
Figure 5 (left) plots our ℓ2 robustness guarantee against
the guarantees derived in prior work. Observe that our
R is much larger than that of Lecuyer et al. (2019) and
moderately larger than that of Li et al. (2018). Appendix I
derives the other two guarantees using this paper’s notation.
Linear base classiﬁer
A two-class linear classiﬁer
f(x) = sign(wT x + b) is already certiﬁable: the dis-
tance from any input x to the decision boundary is |wT x +
b|/∥w∥2, and no perturbation δ with ℓ2 norm less than this
distance can possibly change f’s prediction. In Appendix B
we show that if f is linear, then the smoothed classiﬁer g is
identical to the base classiﬁer f. Moreover, we show that our
bound (3) will certify the true robust radius |wT x + b|/∥w∥,
rather than a smaller, overconservative radius. Therefore,
when f is linear, there always exists a perturbation δ just
beyond the certiﬁed radius which changes g’s prediction.
Noise level can scale with image resolution
Since our
expression (3) for the certiﬁed radius does not depend ex-
plicitly on the data dimension d, one might worry that ran-
domized smoothing is less effective for images of higher
resolution — certifying a ﬁxed ℓ2 radius is “less impressive”
for, say, a 224 × 224 image than for a 56 × 56 image. How-
ever, as illustrated by Figure 4, images in higher resolution
can tolerate higher levels σ of isotropic Gaussian noise be-
fore their class-distinguishing content gets destroyed. As
a consequence, in high resolution, smoothing can be per-
formed with a larger σ, leading to larger certiﬁed radii. See
Appendix G for a more rigorous version of this argument.
3.2. Practical algorithms
We now present practical Monte Carlo algorithms for eval-
uating g(x) and certifying the robustness of g around x.
More details can be found in Appendix C.
3.2.1. PREDICTION
Evaluating the smoothed classiﬁer’s prediction g(x) re-
quires identifying the class cA with maximal weight in the
categorical distribution f(x + ε). The procedure described
in pseudocode as PREDICT draws n samples of f(x + ε)
by running n noise-corrupted copies of x through the base
classiﬁer. Let ˆcA be the class which appeared the largest
number of times. If ˆcA appeared much more often than any
other class, then PREDICT returns ˆcA. Otherwise, it abstains
from making a prediction. We use the hypothesis test from
Hung & Fithian (2019) to calibrate the abstention threshold
so as to bound by α the probability of returning an incorrect
answer. PREDICT satisﬁes the following guarantee:
Proposition 1. With probability at least 1 −α over the
randomness in PREDICT, PREDICT will either abstain or
return g(x). (Equivalently: the probability that PREDICT
returns a class other than g(x) is at most α.)
The function SAMPLEUNDERNOISE(f, x, num, σ) in the
pseudocode draws num samples of noise, ε1 . . . εnum ∼
N(0, σ2I), runs each x + εi through the base classiﬁer f,
and returns a vector of class counts. BINOMPVALUE(nA,
nA + nB, p) returns the p-value of the two-sided hypothesis
test that nA ∼Binomial(nA + nB, p).
Even if the true smoothed classiﬁer g is robust at radius R,
PREDICT will be vulnerable in a certain sense to adversarial
perturbations with ℓ2 norm slightly less than R. By engi-
neering a perturbation δ for which f(x + δ + ε) puts mass
just over 1
2 on class cA and mass just under 1
2 on class cB,
an adversary can force PREDICT to abstain at a high rate. If
this scenario is of concern, a variant of Theorem 1 could be
proved to certify a radius in which P(f(x + δ + ε) = cA) is
larger by some margin than maxc̸=cA P(f(x + δ + ε) = c).
3.2.2. CERTIFICATION
Evaluating and certifying the robustness of g around an
input x requires not only identifying the class cA with maxi-
mal weight in f(x + ε), but also estimating a lower bound
Figure 4. Left to right: clean 56 x 56 image, clean 224 x 224 image,
noisy 56 x 56 image (σ = 0.5), noisy 224 x 224 image (σ = 0.5).


**Table 5 from page 5**

| 0                                                                                 | 1                                                                    |
|:----------------------------------------------------------------------------------|:---------------------------------------------------------------------|
| Certiﬁed Adversarial Robustness via Randomized Smoothing                          |                                                                      |
|                                                                                   | 3.2. Practical algorithms                                            |
|                                                                                   | We now present practical Monte Carlo algorithms for eval-            |
| x + δ                                                                             | uating g(x) and certifying the robustness of g around x.             |
| x + δ                                                                             |                                                                      |
|                                                                                   | More details can be found in Appendix C.                             |
| x                                                                                 | 3.2.1. PREDICTION                                                    |
| x                                                                                 |                                                                      |
|                                                                                   | Evaluating the smoothed classiﬁer’s prediction g(x)                  |
|                                                                                   | re-                                                                  |
|                                                                                   | quires identifying the class cA with maximal weight in the           |
|                                                                                   | categorical distribution f (x + ε). The procedure described          |
|                                                                                   | in pseudocode as PREDICT draws n samples of f (x + ε)                |
| Figure 3. Illustration of f ∗ in two dimensions.                                  |                                                                      |
| The concentric                                                                    |                                                                      |
|                                                                                   | by running n noise-corrupted copies of x through the base            |
| circles are the density contours of N (x, σ2I) and N (x + δ, σ2I).                |                                                                      |
|                                                                                   | classiﬁer. Let ˆcA be the class which appeared the largest           |
| Out of all base classiﬁers f which classify N (x, σ2I) as cA (blue)               |                                                                      |
|                                                                                   | number of times. If ˆcA appeared much more often than any            |
| with probability ≥ pA, such as both classiﬁers depicted above,                    |                                                                      |
|                                                                                   | other class, then PREDICT returns ˆcA. Otherwise, it abstains        |
| the “worst-case” f ∗ — the one which classiﬁes N (x + δ, σ2I) as                  |                                                                      |
|                                                                                   | from making a prediction. We use the hypothesis test from            |
| cA with minimal probability — is depicted on the right: a linear                  |                                                                      |
|                                                                                   | Hung & Fithian (2019) to calibrate the abstention threshold          |
| classiﬁer with decision boundary normal to the perturbation δ.                    |                                                                      |
|                                                                                   | so as to bound by α the probability of returning an incorrect        |
|                                                                                   | answer. PREDICT satisﬁes the following guarantee:                    |
|                                                                                   | least 1 − α over the                                                 |
|                                                                                   | Proposition 1. With probability at                                   |
| Theorem 2 is a simple consequence: for any δ with (cid:107)δ(cid:107)2 >          |                                                                      |
| R, the base classiﬁer f ∗ deﬁned in (4) is consistent with (2);                   | randomness in PREDICT, PREDICT will either abstain or                |
|                                                                                   | (Equivalently:                                                       |
|                                                                                   | return g(x).                                                         |
|                                                                                   | the probability that PREDICT                                         |
| yet if f ∗ is the base classiﬁer, then g(x + δ) = cB.                             |                                                                      |
|                                                                                   | returns a class other than g(x) is at most α.)                       |
| robustness guarantee against                                                      |                                                                      |
| Figure 5 (left) plots our (cid:96)2                                               |                                                                      |
| the guarantees derived in prior work.                                             |                                                                      |
| Observe that our                                                                  |                                                                      |
|                                                                                   | The function SAMPLEUNDERNOISE(f , x, num, σ) in the                  |
| R is much larger                                                                  |                                                                      |
| than that of Lecuyer et al.                                                       |                                                                      |
| (2019) and                                                                        |                                                                      |
|                                                                                   | pseudocode draws num samples of noise, ε1 . . . εnum ∼               |
| moderately larger than that of Li et al. (2018). Appendix I                       |                                                                      |
|                                                                                   | through the base classiﬁer f ,                                       |
|                                                                                   | N (0, σ2I), runs each x + εi                                         |
| derives the other two guarantees using this paper’s notation.                     |                                                                      |
|                                                                                   | and returns a vector of class counts. BINOMPVALUE(nA,                |
|                                                                                   | nA + nB, p) returns the p-value of the two-sided hypothesis          |
|                                                                                   | test that nA ∼ Binomial(nA + nB, p).                                 |
| Linear                                                                            |                                                                      |
| base                                                                              |                                                                      |
| classiﬁer                                                                         |                                                                      |
| A two-class                                                                       |                                                                      |
| linear                                                                            |                                                                      |
| classiﬁer                                                                         |                                                                      |
| f (x) = sign(wT x + b)                                                            | Even if the true smoothed classiﬁer g is robust at radius R,         |
| is already certiﬁable:                                                            |                                                                      |
| the dis-                                                                          |                                                                      |
| tance from any input x to the decision boundary is |wT x +                        | PREDICT will be vulnerable in a certain sense to adversarial         |
| b|/(cid:107)w(cid:107)2, and no perturbation δ with (cid:96)2 norm less than this | perturbations with (cid:96)2 norm slightly less than R. By engi-     |
| distance can possibly change f ’s prediction. In Appendix B                       | neering a perturbation δ for which f (x + δ + ε) puts mass           |
| we show that if f is linear, then the smoothed classiﬁer g is                     | just over 1                                                          |
|                                                                                   | 2 on class cA and mass just under 1                                  |
|                                                                                   | 2 on class cB,                                                       |
| identical to the base classiﬁer f . Moreover, we show that our                    | an adversary can force PREDICT to abstain at a high rate. If         |
| bound (3) will certify the true robust radius |wT x + b|/(cid:107)w(cid:107),     | this scenario is of concern, a variant of Theorem 1 could be         |
| rather than a smaller, overconservative radius. Therefore,                        |                                                                      |
|                                                                                   | proved to certify a radius in which P(f (x + δ + ε) = cA) is         |
| when f is linear,                                                                 | P(f (x + δ + ε) = c).                                                |
| there always exists a perturbation δ just                                         |                                                                      |
|                                                                                   | larger by some margin than maxc(cid:54)=cA                           |
| beyond the certiﬁed radius which changes g’s prediction.                          |                                                                      |
|                                                                                   | 3.2.2. CERTIFICATION                                                 |
| Noise level can scale with image resolution                                       | Evaluating and certifying the robustness of g around an              |
| Since our                                                                         |                                                                      |
| expression (3) for the certiﬁed radius does not depend ex-                        | input x requires not only identifying the class cA with maxi-        |
| plicitly on the data dimension d, one might worry that ran-                       | mal weight in f (x + ε), but also estimating a lower bound           |
| domized smoothing is less effective for images of higher                          |                                                                      |
| resolution — certifying a ﬁxed (cid:96)2 radius is “less impressive”              |                                                                      |
| for, say, a 224 × 224 image than for a 56 × 56 image. How-                        |                                                                      |
| ever, as illustrated by Figure 4, images in higher resolution                     |                                                                      |
| can tolerate higher levels σ of isotropic Gaussian noise be-                      |                                                                      |
| fore their class-distinguishing content gets destroyed. As                        |                                                                      |
| a consequence,                                                                    |                                                                      |
| in high resolution, smoothing can be per-                                         |                                                                      |
| formed with a larger σ, leading to larger certiﬁed radii. See                     | Figure 4. Left to right: clean 56 x 56 image, clean 224 x 224 image, |
|                                                                                   | noisy 56 x 56 image (σ = 0.5), noisy 224 x 224 image (σ = 0.5).      |
| Appendix G for a more rigorous version of this argument.                          |                                                                      |



## Page 6

Certiﬁed Adversarial Robustness via Randomized Smoothing
Pseudocode for certiﬁcation and prediction
# evaluate g at x
function PREDICT(f, σ, x, n, α)
counts ←SAMPLEUNDERNOISE(f, x, n, σ)
ˆcA, ˆcB ←top two indices in counts
nA, nB ←counts[ˆcA], counts[ˆcB]
if BINOMPVALUE(nA, nA + nB, 0.5) ≤α return ˆcA
else return ABSTAIN
# certify the robustness of g around x
function CERTIFY(f, σ, x, n0, n, α)
counts0 ←SAMPLEUNDERNOISE(f, x, n0, σ)
ˆcA ←top index in counts0
counts ←SAMPLEUNDERNOISE(f, x, n, σ)
pA ←LOWERCONFBOUND(counts[ˆcA], n, 1 −α)
if pA > 1
2 return prediction ˆcA and radius σ Φ−1(pA)
else return ABSTAIN
pA on the probability that f(x + ε) = cA and an upper
bound pB on the probability that f(x + ε) equals any other
class. Doing all three of these at the same time in a sta-
tistically correct manner requires some care. One simple
solution is presented in pseudocode as CERTIFY: ﬁrst, use
a small number of samples from f(x + ε) to take a guess
at cA; then use a larger number of samples to estimate pA;
then simply take pB = 1 −pA.
Proposition 2. With probability at least 1 −α over the
randomness in CERTIFY, if CERTIFY returns a class ˆcA
and a radius R (i.e. does not abstain), then g predicts ˆcA
within radius R around x: g(x + δ) = ˆcA ∀∥δ∥2 < R.
The function LOWERCONFBOUND(k, n, 1−α) in the pseu-
docode returns a one-sided (1 −α) lower conﬁdence in-
terval for the Binomial parameter p given a sample k ∼
Binomial(n, p).
Certifying large radii requires many samples
Recall
from Theorem 1 that R approaches ∞as pA approaches 1.
Unfortunately, it turns out that pA approaches 1 so slowly
with n that R also approaches ∞very slowly with n. Con-
sider the most favorable situation: f(x) = cA everywhere.
This means that g is robust at radius ∞. But after observing
n samples of f(x + ε) which all equal cA, the tightest (to
our knowledge) lower bound would say that with probabil-
ity least 1 −α, pA ≥α(1/n). Plugging pA = α(1/n) and
pB = 1 −pA into (3) yields an expression for the certiﬁed
radius as a function of n: R = σ Φ−1(α1/n). Figure 5
(right) plots this function for α = 0.001, σ = 1. Observe
that certifying a radius of 4σ with 99.9% conﬁdence would
require ≈105 samples.
0.5
0.6
0.7
0.8
0.9
1.0
pA
0
1
2
3
radius
ours
(Lecuyer et al, 2018)
(Li et al, 2018)
10
2
10
4
10
6
number of samples
0
1
2
3
4
5
radius
Figure 5. Left: Certiﬁed radius R as a function of pA (with pB =
1 −pA and σ = 1) under all three randomized smoothing bounds.
Right: A plot of R = σ Φ−1(α1/n) for α = 0.001 and σ = 1.
The radius we can certify with high probability grows slowly with
the number of samples, even in the best case where f(x) = cA
everywhere.
3.3. Training the base classiﬁer
Theorem 1 holds regardless of how the base classiﬁer f is
trained. However, in order for g to classify the labeled ex-
ample (x, c) correctly and robustly, f needs to consistently
classify N(x, σ2I) as c. In high dimension, the Gaussian
distribution N(x, σ2I) places almost no mass near its mode
x. As a consequence, when σ is moderately high, the distri-
bution of natural images has virtually disjoint support from
the distribution of natural images corrupted by N(0, σ2I);
see Figure 2 for a visual demonstration. Therefore, if the
base classiﬁer f is trained via standard supervised learning
on the data distribution, it will see no noisy images during
training, and hence will not necessarily learn to classify
N(x, σ2I) with x’s true label. Therefore, in this paper we
follow Lecuyer et al. (2019) and train the base classiﬁer
with Gaussian data augmentation at variance σ2. A justiﬁca-
tion for this procedure is provided in Appendix F. However,
we suspect that there may be room to improve upon this
training scheme, perhaps by training the base classiﬁer so
as to maximize the smoothed classiﬁer’s certiﬁed accuracy
at some tunable radius r.
4. Experiments
In adversarially robust classiﬁcation, one metric of interest
is the certiﬁed test set accuracy at radius r, deﬁned as the
fraction of the test set which g classiﬁes correctly with a pre-
diction that is certiﬁably robust within an ℓ2 ball of radius r.
However, if g is a randomized smoothing classiﬁer, comput-
ing this quantity exactly is not possible, so we instead report
the approximate certiﬁed test set accuracy, deﬁned as the
fraction of the test set which CERTIFY classiﬁes correctly
(without abstaining) and certiﬁes robust with a radius R ≥r.
Appendix D shows how to convert the approximate certiﬁed
accuracy into a lower bound on the true certiﬁed accuracy
that holds with high probability over the randomness in
CERTIFY. However Appendix H.2 demonstrates that when


**Table 6 from page 6**

| 0                                                            | 1                                                                       |
|:-------------------------------------------------------------|:------------------------------------------------------------------------|
| Pseudocode for certiﬁcation and prediction                   | 5                                                                       |
|                                                              | 3                                                                       |
|                                                              | ours                                                                    |
|                                                              | (Lecuyer et al, 2018)                                                   |
|                                                              | 4                                                                       |
| # evaluate g at x                                            | (Li et al, 2018)                                                        |
|                                                              | 2                                                                       |
|                                                              | 3                                                                       |
| function PREDICT(f , σ, x, n, α)                             | radius                                                                  |
|                                                              | radius                                                                  |
|                                                              | 2                                                                       |
| counts ← SAMPLEUNDERNOISE(f , x, n, σ)                       | 1                                                                       |
|                                                              | 1                                                                       |
| cA, ˆcB ← top two indices in counts                          |                                                                         |
|                                                              | 0                                                                       |
| nA, nB ← counts[ˆcA], counts[ˆcB]                            | 4                                                                       |
|                                                              | 2                                                                       |
|                                                              | 6                                                                       |
|                                                              | 0                                                                       |
|                                                              | 10                                                                      |
|                                                              | 10                                                                      |
|                                                              | 10                                                                      |
|                                                              | 0.5                                                                     |
|                                                              | 0.6                                                                     |
|                                                              | 0.7                                                                     |
|                                                              | 0.8                                                                     |
|                                                              | 0.9                                                                     |
|                                                              | 1.0                                                                     |
|                                                              | number of samples                                                       |
|                                                              | pA                                                                      |
| if BINOMPVALUE(nA, nA + nB, 0.5) ≤ α return ˆcA              |                                                                         |
| else return ABSTAIN                                          |                                                                         |
|                                                              | Figure 5. Left: Certiﬁed radius R as a function of pA (with pB =        |
|                                                              | 1 − pA and σ = 1) under all three randomized smoothing bounds.          |
| # certify the robustness of g around x                       |                                                                         |
|                                                              | Right: A plot of R = σ Φ−1(α1/n) for α = 0.001 and σ = 1.               |
| function CERTIFY(f , σ, x, n0, n, α)                         |                                                                         |
|                                                              | The radius we can certify with high probability grows slowly with       |
| counts0 ← SAMPLEUNDERNOISE(f, x, n0, σ)                      |                                                                         |
|                                                              | the number of samples, even in the best case where f (x) = cA           |
| cA ← top index in counts0                                    |                                                                         |
|                                                              | everywhere.                                                             |
| counts ← SAMPLEUNDERNOISE(f, x, n, σ)                        |                                                                         |
| pA ← LOWERCONFBOUND(counts[ˆcA], n, 1 − α)                   |                                                                         |
| if pA > 1                                                    |                                                                         |
| 2 return prediction ˆcA and radius σ Φ−1(pA)                 |                                                                         |
| else return ABSTAIN                                          |                                                                         |
|                                                              | 3.3. Training the base classiﬁer                                        |
|                                                              | Theorem 1 holds regardless of how the base classiﬁer f is               |
|                                                              | trained. However, in order for g to classify the labeled ex-            |
| pA on the probability that f (x + ε) = cA and an upper       |                                                                         |
|                                                              | ample (x, c) correctly and robustly, f needs to consistently            |
| bound pB on the probability that f (x + ε) equals any other  |                                                                         |
|                                                              | classify N (x, σ2I) as c.                                               |
|                                                              | In high dimension,                                                      |
|                                                              | the Gaussian                                                            |
| class. Doing all                                             |                                                                         |
| three of these at                                            |                                                                         |
| the same time in a sta-                                      |                                                                         |
|                                                              | distribution N (x, σ2I) places almost no mass near its mode             |
| tistically correct manner requires some care. One simple     |                                                                         |
|                                                              | x. As a consequence, when σ is moderately high, the distri-             |
| solution is presented in pseudocode as CERTIFY: ﬁrst, use    |                                                                         |
|                                                              | bution of natural images has virtually disjoint support from            |
| a small number of samples from f (x + ε) to take a guess     |                                                                         |
|                                                              | the distribution of natural images corrupted by N (0, σ2I);             |
| at cA; then use a larger number of samples to estimate pA;   |                                                                         |
|                                                              | see Figure 2 for a visual demonstration. Therefore,                     |
|                                                              | if the                                                                  |
| then simply take pB = 1 − pA.                                |                                                                         |
|                                                              | base classiﬁer f is trained via standard supervised learning            |
|                                                              | on the data distribution, it will see no noisy images during            |
| least 1 − α over the                                         |                                                                         |
| Proposition 2. With probability at                           |                                                                         |
|                                                              | training, and hence will not necessarily learn to classify              |
| randomness in CERTIFY,                                       |                                                                         |
| if CERTIFY returns a class ˆcA                               |                                                                         |
|                                                              | N (x, σ2I) with x’s true label. Therefore, in this paper we             |
| and a radius R (i.e. does not abstain), then g predicts ˆcA  |                                                                         |
|                                                              | follow Lecuyer et al.                                                   |
|                                                              | (2019) and train the base classiﬁer                                     |
| within radius R around x: g(x + δ) = ˆcA                     |                                                                         |
| ∀ (cid:107)δ(cid:107)2 < R.                                  |                                                                         |
|                                                              | with Gaussian data augmentation at variance σ2. A justiﬁca-             |
|                                                              | tion for this procedure is provided in Appendix F. However,             |
| The function LOWERCONFBOUND(k, n, 1 − α) in the pseu-        |                                                                         |
|                                                              | we suspect                                                              |
|                                                              | that                                                                    |
|                                                              | there may be room to improve upon this                                  |
| docode returns a one-sided (1 − α) lower conﬁdence in-       |                                                                         |
|                                                              | training scheme, perhaps by training the base classiﬁer so              |
| terval for the Binomial parameter p given a sample k ∼       |                                                                         |
|                                                              | as to maximize the smoothed classiﬁer’s certiﬁed accuracy               |
| Binomial(n, p).                                              |                                                                         |
|                                                              | at some tunable radius r.                                               |
|                                                              | 4. Experiments                                                          |
| Certifying large radii requires many samples                 |                                                                         |
| Recall                                                       |                                                                         |
| from Theorem 1 that R approaches ∞ as pA approaches 1.       | In adversarially robust classiﬁcation, one metric of interest           |
| Unfortunately, it turns out that pA approaches 1 so slowly   | is the certiﬁed test set accuracy at radius r, deﬁned as the            |
| with n that R also approaches ∞ very slowly with n. Con-     | fraction of the test set which g classiﬁes correctly with a pre-        |
| sider the most favorable situation: f (x) = cA everywhere.   | diction that is certiﬁably robust within an (cid:96)2 ball of radius r. |
| This means that g is robust at radius ∞. But after observing | However, if g is a randomized smoothing classiﬁer, comput-              |
| n samples of f (x + ε) which all equal cA, the tightest (to  | ing this quantity exactly is not possible, so we instead report         |
| our knowledge) lower bound would say that with probabil-     | the approximate certiﬁed test set accuracy, deﬁned as the               |
| ity least 1 − α, pA ≥ α(1/n). Plugging pA = α(1/n) and       | fraction of the test set which CERTIFY classiﬁes correctly              |
| pB = 1 − pA into (3) yields an expression for the certiﬁed   | (without abstaining) and certiﬁes robust with a radius R ≥ r.           |
| radius as a function of n: R = σ Φ−1(α1/n).                  | Appendix D shows how to convert the approximate certiﬁed                |
| Figure 5                                                     |                                                                         |
| (right) plots this function for α = 0.001, σ = 1. Observe    | accuracy into a lower bound on the true certiﬁed accuracy               |
| that certifying a radius of 4σ with 99.9% conﬁdence would    | that holds with high probability over                                   |
|                                                              | the randomness in                                                       |
| require ≈ 105 samples.                                       | CERTIFY. However Appendix H.2 demonstrates that when                    |



## Page 7

Certiﬁed Adversarial Robustness via Randomized Smoothing
0.0
0.2
0.4
0.6
0.8
1.0
1.2
1.4
radius
0.0
0.2
0.4
0.6
0.8
1.0
certified accuracy
= 0.12
= 0.25
= 0.50
= 1.00
undefended
0.0
0.5
1.0
1.5
2.0
2.5
3.0
3.5
4.0
radius
0.0
0.2
0.4
0.6
0.8
1.0
certified accuracy
= 0.25
= 0.50
= 1.00
undefended
Figure 6. Approximate certiﬁed accuracy attained by randomized
smoothing on CIFAR-10 (top) and ImageNet (bottom). The hyper-
parameter σ controls a robustness/accuracy tradeoff. The dashed
black line is an upper bound on the empirical robust accuracy of
an undefended classiﬁer with the base classiﬁer’s architecture.
α is small, the difference between these two quantities is
negligible. Therefore, in our experiments we omit the step
for simplicity and report approximate certiﬁed accuracies.
In all experiments, unless otherwise stated, we ran CERTIFY
with α = 0.001, so there was at most a 0.1% chance that
CERTIFY returned a radius in which g was not truly robust.
Unless otherwise stated, when running CERTIFY we used
n0 = 100 Monte Carlo samples for selection and n =
100,000 samples for estimation.
In the ﬁgures above that plot certiﬁed accuracy as a function
of radius r, the certiﬁed accuracy always decreases gradually
with r until reaching some point where it plummets to zero.
This drop occurs because for each noise level σ and number
of samples n, there is a hard upper limit to the radius we can
certify with high probability, achieved when all n samples
are classiﬁed by f as the same class.
ImageNet and CIFAR-10 results
We applied random-
ized smoothing to CIFAR-10 (Krizhevsky, 2009) and Im-
ageNet (Deng et al., 2009). On each dataset we trained
several smoothed classiﬁers, each with a different σ. On
CIFAR-10 our base classiﬁer was a 110-layer residual
network; certifying each example took 15 seconds on an
NVIDIA RTX 2080 Ti. On ImageNet our base classiﬁer
was a ResNet-50; certifying each example took 110 seconds.
We also trained a neural network with the base classiﬁer’s
architecture on clean data, and subjected it to a DeepFool ℓ2
adversarial attack (Moosavi-Dezfooli et al., 2016), in order
0.0
0.5
1.0
1.5
2.0
2.5
3.0
radius
0.0
0.2
0.4
0.6
0.8
1.0
certified accuracy
smoothing, large network 
smoothing, small network
(Wong et al, 2018) 1
(Wong et al, 2018) 2
(Wong et al, 2018) 3
Figure 7. Comparison betwen randomized smoothing and Wong
et al. (2018). Each green line is a small resnet classiﬁer trained and
certiﬁed using the method of Wong et al. (2018) with a different
setting of its hyperparameter ϵ. The purple line is our method
using the same small resnet architecture as the base classiﬁer;
the blue line is our method with a larger neural network as the
base classiﬁer. Wong et al. (2018) gives deterministic robustness
guarantees, whereas smoothing gives high-probability guaranatees;
therefore, we plot here the certiﬁed accuracy of Wong et al. (2018)
against the “approximate” certiﬁed accuracy of smoothing.
to obtain an empirical upper bound on its robust accuracy.
We certiﬁed the full CIFAR-10 test set and a subsample of
500 examples from the ImageNet test set.
Figure 6 plots the certiﬁed accuracy attained by smoothing
with each σ. The dashed black line is the empirical upper
bound on the robust accuracy of the base classiﬁer architec-
ture; observe that smoothing improves substantially upon
the robustness of the undefended base classiﬁer architecture.
We see that σ controls a robustness/accuracy tradeoff. When
σ is low, small radii can be certiﬁed with high accuracy, but
large radii cannot be certiﬁed. When σ is high, larger radii
can be certiﬁed, but smaller radii are certiﬁed at a lower ac-
curacy. This observation echoes the ﬁnding in Tsipras et al.
(2019) that adversarially trained networks with higher ro-
bust accuracy tend to have lower standard accuracy. Tables
of these results are in Appendix E.
Figure 8 (left) plots the certiﬁed accuracy obtained using our
Theorem 1 guarantee alongside the certiﬁed accuracy ob-
tained using the analogous bounds of Lecuyer et al. (2019)
and Li et al. (2018). Since our expression for the certiﬁed
radius R is greater (and, in fact, tight), our bound delivers
higher certiﬁed accuracies. Figure 8 (middle) projects how
the certiﬁed accuracy would have changed had CERTIFY
used more or fewer samples n (under the assumption that the
relative class proportions in counts would have remained
constant). Finally, Figure 8 (right) plots the certiﬁed accu-
racy as the conﬁdence parameter α is varied. Observe that
the certiﬁed accuracy is not very sensitive to α.


**Table 7 from page 7**

| 0                                                                    | 1                                                                      |
|:---------------------------------------------------------------------|:-----------------------------------------------------------------------|
| Certiﬁed Adversarial Robustness via Randomized Smoothing             |                                                                        |
| 1.0                                                                  | 1.0                                                                    |
| = 0.12                                                               | smoothing, large network                                               |
| 0.8                                                                  | smoothing, small network                                               |
| = 0.25                                                               |                                                                        |
|                                                                      | 0.8                                                                    |
|                                                                      | (Wong et al, 2018) 1                                                   |
| = 0.50                                                               |                                                                        |
| 0.6                                                                  |                                                                        |
|                                                                      | (Wong et al, 2018) 2                                                   |
| certified accuracy                                                   |                                                                        |
| = 1.00                                                               |                                                                        |
|                                                                      | (Wong et al, 2018) 3                                                   |
|                                                                      | 0.6                                                                    |
| undefended                                                           | certified accuracy                                                     |
| 0.4                                                                  |                                                                        |
| 0.2                                                                  | 0.4                                                                    |
| 0.0                                                                  |                                                                        |
| 0.0                                                                  |                                                                        |
| 0.2                                                                  |                                                                        |
| 0.4                                                                  |                                                                        |
| 0.6                                                                  |                                                                        |
| 0.8                                                                  |                                                                        |
| 1.0                                                                  |                                                                        |
| 1.2                                                                  |                                                                        |
| 1.4                                                                  |                                                                        |
|                                                                      | 0.2                                                                    |
| radius                                                               |                                                                        |
| 1.0                                                                  | 0.0                                                                    |
|                                                                      | 0.0                                                                    |
|                                                                      | 0.5                                                                    |
|                                                                      | 1.0                                                                    |
|                                                                      | 1.5                                                                    |
|                                                                      | 2.0                                                                    |
|                                                                      | 2.5                                                                    |
|                                                                      | 3.0                                                                    |
| = 0.25                                                               |                                                                        |
| 0.8                                                                  | radius                                                                 |
| = 0.50                                                               |                                                                        |
| = 1.00                                                               |                                                                        |
| 0.6                                                                  |                                                                        |
| certified accuracy                                                   |                                                                        |
| undefended                                                           |                                                                        |
| 0.4                                                                  | Figure 7. Comparison betwen randomized smoothing and Wong              |
|                                                                      | et al. (2018). Each green line is a small resnet classiﬁer trained and |
| 0.2                                                                  |                                                                        |
|                                                                      | certiﬁed using the method of Wong et al. (2018) with a different       |
| 0.0                                                                  | setting of                                                             |
|                                                                      | its hyperparameter (cid:15). The purple line is our method             |
| 0.0                                                                  |                                                                        |
| 0.5                                                                  |                                                                        |
| 1.0                                                                  |                                                                        |
| 1.5                                                                  |                                                                        |
| 2.0                                                                  |                                                                        |
| 2.5                                                                  |                                                                        |
| 3.0                                                                  |                                                                        |
| 3.5                                                                  |                                                                        |
| 4.0                                                                  |                                                                        |
| radius                                                               | using the same small                                                   |
|                                                                      | resnet architecture as the base classiﬁer;                             |
|                                                                      | the blue line is our method with a larger neural network as the        |
|                                                                      | base classiﬁer. Wong et al. (2018) gives deterministic robustness      |
| Figure 6. Approximate certiﬁed accuracy attained by randomized       |                                                                        |
|                                                                      | guarantees, whereas smoothing gives high-probability guaranatees;      |
| smoothing on CIFAR-10 (top) and ImageNet (bottom). The hyper-        |                                                                        |
|                                                                      | therefore, we plot here the certiﬁed accuracy of Wong et al. (2018)    |
| parameter σ controls a robustness/accuracy tradeoff. The dashed      |                                                                        |
|                                                                      | against the “approximate” certiﬁed accuracy of smoothing.              |
| black line is an upper bound on the empirical robust accuracy of     |                                                                        |
| an undefended classiﬁer with the base classiﬁer’s architecture.      |                                                                        |
| α is small,                                                          | to obtain an empirical upper bound on its robust accuracy.             |
| the difference between these two quantities is                       |                                                                        |
| negligible. Therefore, in our experiments we omit the step           | We certiﬁed the full CIFAR-10 test set and a subsample of              |
| for simplicity and report approximate certiﬁed accuracies.           | 500 examples from the ImageNet test set.                               |
| In all experiments, unless otherwise stated, we ran CERTIFY          | Figure 6 plots the certiﬁed accuracy attained by smoothing             |
| with α = 0.001, so there was at most a 0.1% chance that              | with each σ. The dashed black line is the empirical upper              |
| CERTIFY returned a radius in which g was not truly robust.           | bound on the robust accuracy of the base classiﬁer architec-           |
| Unless otherwise stated, when running CERTIFY we used                | ture; observe that smoothing improves substantially upon               |
| n0 = 100 Monte Carlo samples for selection and n =                   | the robustness of the undefended base classiﬁer architecture.          |
| 100,000 samples for estimation.                                      | We see that σ controls a robustness/accuracy tradeoff. When            |
|                                                                      | σ is low, small radii can be certiﬁed with high accuracy, but          |
| In the ﬁgures above that plot certiﬁed accuracy as a function        |                                                                        |
|                                                                      | large radii cannot be certiﬁed. When σ is high, larger radii           |
| of radius r, the certiﬁed accuracy always decreases gradually        |                                                                        |
|                                                                      | can be certiﬁed, but smaller radii are certiﬁed at a lower ac-         |
| with r until reaching some point where it plummets to zero.          |                                                                        |
|                                                                      | curacy. This observation echoes the ﬁnding in Tsipras et al.           |
| This drop occurs because for each noise level σ and number           |                                                                        |
|                                                                      | (2019) that adversarially trained networks with higher ro-             |
| of samples n, there is a hard upper limit to the radius we can       |                                                                        |
|                                                                      | bust accuracy tend to have lower standard accuracy. Tables             |
| certify with high probability, achieved when all n samples           |                                                                        |
|                                                                      | of these results are in Appendix E.                                    |
| are classiﬁed by f as the same class.                                |                                                                        |
|                                                                      | Figure 8 (left) plots the certiﬁed accuracy obtained using our         |
| ImageNet and CIFAR-10 results                                        | Theorem 1 guarantee alongside the certiﬁed accuracy ob-                |
| We applied random-                                                   |                                                                        |
| ized smoothing to CIFAR-10 (Krizhevsky, 2009) and Im-                | tained using the analogous bounds of Lecuyer et al. (2019)             |
| ageNet                                                               | and Li et al. (2018). Since our expression for the certiﬁed            |
| (Deng et al., 2009). On each dataset we trained                      |                                                                        |
| several smoothed classiﬁers, each with a different σ. On             | radius R is greater (and, in fact, tight), our bound delivers          |
| CIFAR-10 our base                                                    | higher certiﬁed accuracies. Figure 8 (middle) projects how             |
| classiﬁer was                                                        |                                                                        |
| a 110-layer                                                          |                                                                        |
| residual                                                             |                                                                        |
| network; certifying each example took 15 seconds on an               | the certiﬁed accuracy would have changed had CERTIFY                   |
| NVIDIA RTX 2080 Ti. On ImageNet our base classiﬁer                   | used more or fewer samples n (under the assumption that the            |
| was a ResNet-50; certifying each example took 110 seconds.           | relative class proportions in counts would have remained               |
| We also trained a neural network with the base classiﬁer’s           | constant). Finally, Figure 8 (right) plots the certiﬁed accu-          |
| architecture on clean data, and subjected it to a DeepFool (cid:96)2 | racy as the conﬁdence parameter α is varied. Observe that              |
| adversarial attack (Moosavi-Dezfooli et al., 2016), in order         | the certiﬁed accuracy is not very sensitive to α.                      |



## Page 8

Certiﬁed Adversarial Robustness via Randomized Smoothing
0.0
0.2
0.4
0.6
0.8
1.0
radius
0.0
0.2
0.4
0.6
0.8
1.0
certified accuracy
ours
(Lecuyer et al, 2018)
(Li et al, 2018)
0.0
0.2
0.4
0.6
0.8
1.0
1.2
1.4
radius
0.0
0.2
0.4
0.6
0.8
1.0
certified accuracy
n = 1,000
n = 10,000
n = 100,000
n = 1,000,000
n = 10,000,000
0.0
0.2
0.4
0.6
0.8
1.0
radius
0.0
0.2
0.4
0.6
0.8
1.0
certified accuracy
99.999% confidence
99.99% confidence
99.9% confidence
99% confidence
Figure 8. Experiments with randomized smoothing on ImageNet with σ = 0.25. Left: certiﬁed accuracies obtained using our Theorem 1
versus those obtained using the robustness guarantees derived in prior work. Middle: projections for the certiﬁed accuracy if the number
of samples n used by CERTIFY had been larger or smaller. Right: certiﬁed accuracy as the failure probability α of CERTIFY is varied.
Comparison to baselines
We compared randomized
smoothing to three baseline approaches for certiﬁed ℓ2 ro-
bustness: the duality approach from Wong et al. (2018),
the Lipschitz approach from Tsuzuku et al. (2018), and the
approach from Weng et al. (2018a); Zhang et al. (2018).
The strongest baseline was Wong et al. (2018); we defer the
comparison to the other two baselines to Appendix H.
In Figure 7, we compare the largest publicly released model
from Wong et al. (2018), a small resnet, to two randomized
smoothing classiﬁers: one which used the same small resnet
architecture for its base classiﬁer, and one which used a
larger 110-layer resnet for its base classiﬁer. First, observe
that smoothing with the large 110-layer resnet substantially
outperforms the baseline (across all hyperparameter set-
tings) at all radii. Second, observe that smoothing with the
small resnet also outperformed the method of Wong et al.
(2018) at all but the smallest radii. We attribute this latter re-
sult to the fact that neural networks trained using the method
of Wong et al. (2018) are “typically overregularized to the
point that many ﬁlters/weights become identically zero,” per
that paper. In contrast, the base classiﬁer in randomized
smoothing is a fully expressive neural network.
Prediction
It is computationally expensive to certify the
robustness of g around a point x, since the value of n in
CERTIFY must be very large. However, it is far cheaper
to evaluate g at x using PREDICT, since n can be small.
For example, when we ran PREDICT on ImageNet (σ =
0.25) using n = 100, making each prediction only took
0.15 seconds, and we attained a top-1 test accuracy of 65%
(Appendix E).
As discussed earlier, an adversary can potentially force PRE-
DICT to abstain with high probability. However, it is rela-
tively rare for PREDICT to abstain on the actual data dis-
tribution. On ImageNet (σ = 0.25), PREDICT with failure
probability α = 0.001 abstained 12% of the time when n =
100, 4% when n = 1000, and 1% when n = 10,000.
Empirical tightness of bound
When f is linear, there al-
ways exists a class-changing perturbation just beyond the
certiﬁed radius. Since neural networks are not linear, we em-
pirically assessed the tightness of our bound by subjecting
an ImageNet smoothed classiﬁer (σ = 0.25) to a projected
gradient descent-style adversarial attack (Appendix J.3). For
each example, we ran CERTIFY with α = 0.01, and, if the
example was correctly classiﬁed and certiﬁed robust at ra-
dius R, we tried ﬁnding an adversarial example for g within
radius 1.5R and within radius 2R. We succeeded 17% of
the time at radius 1.5R and 53% of the time at radius 2R.
5. Conclusion
Theorem 2 establishes that smoothing with Gaussian noise
naturally confers adversarial robustness in ℓ2 norm: if we
have no knowledge about the base classiﬁer beyond the dis-
tribution of f(x + ε), then the set of perturbations to which
the smoothed classiﬁer is provably robust is precisely an ℓ2
ball. We suspect that smoothing with other noise distribu-
tions may lead to similarly natural robustness guarantees for
other perturbation sets such as general ℓp norm balls.
Our strong empirical results suggest that randomized
smoothing is a promising direction for future research
into adversarially robust classiﬁcation. Many empirical
approaches have been “broken,” and provable approaches
based on certifying neural network classiﬁers have not been
shown to scale to networks of modern size. It seems to be
computationally infeasible to reason in any sophisticated
way about the decision boundaries of a large, expressive neu-
ral network. Randomized smoothing circumvents this prob-
lem: the smoothed classiﬁer is not itself a neural network,
though it leverages the discriminative ability of a neural
network base classiﬁer. To make the smoothed classiﬁer ro-
bust, one need simply make the base classiﬁer classify well
under noise. In this way, randomized smoothing reduces the
unsolved problem of adversarially robust classiﬁcation to
the comparably solved domain of supervised learning.


**Table 8 from page 8**

| 0                  | 1                     | 2                                                        | 3   | 4   | 5   | 6      | 7              | 8   | 9                  | 10                 |
|:-------------------|:----------------------|:---------------------------------------------------------|:----|:----|:----|:-------|:---------------|:----|:-------------------|:-------------------|
|                    |                       | Certiﬁed Adversarial Robustness via Randomized Smoothing |     |     |     |        |                |     |                    |                    |
| 1.0                |                       | 1.0                                                      |     |     |     |        |                |     | 1.0                |                    |
|                    | ours                  |                                                          |     |     |     |        | n = 1,000      |     |                    | 99.999% confidence |
|                    | (Lecuyer et al, 2018) |                                                          |     |     |     |        | n = 10,000     |     |                    | 99.99% confidence  |
| 0.8                |                       | 0.8                                                      |     |     |     |        |                |     | 0.8                |                    |
|                    | (Li et al, 2018)      |                                                          |     |     |     |        | n = 100,000    |     |                    | 99.9% confidence   |
|                    |                       |                                                          |     |     |     |        | n = 1,000,000  |     |                    | 99% confidence     |
| 0.6                |                       | 0.6                                                      |     |     |     |        |                |     | 0.6                |                    |
| certified accuracy |                       | certified accuracy                                       |     |     |     |        | n = 10,000,000 |     | certified accuracy |                    |
| 0.4                |                       | 0.4                                                      |     |     |     |        |                |     | 0.4                |                    |
| 0.2                |                       | 0.2                                                      |     |     |     |        |                |     | 0.2                |                    |
| 0.0                |                       | 0.0                                                      |     |     |     |        |                |     | 0.0                |                    |
| 0.0                | 0.4                   | 0.0                                                      | 0.2 | 0.4 | 0.6 | 0.8    | 1.0            | 1.4 | 0.0                | 0.4                |
| 0.2                | 0.6                   |                                                          |     |     |     |        | 1.2            |     | 0.2                | 0.6                |
|                    | 0.8                   |                                                          |     |     |     |        |                |     |                    | 0.8                |
|                    | 1.0                   |                                                          |     |     |     |        |                |     |                    | 1.0                |
|                    | radius                |                                                          |     |     |     | radius |                |     |                    | radius             |

**Table 9 from page 8**

| 0                                                                                                                                        | 1                                                                   |
|:-----------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------|
| 0.2                                                                                                                                      | 0.2                                                                 |
| 0.2                                                                                                                                      |                                                                     |
| 0.0                                                                                                                                      | 0.0                                                                 |
| 0.0                                                                                                                                      |                                                                     |
| 0.0                                                                                                                                      | 0.8                                                                 |
| 0.2                                                                                                                                      | 1.0                                                                 |
| 0.4                                                                                                                                      | 1.2                                                                 |
| 0.6                                                                                                                                      | 1.4                                                                 |
| 0.8                                                                                                                                      | 0.0                                                                 |
| 1.0                                                                                                                                      | 0.2                                                                 |
| 0.0                                                                                                                                      | 0.4                                                                 |
| 0.2                                                                                                                                      | 0.6                                                                 |
| 0.4                                                                                                                                      | 0.8                                                                 |
| 0.6                                                                                                                                      | 1.0                                                                 |
| radius                                                                                                                                   | radius                                                              |
|                                                                                                                                          | radius                                                              |
| Figure 8. Experiments with randomized smoothing on ImageNet with σ = 0.25. Left: certiﬁed accuracies obtained using our Theorem 1        |                                                                     |
| versus those obtained using the robustness guarantees derived in prior work. Middle: projections for the certiﬁed accuracy if the number |                                                                     |
| of samples n used by CERTIFY had been larger or smaller. Right: certiﬁed accuracy as the failure probability α of CERTIFY is varied.     |                                                                     |
| Comparison                                                                                                                               | Empirical tightness of bound                                        |
| to                                                                                                                                       | When f is linear, there al-                                         |
| baselines                                                                                                                                |                                                                     |
| We                                                                                                                                       |                                                                     |
| compared                                                                                                                                 |                                                                     |
| randomized                                                                                                                               |                                                                     |
| smoothing to three baseline approaches for certiﬁed (cid:96)2 ro-                                                                        | ways exists a class-changing perturbation just beyond the           |
| bustness:                                                                                                                                | certiﬁed radius. Since neural networks are not linear, we em-       |
| the duality approach from Wong et al.                                                                                                    |                                                                     |
| (2018),                                                                                                                                  |                                                                     |
| the Lipschitz approach from Tsuzuku et al. (2018), and the                                                                               | pirically assessed the tightness of our bound by subjecting         |
| approach from Weng et al.                                                                                                                | an ImageNet smoothed classiﬁer (σ = 0.25) to a projected            |
| (2018a); Zhang et al.                                                                                                                    |                                                                     |
| (2018).                                                                                                                                  |                                                                     |
| The strongest baseline was Wong et al. (2018); we defer the                                                                              | gradient descent-style adversarial attack (Appendix J.3). For       |
| comparison to the other two baselines to Appendix H.                                                                                     | each example, we ran CERTIFY with α = 0.01, and, if the             |
|                                                                                                                                          | example was correctly classiﬁed and certiﬁed robust at ra-          |
| In Figure 7, we compare the largest publicly released model                                                                              |                                                                     |
|                                                                                                                                          | dius R, we tried ﬁnding an adversarial example for g within         |
| from Wong et al. (2018), a small resnet, to two randomized                                                                               |                                                                     |
|                                                                                                                                          | radius 1.5R and within radius 2R. We succeeded 17% of               |
| smoothing classiﬁers: one which used the same small resnet                                                                               |                                                                     |
|                                                                                                                                          | the time at radius 1.5R and 53% of the time at radius 2R.           |
| architecture for                                                                                                                         |                                                                     |
| its base classiﬁer, and one which used a                                                                                                 |                                                                     |
| larger 110-layer resnet for its base classiﬁer. First, observe                                                                           |                                                                     |
| that smoothing with the large 110-layer resnet substantially                                                                             | 5. Conclusion                                                       |
| outperforms the baseline (across all hyperparameter set-                                                                                 |                                                                     |
|                                                                                                                                          | Theorem 2 establishes that smoothing with Gaussian noise            |
| tings) at all radii. Second, observe that smoothing with the                                                                             |                                                                     |
|                                                                                                                                          | if we                                                               |
|                                                                                                                                          | naturally confers adversarial robustness in (cid:96)2 norm:         |
| small resnet also outperformed the method of Wong et al.                                                                                 |                                                                     |
|                                                                                                                                          | have no knowledge about the base classiﬁer beyond the dis-          |
| (2018) at all but the smallest radii. We attribute this latter re-                                                                       |                                                                     |
|                                                                                                                                          | tribution of f (x + ε), then the set of perturbations to which      |
| sult to the fact that neural networks trained using the method                                                                           |                                                                     |
|                                                                                                                                          | the smoothed classiﬁer is provably robust is precisely an (cid:96)2 |
| of Wong et al. (2018) are “typically overregularized to the                                                                              |                                                                     |
|                                                                                                                                          | ball. We suspect that smoothing with other noise distribu-          |
| point that many ﬁlters/weights become identically zero,” per                                                                             |                                                                     |
|                                                                                                                                          | tions may lead to similarly natural robustness guarantees for       |
| that paper.                                                                                                                              |                                                                     |
| In contrast,                                                                                                                             |                                                                     |
| the base classiﬁer in randomized                                                                                                         |                                                                     |
|                                                                                                                                          | other perturbation sets such as general (cid:96)p norm balls.       |
| smoothing is a fully expressive neural network.                                                                                          |                                                                     |
|                                                                                                                                          | Our                                                                 |
|                                                                                                                                          | strong                                                              |
|                                                                                                                                          | empirical                                                           |
|                                                                                                                                          | results                                                             |
|                                                                                                                                          | suggest                                                             |
|                                                                                                                                          | that                                                                |
|                                                                                                                                          | randomized                                                          |
|                                                                                                                                          | smoothing is                                                        |
|                                                                                                                                          | a promising direction for                                           |
|                                                                                                                                          | future                                                              |
|                                                                                                                                          | research                                                            |
| Prediction                                                                                                                               |                                                                     |
| It is computationally expensive to certify the                                                                                           |                                                                     |
|                                                                                                                                          | into adversarially robust classiﬁcation. Many empirical             |
| robustness of g around a point x, since the value of n in                                                                                |                                                                     |
|                                                                                                                                          | approaches have been “broken,” and provable approaches              |
| CERTIFY must be very large. However,                                                                                                     |                                                                     |
| it                                                                                                                                       |                                                                     |
| is far cheaper                                                                                                                           |                                                                     |
|                                                                                                                                          | based on certifying neural network classiﬁers have not been         |
| to evaluate g at x using PREDICT, since n can be small.                                                                                  |                                                                     |
|                                                                                                                                          | shown to scale to networks of modern size. It seems to be           |
| For example, when we ran PREDICT on ImageNet (σ =                                                                                        |                                                                     |
|                                                                                                                                          | computationally infeasible to reason in any sophisticated           |
| 0.25) using n = 100, making each prediction only took                                                                                    |                                                                     |
|                                                                                                                                          | way about the decision boundaries of a large, expressive neu-       |
| 0.15 seconds, and we attained a top-1 test accuracy of 65%                                                                               |                                                                     |
|                                                                                                                                          | ral network. Randomized smoothing circumvents this prob-            |
| (Appendix E).                                                                                                                            |                                                                     |
|                                                                                                                                          | lem:                                                                |
|                                                                                                                                          | the smoothed classiﬁer is not itself a neural network,              |
|                                                                                                                                          | though it                                                           |
|                                                                                                                                          | leverages the discriminative ability of a neural                    |
| As discussed earlier, an adversary can potentially force PRE-                                                                            |                                                                     |
|                                                                                                                                          | network base classiﬁer. To make the smoothed classiﬁer ro-          |
| DICT to abstain with high probability. However, it is rela-                                                                              |                                                                     |
|                                                                                                                                          | bust, one need simply make the base classiﬁer classify well         |
| tively rare for PREDICT to abstain on the actual data dis-                                                                               |                                                                     |
|                                                                                                                                          | under noise. In this way, randomized smoothing reduces the          |
| tribution. On ImageNet (σ = 0.25), PREDICT with failure                                                                                  |                                                                     |
|                                                                                                                                          | unsolved problem of adversarially robust classiﬁcation to           |
| probability α = 0.001 abstained 12% of the time when n =                                                                                 |                                                                     |
|                                                                                                                                          | the comparably solved domain of supervised learning.                |
| 100, 4% when n = 1000, and 1% when n = 10,000.                                                                                           |                                                                     |



## Page 9

Certiﬁed Adversarial Robustness via Randomized Smoothing
6. Acknowledgements
We thank Mateusz Kwa´snicki for help with Lemma 4 in the
appendix, Aaditya Ramdas for pointing us toward the work
of Hung & Fithian (2019), and Siva Balakrishnan for helpful
discussions regarding the conﬁdence interval in Appendix
D. We thank Tolani Olarinre, Adarsh Prasad, Ben Cousins,
Ramon Van Handel, Matthias Lecuyer, and Bai Li for useful
conversations. Finally, we are very grateful to Vaishnavh
Nagarajan, Arun Sai Suggala, Shaojie Bai, Mikhail Khodak,
Han Zhao, and Zachary Lipton for reviewing drafts of this
work. Jeremy Cohen is supported by a grant from the Bosch
Center for AI.
References
Anil, C., Lucas, J., and Grosse, R. B. Sorting out lips-
chitz function approximation. In Proceedings of the 36th
International Conference on Machine Learning, 2019.
Athalye, A. and Carlini, N. On the robustness of the cvpr
2018 white-box adversarial example defenses. The Bright
and Dark Sides of Computer Vision: Challenges and
Opportunities for Privacy and Security, 2018.
Athalye, A., Carlini, N., and Wagner, D. Obfuscated gra-
dients give a false sense of security: Circumventing de-
fenses to adversarial examples. In Proceedings of the 35th
International Conference on Machine Learning, 2018.
Biggio, B., Corona, I., Maiorca, D., Nelson, B., rndi, N.,
Laskov, P., Giacinto, G., and Roli, F. Evasion attacks
against machine learning at test time. Joint European
Conference on Machine Learning and Knowledge Dis-
covery in Database, 2013.
Blanchard,
G.
Lecture
Notes,
2007.
URL
http://www.math.uni-potsdam.de/
˜blanchard/lectures/lect_2.pdf.
Bunel, R. R., Turkaslan, I., Torr, P., Kohli, P., and
Mudigonda, P. K. A uniﬁed view of piecewise linear
neural network veriﬁcation. In Advances in Neural Infor-
mation Processing Systems 31. 2018.
Cao, X. and Gong, N. Z. Mitigating evasion attacks to deep
neural networks via region-based classiﬁcation. 33rd An-
nual Computer Security Applications Conference, 2017.
Carlini, N. and Wagner, D. Adversarial examples are not
easily detected: Bypassing ten detection methods. In
Proceedings of the 10th ACM Workshop on Artiﬁcial
Intelligence and Security, 2017.
Carlini, N., Katz, G., Barrett, C., and Dill, D. L. Provably
minimally-distorted adversarial examples. arXiv preprint
arXiv: 1709.10207, 2017.
Cheng, C.-H., Nhrenberg, G., and Ruess, H. Maximum
resilience of artiﬁcial neural networks.
International
Symposium on Automated Technology for Veriﬁcation
and Analysis, 2017.
Cisse, M., Bojanowski, P., Grave, E., Dauphin, Y., and
Usunier, N. Parseval networks: Improving robustness to
adversarial examples. In Proceedings of the 34th Interna-
tional Conference on Machine Learning, 2017.
Clopper, C. J. and Pearson, E. S. The use of conﬁdence
or ﬁducial limits illustrated in the case of the binomial.
Biometrika, 26(4):pp. 404–413, 1934. ISSN 00063444.
Croce, F., Andriushchenko, M., and Hein, M. Provable
robustness of relu networks via maximization of linear
regions. In Proceedings of the 22nd International Con-
ference on Artiﬁcial Intelligence and Statistics, 2019.
Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-
Fei, L. ImageNet: A Large-Scale Hierarchical Image
Database. In IEEE Conference on Computer Vision and
Pattern Recognition (CVPR), 2009.
Dodge, S. and Karam, L. A study and comparison of hu-
man and deep learning recognition performance under
visual distortions. 2017 26th International Conference on
Computer Communication and Networks (ICCCN), 2017.
Dutta, S., Jha, S., Sanakaranarayanan, S., and Tiwari, A.
Output range analysis for deep neural networks. arXiv
preprint arXiv:1709.09130, 2017.
Dvijotham, K., Gowal, S., Stanforth, R., Arandjelovic, R.,
O’Donoghue, B., Uesato, J., and Kohli, P.
Training
veriﬁed learners with learned veriﬁers. arXiv preprint
arXiv:1805.10265, 2018a.
Dvijotham, K., Stanforth, R., Gowal, S., Mann, T., and
Kohli, P. A dual approach to scalable veriﬁcation of
deep networks. Proceedings of the Thirty-Fourth Con-
ference Annual Conference on Uncertainty in Artiﬁcial
Intelligence (UAI-18), 2018b.
Ehlers, R. Formal veriﬁcation of piece-wise linear feed-
forward neural networks. In Automated Technology for
Veriﬁcation and Analysis, 2017.
Fawzi, A., Moosavi-Dezfooli, S.-M., and Frossard, P. Ro-
bustness of classiﬁers: from adversarial to random noise.
In Advances in Neural Information Processing Systems
29. 2016.
Fischetti, M. and Jo, J. Deep neural networks and mixed
integer linear optimization. Constraints, 23(3):296–309,
July 2018.


**Table 10 from page 9**

| 0                                                            | 1                                                           |
|:-------------------------------------------------------------|:------------------------------------------------------------|
| Certiﬁed Adversarial Robustness via Randomized Smoothing     |                                                             |
| 6. Acknowledgements                                          | Cheng, C.-H., Nhrenberg, G., and Ruess, H. Maximum          |
|                                                              | International                                               |
|                                                              | resilience of artiﬁcial neural networks.                    |
| We thank Mateusz Kwa´snicki for help with Lemma 4 in the     |                                                             |
|                                                              | Symposium on Automated Technology for Veriﬁcation           |
| appendix, Aaditya Ramdas for pointing us toward the work     |                                                             |
|                                                              | and Analysis, 2017.                                         |
| of Hung & Fithian (2019), and Siva Balakrishnan for helpful  |                                                             |
| discussions regarding the conﬁdence interval in Appendix     |                                                             |
|                                                              | Cisse, M., Bojanowski, P., Grave, E., Dauphin, Y., and      |
| D. We thank Tolani Olarinre, Adarsh Prasad, Ben Cousins,     |                                                             |
|                                                              | Usunier, N. Parseval networks: Improving robustness to      |
| Ramon Van Handel, Matthias Lecuyer, and Bai Li for useful    |                                                             |
|                                                              | adversarial examples.                                       |
|                                                              | In Proceedings of the 34th Interna-                         |
| conversations. Finally, we are very grateful                 |                                                             |
| to Vaishnavh                                                 |                                                             |
|                                                              | tional Conference on Machine Learning, 2017.                |
| Nagarajan, Arun Sai Suggala, Shaojie Bai, Mikhail Khodak,    |                                                             |
| Han Zhao, and Zachary Lipton for reviewing drafts of this    | Clopper, C. J. and Pearson, E. S. The use of conﬁdence      |
| work. Jeremy Cohen is supported by a grant from the Bosch    | or ﬁducial                                                  |
|                                                              | limits illustrated in the case of the binomial.             |
| Center for AI.                                               | Biometrika, 26(4):pp. 404–413, 1934.                        |
|                                                              | ISSN 00063444.                                              |
|                                                              | Croce, F., Andriushchenko, M., and Hein, M.                 |
|                                                              | Provable                                                    |
| References                                                   |                                                             |
|                                                              | robustness of relu networks via maximization of linear      |
| Anil, C., Lucas,                                             | regions.                                                    |
| J., and Grosse, R. B.                                        | In Proceedings of the 22nd International Con-               |
| Sorting out                                                  |                                                             |
| lips-                                                        |                                                             |
| chitz function approximation.                                | ference on Artiﬁcial Intelligence and Statistics, 2019.     |
| In Proceedings of the 36th                                   |                                                             |
| International Conference on Machine Learning, 2019.          |                                                             |
|                                                              | Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei- |
| Athalye, A. and Carlini, N. On the robustness of the cvpr    | Fei, L.                                                     |
|                                                              | ImageNet: A Large-Scale Hierarchical Image                  |
| 2018 white-box adversarial example defenses. The Bright      | Database.                                                   |
|                                                              | In IEEE Conference on Computer Vision and                   |
| and Dark Sides of Computer Vision: Challenges and            |                                                             |
|                                                              | Pattern Recognition (CVPR), 2009.                           |
| Opportunities for Privacy and Security, 2018.                |                                                             |
|                                                              | Dodge, S. and Karam, L. A study and comparison of hu-       |
| Athalye, A., Carlini, N., and Wagner, D. Obfuscated gra-     |                                                             |
|                                                              | man and deep learning recognition performance under         |
| dients give a false sense of security: Circumventing de-     |                                                             |
|                                                              | visual distortions. 2017 26th International Conference on   |
| fenses to adversarial examples. In Proceedings of the 35th   |                                                             |
|                                                              | Computer Communication and Networks (ICCCN), 2017.          |
| International Conference on Machine Learning, 2018.          |                                                             |
|                                                              | Dutta, S., Jha, S., Sanakaranarayanan, S., and Tiwari, A.   |
| Biggio, B., Corona, I., Maiorca, D., Nelson, B., rndi, N.,   |                                                             |
|                                                              | Output range analysis for deep neural networks. arXiv       |
| Laskov, P., Giacinto, G., and Roli, F. Evasion attacks       |                                                             |
|                                                              | preprint arXiv:1709.09130, 2017.                            |
| Joint European                                               |                                                             |
| against machine learning at                                  |                                                             |
| test                                                         |                                                             |
| time.                                                        |                                                             |
| Conference on Machine Learning and Knowledge Dis-            |                                                             |
|                                                              | Dvijotham, K., Gowal, S., Stanforth, R., Arandjelovic, R.,  |
| covery in Database, 2013.                                    |                                                             |
|                                                              | O’Donoghue, B., Uesato,                                     |
|                                                              | J., and Kohli, P.                                           |
|                                                              | Training                                                    |
|                                                              | arXiv preprint                                              |
|                                                              | veriﬁed learners with learned veriﬁers.                     |
| Blanchard,                                                   |                                                             |
| G.                                                           |                                                             |
| Lecture                                                      |                                                             |
| Notes,                                                       |                                                             |
| 2007.                                                        |                                                             |
| URL                                                          |                                                             |
|                                                              | arXiv:1805.10265, 2018a.                                    |
| http://www.math.uni-potsdam.de/                              |                                                             |
| ˜blanchard/lectures/lect_2.pdf.                              |                                                             |
|                                                              | Dvijotham, K., Stanforth, R., Gowal, S., Mann, T., and      |
|                                                              | Kohli, P.                                                   |
|                                                              | A dual approach to scalable veriﬁcation of                  |
| Bunel, R. R.,                                                |                                                             |
| Turkaslan,                                                   |                                                             |
| I.,                                                          |                                                             |
| Torr,                                                        |                                                             |
| P., Kohli,                                                   |                                                             |
| P.,                                                          |                                                             |
| and                                                          |                                                             |
|                                                              | the Thirty-Fourth Con-                                      |
|                                                              | deep networks. Proceedings of                               |
| Mudigonda, P. K. A uniﬁed view of piecewise linear           |                                                             |
|                                                              | ference Annual Conference on Uncertainty in Artiﬁcial       |
| neural network veriﬁcation.                                  |                                                             |
| In Advances in Neural Infor-                                 |                                                             |
|                                                              | Intelligence (UAI-18), 2018b.                               |
| mation Processing Systems 31. 2018.                          |                                                             |
|                                                              | Ehlers, R.                                                  |
|                                                              | Formal veriﬁcation of piece-wise linear feed-               |
| Cao, X. and Gong, N. Z. Mitigating evasion attacks to deep   |                                                             |
|                                                              | forward neural networks.                                    |
|                                                              | In Automated Technology for                                 |
| neural networks via region-based classiﬁcation. 33rd An-     |                                                             |
|                                                              | Veriﬁcation and Analysis, 2017.                             |
| nual Computer Security Applications Conference, 2017.        |                                                             |
| Carlini, N. and Wagner, D. Adversarial examples are not      | Fawzi, A., Moosavi-Dezfooli, S.-M., and Frossard, P. Ro-    |
| easily detected: Bypassing ten detection methods.            | bustness of classiﬁers: from adversarial to random noise.   |
| In                                                           |                                                             |
| Proceedings of                                               | In Advances in Neural Information Processing Systems        |
| the 10th ACM Workshop on Artiﬁcial                           |                                                             |
| Intelligence and Security, 2017.                             | 29. 2016.                                                   |
| Carlini, N., Katz, G., Barrett, C., and Dill, D. L. Provably | Fischetti, M. and Jo, J. Deep neural networks and mixed     |
| minimally-distorted adversarial examples. arXiv preprint     | integer linear optimization. Constraints, 23(3):296–309,    |
| arXiv: 1709.10207, 2017.                                     | July 2018.                                                  |



## Page 10

Certiﬁed Adversarial Robustness via Randomized Smoothing
Ford, N., Gilmer, J., and Cubuk, E. D. Adversarial ex-
amples are a natural consequence of test error in noise.
In Proceedings of the 36th International Conference on
Machine Learning, 2019.
Franceschi, J.-Y., Fawzi, A., and Fawzi, O. Robustness
of classiﬁers to uniform ℓp and gaussian noise. In 21st
International Conference on Artiﬁcial Intelligence and
Statistics (AISTATS). 2018.
Gehr, T., Mirman, M., Drachsler-Cohen, D., Tsankov, P.,
Chaudhuri, S., and Vechev, M. T. AI2: safety and ro-
bustness certiﬁcation of neural networks with abstract
interpretation. In 2018 IEEE Symposium on Security and
Privacy, SP 2018, Proceedings, 21-23 May 2018, San
Francisco, California, USA, pp. 3–18, 2018.
Goodfellow, I. J., Shlens, J., and Szegedy, C. Explaining
and harnessing adversarial examples. In International
Conference on Learning Representations, 2015.
Gouk, H., Frank, E., Pfahringer, B., and Cree, M. Regulari-
sation of neural networks by enforcing lipschitz continu-
ity. arXiv preprint arXiv:1804.04368, 2018.
Gowal, S., Dvijotham, K., Stanforth, R., Bunel, R., Qin,
C., Uesato, J., Arandjelovic, R., Mann, T., and Kohli, P.
On the effectiveness of interval bound propagation for
training veriﬁably robust models, 2018.
Hein, M. and Andriushchenko, M. Formal guarantees on the
robustness of a classiﬁer against adversarial manipulation.
In Advances in Neural Information Processing Systems
30. 2017.
Huang, X., Kwiatkowska, M., Wang, S., and Wu, M. Safety
veriﬁcation of deep neural networks. Computer Aided
Veriﬁcation, 2017.
Hung, K. and Fithian, W. Rank veriﬁcation for exponential
families. The Annals of Statistics, (2):758–782, 04 2019.
Kannan, H., Kurakin, A., and Goodfellow, I. Adversarial
logit pairing. arXiv preprint arXiv:1803.06373, 2018.
Katz, G., Barrett, C., Dill, D. L., Julian, K., and Kochender-
fer, M. J. Reluplex: An efﬁcient smt solver for verifying
deep neural networks. Lecture Notes in Computer Sci-
ence, pp. 97117, 2017. ISSN 1611-3349.
Kolter,
J.
Z.
and
Madry,
A.
Adversarial
ro-
bustness:
Theory
and
practice.
https:
//adversarial-ml-tutorial.org/
adversarial_examples/, 2018.
Krizhevsky, A. Learning multiple layers of features from
tiny images. Technical report, 2009.
Kurakin, A., Goodfellow, I. J., and Bengio, S.
Adver-
sarial machine learning at scale. 2017. URL https:
//arxiv.org/abs/1611.01236.
Lecuyer, M., Atlidakis, V., Geambasu, R., Hsu, D., and
Jana, S. Certiﬁed robustness to adversarial examples with
differential privacy. In IEEE Symposium on Security and
Privacy (SP), 2019.
Levine, A., Singla, S., and Feizi, S.
Certiﬁably ro-
bust interpretation in deep learning.
arXiv preprint
arXiv:1905.12105, 2019.
Li, B., Chen, C., Wang, W., and Carin, L. Second-order ad-
versarial attack and certiﬁable robustness. arXiv preprint
arXiv:1809.03113, 2018.
Liu, X., Cheng, M., Zhang, H., and Hsieh, C.-J. Towards
robust neural networks via random self-ensemble. In
The European Conference on Computer Vision (ECCV),
September 2018.
Lomuscio, A. and Maganti, L. An approach to reachability
analysis for feed-forward relu neural networks, 2017.
Madry, A., Makelov, A., Schmidt, L., Tsipras, D., and
Vladu, A. Towards deep learning models resistant to
adversarial attacks. In International Conference on Learn-
ing Representations, 2018.
Mirman, M., Gehr, T., and Vechev, M. Differentiable ab-
stract interpretation for provably robust neural networks.
In Proceedings of the 35th International Conference on
Machine Learning, 2018.
Moosavi-Dezfooli, S.-M., Fawzi, A., and Frossard, P. Deep-
fool: A simple and accurate method to fool deep neural
networks. 2016 IEEE Conference on Computer Vision
and Pattern Recognition (CVPR), 2016.
Neyman, J. and Pearson, E. S. On the problem of the most
efﬁcient tests of statistical hypotheses. Philosophical
Transactions of the Royal Society of London. Series A,
Containing Papers of a Mathematical or Physical Char-
acter, 231:289–337, 1933.
Raghunathan, A., Steinhardt, J., and Liang, P. Certiﬁed
defenses against adversarial examples. In International
Conference on Learning Representations, 2018a.
Raghunathan, A., Steinhardt, J., and Liang, P. Semideﬁ-
nite relaxations for certifying robustness to adversarial
examples. In Advances in Neural Information Processing
Systems 31, 2018b.
Salman, H., Yang, G., Li, J., Zhang, P., Zhang, H., Razen-
shteyn, I., and Bubeck, S. Provably robust deep learn-
ing via adversarially trained smoothed classiﬁers. arXiv
preprint arXiv:1906.04584, 2019.


**Table 11 from page 10**

| 0                                                              | 1                                                          |
|:---------------------------------------------------------------|:-----------------------------------------------------------|
| Certiﬁed Adversarial Robustness via Randomized Smoothing       |                                                            |
| Ford, N., Gilmer,                                              | Kurakin, A., Goodfellow,                                   |
| J., and Cubuk, E. D.                                           | I.                                                         |
| Adversarial ex-                                                | J., and Bengio, S.                                         |
|                                                                | Adver-                                                     |
| amples are a natural consequence of test error in noise.       | sarial machine learning at scale.                          |
|                                                                | 2017. URL https:                                           |
| In Proceedings of the 36th International Conference on         | //arxiv.org/abs/1611.01236.                                |
| Machine Learning, 2019.                                        |                                                            |
|                                                                | Lecuyer, M., Atlidakis, V., Geambasu, R., Hsu, D., and     |
| Franceschi, J.-Y., Fawzi, A., and Fawzi, O.                    | Jana, S. Certiﬁed robustness to adversarial examples with  |
| Robustness                                                     |                                                            |
| of classiﬁers to uniform (cid:96) p and gaussian noise.        | differential privacy.                                      |
| In 21st                                                        | In IEEE Symposium on Security and                          |
| International Conference on Artiﬁcial Intelligence and         | Privacy (SP), 2019.                                        |
| Statistics (AISTATS). 2018.                                    |                                                            |
|                                                                | Levine, A.,                                                |
|                                                                | Singla,                                                    |
|                                                                | S.,                                                        |
|                                                                | and Feizi,                                                 |
|                                                                | S.                                                         |
|                                                                | Certiﬁably                                                 |
|                                                                | ro-                                                        |
|                                                                | arXiv preprint                                             |
|                                                                | bust                                                       |
|                                                                | interpretation in deep learning.                           |
| Gehr, T., Mirman, M., Drachsler-Cohen, D., Tsankov, P.,        |                                                            |
|                                                                | arXiv:1905.12105, 2019.                                    |
| Chaudhuri, S., and Vechev, M. T. AI2:                          |                                                            |
| safety and ro-                                                 |                                                            |
| bustness certiﬁcation of neural networks with abstract         |                                                            |
|                                                                | Li, B., Chen, C., Wang, W., and Carin, L. Second-order ad- |
| interpretation.                                                |                                                            |
| In 2018 IEEE Symposium on Security and                         |                                                            |
|                                                                | versarial attack and certiﬁable robustness. arXiv preprint |
| Privacy, SP 2018, Proceedings, 21-23 May 2018, San             |                                                            |
|                                                                | arXiv:1809.03113, 2018.                                    |
| Francisco, California, USA, pp. 3–18, 2018.                    |                                                            |
|                                                                | Liu, X., Cheng, M., Zhang, H., and Hsieh, C.-J. Towards    |
| Goodfellow, I. J., Shlens, J., and Szegedy, C. Explaining      | robust neural networks via random self-ensemble.           |
|                                                                | In                                                         |
| and harnessing adversarial examples.                           | The European Conference on Computer Vision (ECCV),         |
| In International                                               |                                                            |
| Conference on Learning Representations, 2015.                  | September 2018.                                            |
| Gouk, H., Frank, E., Pfahringer, B., and Cree, M. Regulari-    | Lomuscio, A. and Maganti, L. An approach to reachability   |
| sation of neural networks by enforcing lipschitz continu-      | analysis for feed-forward relu neural networks, 2017.      |
| ity. arXiv preprint arXiv:1804.04368, 2018.                    |                                                            |
|                                                                | Madry, A., Makelov, A., Schmidt, L., Tsipras, D., and      |
|                                                                | Vladu, A.                                                  |
|                                                                | Towards deep learning models resistant                     |
|                                                                | to                                                         |
| Gowal, S., Dvijotham, K., Stanforth, R., Bunel, R., Qin,       |                                                            |
|                                                                | adversarial attacks. In International Conference on Learn- |
| C., Uesato, J., Arandjelovic, R., Mann, T., and Kohli, P.      |                                                            |
|                                                                | ing Representations, 2018.                                 |
| On the effectiveness of interval bound propagation for         |                                                            |
| training veriﬁably robust models, 2018.                        |                                                            |
|                                                                | Mirman, M., Gehr, T., and Vechev, M. Differentiable ab-    |
|                                                                | stract interpretation for provably robust neural networks. |
| Hein, M. and Andriushchenko, M. Formal guarantees on the       |                                                            |
|                                                                | In Proceedings of the 35th International Conference on     |
| robustness of a classiﬁer against adversarial manipulation.    |                                                            |
|                                                                | Machine Learning, 2018.                                    |
| In Advances in Neural Information Processing Systems           |                                                            |
| 30. 2017.                                                      |                                                            |
|                                                                | Moosavi-Dezfooli, S.-M., Fawzi, A., and Frossard, P. Deep- |
|                                                                | fool: A simple and accurate method to fool deep neural     |
| Huang, X., Kwiatkowska, M., Wang, S., and Wu, M. Safety        |                                                            |
|                                                                | networks. 2016 IEEE Conference on Computer Vision          |
| veriﬁcation of deep neural networks. Computer Aided            |                                                            |
|                                                                | and Pattern Recognition (CVPR), 2016.                      |
| Veriﬁcation, 2017.                                             |                                                            |
|                                                                | Neyman, J. and Pearson, E. S. On the problem of the most   |
| Hung, K. and Fithian, W. Rank veriﬁcation for exponential      |                                                            |
|                                                                | Philosophical                                              |
|                                                                | efﬁcient                                                   |
|                                                                | tests of statistical hypotheses.                           |
| families. The Annals of Statistics, (2):758–782, 04 2019.      |                                                            |
|                                                                | Transactions of                                            |
|                                                                | the Royal Society of London. Series A,                     |
|                                                                | Containing Papers of a Mathematical or Physical Char-      |
| Kannan, H., Kurakin, A., and Goodfellow, I. Adversarial        |                                                            |
|                                                                | acter, 231:289–337, 1933.                                  |
| logit pairing. arXiv preprint arXiv:1803.06373, 2018.          |                                                            |
|                                                                | Raghunathan, A., Steinhardt, J., and Liang, P. Certiﬁed    |
| Katz, G., Barrett, C., Dill, D. L., Julian, K., and Kochender- |                                                            |
|                                                                | defenses against adversarial examples.                     |
|                                                                | In International                                           |
| fer, M. J. Reluplex: An efﬁcient smt solver for verifying      |                                                            |
|                                                                | Conference on Learning Representations, 2018a.             |
| deep neural networks. Lecture Notes in Computer Sci-           |                                                            |
| ence, pp. 97117, 2017.                                         | Raghunathan, A., Steinhardt, J., and Liang, P.             |
| ISSN 1611-3349.                                                | Semideﬁ-                                                   |
|                                                                | nite relaxations for certifying robustness to adversarial  |
| Kolter,                                                        | examples.                                                  |
| J.                                                             | In Advances in Neural Information Processing               |
| Z.                                                             |                                                            |
| and Madry,                                                     |                                                            |
| A.                                                             |                                                            |
| Adversarial                                                    |                                                            |
| ro-                                                            |                                                            |
| https:                                                         |                                                            |
| bustness:                                                      |                                                            |
| Theory                                                         |                                                            |
| and                                                            |                                                            |
| practice.                                                      |                                                            |
|                                                                | Systems 31, 2018b.                                         |
| //adversarial-ml-tutorial.org/                                 |                                                            |
|                                                                | Salman, H., Yang, G., Li, J., Zhang, P., Zhang, H., Razen- |
| adversarial_examples/, 2018.                                   |                                                            |
|                                                                | shteyn, I., and Bubeck, S. Provably robust deep learn-     |
| Krizhevsky, A. Learning multiple layers of features from       | ing via adversarially trained smoothed classiﬁers. arXiv   |
| tiny images. Technical report, 2009.                           | preprint arXiv:1906.04584, 2019.                           |



## Page 11

Certiﬁed Adversarial Robustness via Randomized Smoothing
Singh, G., Gehr, T., Mirman, M., P¨uschel, M., and Vechev,
M. Fast and effective robustness certiﬁcation. In Ad-
vances in Neural Information Processing Systems 31.
2018.
Smilkov, D., Thorat, N., Kim, B., Vigas, F., and Wattenberg,
M. Smoothgrad: removing noise by adding noise. arXiv
preprint arXiv:1706.03825, 2017.
Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan,
D., Goodfellow, I., and Fergus, R. Intriguing proper-
ties of neural networks. In International Conference on
Learning Representations, 2014.
Tjeng, V., Xiao, K. Y., and Tedrake, R. Evaluating robust-
ness of neural networks with mixed integer programming.
In International Conference on Learning Representations,
2019. URL https://openreview.net/forum?
id=HyGIdiRqtm.
Tsipras, D., Santurkar, S., Engstrom, L., Turner, A., and
Madry, A. Robustness may be at odds with accuracy. In
International Conference on Learning Representations,
2019. URL https://openreview.net/forum?
id=SyxAb30cY7.
Tsuzuku, Y., Sato, I., and Sugiyama, M. Lipschitz-margin
training: Scalable certiﬁcation of perturbation invariance
for deep neural networks. In Advances in Neural Infor-
mation Processing Systems 31. 2018.
Uesato, J., O’Donoghue, B., Kohli, P., and van den Oord,
A. Adversarial risk and the dangers of evaluating against
weak attacks. In Proceedings of the 35th International
Conference on Machine Learning, 2018.
Wang, S., Chen, Y., Abdou, A., and Jana, S. Mixtrain: Scal-
able training of formally robust neural networks. arXiv
preprint arXiv:1811.02625, 2018a.
Wang, S., Pei, K., Whitehouse, J., Yang, J., and Jana, S.
Efﬁcient formal safety analysis of neural networks. In
Advances in Neural Information Processing Systems 31.
2018b.
Webb, S., Rainforth, T., Teh, Y. W., and Kumar, M. P.
Statistical veriﬁcation of neural networks.
In In-
ternational Conference on Learning Representations,
2019. URL https://openreview.net/forum?
id=S1xcx3C5FX.
Weng, L., Zhang, H., Chen, H., Song, Z., Hsieh, C.-J.,
Daniel, L., Boning, D., and Dhillon, I. Towards fast
computation of certiﬁed robustness for ReLU networks.
In Proceedings of the 35th International Conference on
Machine Learning, 2018a.
Weng, T.-W., Zhang, H., Chen, P.-Y., Yi, J., Su, D., Gao, Y.,
Hsieh, C.-J., and Daniel, L. Evaluating the robustness of
neural networks: An extreme value theory approach. In
International Conference on Learning Representations,
2018b.
Wong, E. and Kolter, J. Z. Provable defenses against adver-
sarial examples via the convex outer adversarial polytope.
In Proceedings of the 35th International Conference on
Machine Learning, 2018.
Wong, E., Schmidt, F., Metzen, J. H., and Kolter, J. Z.
Scaling provable adversarial defenses. In Advances in
Neural Information Processing Systems 31, 2018.
Zantedeschi, V., Nicolae, M.-I., and Rawat, A. Efﬁcient de-
fenses against adversarial attacks. Proceedings of the 10th
ACM Workshop on Artiﬁcial Intelligence and Security -
AISec 17, 2017.
Zhang, H., Weng, T.-W., Chen, P.-Y., Hsieh, C.-J., and
Daniel, L. Efﬁcient neural network robustness certiﬁca-
tion with general activation functions. In Advances in
Neural Information Processing Systems 31. 2018.


**Table 12 from page 11**

| 0                                                            | 1                                                             |
|:-------------------------------------------------------------|:--------------------------------------------------------------|
| Certiﬁed Adversarial Robustness via Randomized Smoothing     |                                                               |
| Singh, G., Gehr, T., Mirman, M., P¨uschel, M., and Vechev,   | Weng, T.-W., Zhang, H., Chen, P.-Y., Yi, J., Su, D., Gao, Y., |
| M.                                                           | Hsieh, C.-J., and Daniel, L. Evaluating the robustness of     |
| Fast and effective robustness certiﬁcation.                  |                                                               |
| In Ad-                                                       |                                                               |
| vances                                                       | neural networks: An extreme value theory approach.            |
| in Neural                                                    | In                                                            |
| Information Processing Systems 31.                           |                                                               |
| 2018.                                                        | International Conference on Learning Representations,         |
|                                                              | 2018b.                                                        |
| Smilkov, D., Thorat, N., Kim, B., Vigas, F., and Wattenberg, |                                                               |
|                                                              | Wong, E. and Kolter, J. Z. Provable defenses against adver-   |
| M. Smoothgrad: removing noise by adding noise. arXiv         |                                                               |
|                                                              | sarial examples via the convex outer adversarial polytope.    |
| preprint arXiv:1706.03825, 2017.                             |                                                               |
|                                                              | In Proceedings of the 35th International Conference on        |
| Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan,   | Machine Learning, 2018.                                       |
| D., Goodfellow,                                              |                                                               |
| I., and Fergus, R.                                           |                                                               |
| Intriguing proper-                                           |                                                               |
|                                                              | Wong, E., Schmidt, F., Metzen,                                |
|                                                              | J. H., and Kolter,                                            |
|                                                              | J. Z.                                                         |
| ties of neural networks.                                     |                                                               |
| In International Conference on                               |                                                               |
|                                                              | Scaling provable adversarial defenses.                        |
|                                                              | In Advances in                                                |
| Learning Representations, 2014.                              |                                                               |
|                                                              | Neural Information Processing Systems 31, 2018.               |
| Tjeng, V., Xiao, K. Y., and Tedrake, R. Evaluating robust-   |                                                               |
|                                                              | Zantedeschi, V., Nicolae, M.-I., and Rawat, A. Efﬁcient de-   |
| ness of neural networks with mixed integer programming.      |                                                               |
|                                                              | fenses against adversarial attacks. Proceedings of the 10th   |
| In International Conference on Learning Representations,     |                                                               |
|                                                              | ACM Workshop on Artiﬁcial Intelligence and Security -         |
| 2019. URL https://openreview.net/forum?                      |                                                               |
|                                                              | AISec 17, 2017.                                               |
| id=HyGIdiRqtm.                                               |                                                               |
|                                                              | Zhang, H., Weng, T.-W., Chen, P.-Y., Hsieh, C.-J., and        |
| Tsipras, D., Santurkar, S., Engstrom, L., Turner, A., and    |                                                               |
|                                                              | Daniel, L. Efﬁcient neural network robustness certiﬁca-       |
| Madry, A. Robustness may be at odds with accuracy.           |                                                               |
| In                                                           |                                                               |
|                                                              | tion with general activation functions.                       |
|                                                              | In Advances in                                                |
| International Conference on Learning Representations,        |                                                               |
|                                                              | Neural Information Processing Systems 31. 2018.               |
| 2019. URL https://openreview.net/forum?                      |                                                               |
| id=SyxAb30cY7.                                               |                                                               |
| Tsuzuku, Y., Sato, I., and Sugiyama, M. Lipschitz-margin     |                                                               |
| training: Scalable certiﬁcation of perturbation invariance   |                                                               |
| for deep neural networks.                                    |                                                               |
| In Advances in Neural Infor-                                 |                                                               |
| mation Processing Systems 31. 2018.                          |                                                               |
| Uesato, J., O’Donoghue, B., Kohli, P., and van den Oord,     |                                                               |
| A. Adversarial risk and the dangers of evaluating against    |                                                               |
| weak attacks.                                                |                                                               |
| In Proceedings of the 35th International                     |                                                               |
| Conference on Machine Learning, 2018.                        |                                                               |
| Wang, S., Chen, Y., Abdou, A., and Jana, S. Mixtrain: Scal-  |                                                               |
| able training of formally robust neural networks. arXiv      |                                                               |
| preprint arXiv:1811.02625, 2018a.                            |                                                               |
| Wang, S., Pei, K., Whitehouse, J., Yang, J., and Jana, S.    |                                                               |
| Efﬁcient formal safety analysis of neural networks.          |                                                               |
| In                                                           |                                                               |
| Advances in Neural Information Processing Systems 31.        |                                                               |
| 2018b.                                                       |                                                               |
| Webb, S., Rainforth, T., Teh, Y. W., and Kumar, M. P.        |                                                               |
| In-                                                          |                                                               |
| Statistical                                                  |                                                               |
| veriﬁcation                                                  |                                                               |
| of                                                           |                                                               |
| neural                                                       |                                                               |
| networks.                                                    |                                                               |
| In                                                           |                                                               |
| ternational Conference on Learning Representations,          |                                                               |
| 2019. URL https://openreview.net/forum?                      |                                                               |
| id=S1xcx3C5FX.                                               |                                                               |
| Weng, L., Zhang, H., Chen, H., Song, Z., Hsieh, C.-J.,       |                                                               |
| Daniel, L., Boning, D., and Dhillon,                         |                                                               |
| I.                                                           |                                                               |
| Towards fast                                                 |                                                               |
| computation of certiﬁed robustness for ReLU networks.        |                                                               |
| In Proceedings of the 35th International Conference on       |                                                               |
| Machine Learning, 2018a.                                     |                                                               |



## Page 12

Certiﬁed Adversarial Robustness via Randomized Smoothing
A. Proofs of Theorems 1 and 2
Here we provide the complete proofs for Theorem 1 and Theorem 2. We ﬁst prove the following lemma, which is essentially
a restatement of the Neyman-Pearson lemma (Neyman & Pearson, 1933) from statistical hypothesis testing.
Lemma 3 (Neyman-Pearson). Let X and Y be random variables in Rd with densities µX and µY . Let h : Rd →{0, 1}
be a random or deterministic function. Then:
1. If S =
n
z ∈Rd : µY (z)
µX(z) ≤t
o
for some t > 0 and P(h(X) = 1) ≥P(X ∈S), then P(h(Y ) = 1) ≥P(Y ∈S).
2. If S =
n
z ∈Rd : µY (z)
µX(z) ≥t
o
for some t > 0 and P(h(X) = 1) ≤P(X ∈S), then P(h(Y ) = 1) ≤P(Y ∈S).
Proof. Without loss of generality, we assume that h is random and write h(1|x) for the probability that h(x) = 1.
First we prove part 1. We denote the complement of S as Sc.
P(h(Y ) = 1) −P(Y ∈S) =
Z
Rd h(1|z) µY (z)dz −
Z
S
µY (z)dz
=
Z
Sc h(1|z)µY (z)dz +
Z
S
h(1|z)µY (z)dz

−
Z
S
h(1|z)µY (z)dz +
Z
S
h(0|z)µY (z)dz

=
Z
Sc h(1|z)µY (z)dz −
Z
S
h(0|z)µY (z)dz
≥t
Z
Sc h(1|z)µX(z)dz −
Z
S
h(0|z)µX(z)

= t
Z
Sc h(1|z)µX(z)dz +
Z
S
h(1|z)µX(z)dz −
Z
S
h(1|z)µX(z)dz −
Z
S
h(0|z)µX(z)

= t
Z
Rd h(1|z)µX(z)dz −
Z
S
µX(z)dz

= t [P(h(X) = 1) −P(X ∈S)]
≥0
The inequality in the middle is due to the fact that µY (z) ≤t µX(z) ∀z ∈S and µY (z) > t µX(z) ∀z ∈Sc. The inequality
at the end is because both terms in the product are non-negative by assumption.
The proof for part 2 is virtually identical, except both “≥” become “≤.”
Remark: connection to statistical hypothesis testing.
Part 2 of Lemma 3 is known in the ﬁeld of statistical hypothesis
testing as the Neyman-Pearson Lemma (Neyman & Pearson, 1933). The hypothesis testing problem is this: we are given a
sample that comes from one of two distributions over Rd: either the null distribution X or the alternative distribution Y .
We would like to identify which distribution the sample came from. It is worse to say “Y ” when the true answer is “X”
than to say “X” when the true answer is “Y .” Therefore we seek a (potentially randomized) procedure h : Rd →{0, 1}
which returns “Y ” when the sample really came from X with probability no greater than some failure rate α. In particular,
out of all such rules h, we would like the uniformly most powerful one h∗, i.e. the rule which is most likely to correctly
say “Y ” when the sample really came from Y . Neyman & Pearson (1933) showed that h∗is the rule which returns “Y ”
deterministically on the set S∗= {z ∈Rd : µY (z)
µX(z) ≥t} for whichever t makes P(X ∈S∗) = α. In other words, to state
this in a form that looks like Part 2 of Lemma 3: if h is a different rule with P(h(X) = 1) ≤α, then h∗is more powerful
than h, i.e. P(h(Y ) = 1) ≤P(Y ∈S∗).
Now we state the special case of Lemma 3 for when X and Y are isotropic Gaussians.
Lemma 4 (Neyman-Pearson for Gaussians with different means). Let X ∼N(x, σ2I) and Y ∼N(x + δ, σ2I). Let
h : Rd →{0, 1} be any deterministic or random function. Then:
1. If S =

z ∈Rd : δT z ≤β
	
for some β and P(h(X) = 1) ≥P(X ∈S), then P(h(Y ) = 1) ≥P(Y ∈S)


**Table 13 from page 12**

| 0                                                                                                                           |
|:----------------------------------------------------------------------------------------------------------------------------|
| Certiﬁed Adversarial Robustness via Randomized Smoothing                                                                    |
| A. Proofs of Theorems 1 and 2                                                                                               |
| Here we provide the complete proofs for Theorem 1 and Theorem 2. We ﬁst prove the following lemma, which is essentially     |
| a restatement of the Neyman-Pearson lemma (Neyman & Pearson, 1933) from statistical hypothesis testing.                     |
| Lemma 3 (Neyman-Pearson). Let X and Y be random variables in Rd with densities µX and µY . Let h : Rd → {0, 1}              |
| be a random or deterministic function. Then:                                                                                |
| (cid:110)                                                                                                                   |
| (cid:111)                                                                                                                   |
| 1.                                                                                                                          |
| If S =                                                                                                                      |
| z ∈ Rd : µY (z)                                                                                                             |
| for some t > 0 and P(h(X) = 1) ≥ P(X ∈ S), then P(h(Y ) = 1) ≥ P(Y ∈ S).                                                    |
| µX (z) ≤ t                                                                                                                  |
| (cid:110)                                                                                                                   |
| (cid:111)                                                                                                                   |
| 2.                                                                                                                          |
| If S =                                                                                                                      |
| z ∈ Rd : µY (z)                                                                                                             |
| for some t > 0 and P(h(X) = 1) ≤ P(X ∈ S), then P(h(Y ) = 1) ≤ P(Y ∈ S).                                                    |
| µX (z) ≥ t                                                                                                                  |
| Proof. Without loss of generality, we assume that h is random and write h(1|x) for the probability that h(x) = 1.           |
| First we prove part 1. We denote the complement of S as Sc.                                                                 |
| (cid:90)                                                                                                                    |
| (cid:90)                                                                                                                    |
| P(h(Y ) = 1) − P(Y ∈ S) =                                                                                                   |
| h(1|z) µY (z)dz −                                                                                                           |
| µY (z)dz                                                                                                                    |
| Rd                                                                                                                          |
| S                                                                                                                           |
| (cid:21)                                                                                                                    |
| (cid:21)                                                                                                                    |
| (cid:20)(cid:90)                                                                                                            |
| (cid:90)                                                                                                                    |
| (cid:20)(cid:90)                                                                                                            |
| (cid:90)                                                                                                                    |
| =                                                                                                                           |
| −                                                                                                                           |
| h(1|z)µY (z)dz +                                                                                                            |
| h(1|z)µY (z)dz                                                                                                              |
| h(1|z)µY (z)dz +                                                                                                            |
| h(0|z)µY (z)dz                                                                                                              |
| Sc                                                                                                                          |
| S                                                                                                                           |
| S                                                                                                                           |
| S                                                                                                                           |
| (cid:90)                                                                                                                    |
| (cid:90)                                                                                                                    |
| =                                                                                                                           |
| h(1|z)µY (z)dz −                                                                                                            |
| h(0|z)µY (z)dz                                                                                                              |
| Sc                                                                                                                          |
| S                                                                                                                           |
| (cid:21)                                                                                                                    |
| (cid:20)(cid:90)                                                                                                            |
| (cid:90)                                                                                                                    |
| ≥ t                                                                                                                         |
| h(1|z)µX (z)dz −                                                                                                            |
| h(0|z)µX (z)                                                                                                                |
| Sc                                                                                                                          |
| S                                                                                                                           |
| (cid:21)                                                                                                                    |
| (cid:20)(cid:90)                                                                                                            |
| (cid:90)                                                                                                                    |
| (cid:90)                                                                                                                    |
| (cid:90)                                                                                                                    |
| = t                                                                                                                         |
| h(1|z)µX (z)dz +                                                                                                            |
| h(1|z)µX (z)dz −                                                                                                            |
| h(1|z)µX (z)dz −                                                                                                            |
| h(0|z)µX (z)                                                                                                                |
| Sc                                                                                                                          |
| S                                                                                                                           |
| S                                                                                                                           |
| S                                                                                                                           |
| (cid:21)                                                                                                                    |
| (cid:20)(cid:90)                                                                                                            |
| (cid:90)                                                                                                                    |
| = t                                                                                                                         |
| h(1|z)µX (z)dz −                                                                                                            |
| µX (z)dz                                                                                                                    |
| Rd                                                                                                                          |
| S                                                                                                                           |
| = t [P(h(X) = 1) − P(X ∈ S)]                                                                                                |
| ≥ 0                                                                                                                         |
| The inequality in the middle is due to the fact that µY (z) ≤ t µX (z) ∀z ∈ S and µY (z) > t µX (z) ∀z ∈ Sc. The inequality |
| at the end is because both terms in the product are non-negative by assumption.                                             |
| The proof for part 2 is virtually identical, except both “≥” become “≤.”                                                    |
| Remark: connection to statistical hypothesis testing.                                                                       |
| Part 2 of Lemma 3 is known in the ﬁeld of statistical hypothesis                                                            |
| testing as the Neyman-Pearson Lemma (Neyman & Pearson, 1933). The hypothesis testing problem is this: we are given a        |
| sample that comes from one of two distributions over Rd: either the null distribution X or the alternative distribution Y . |
| We would like to identify which distribution the sample came from.                                                          |
| It is worse to say “Y ” when the true answer is “X”                                                                         |
| than to say “X” when the true answer is “Y .” Therefore we seek a (potentially randomized) procedure h : Rd → {0, 1}        |
| which returns “Y ” when the sample really came from X with probability no greater than some failure rate α. In particular,  |
| out of all such rules h, we would like the uniformly most powerful one h∗, i.e.                                             |
| the rule which is most likely to correctly                                                                                  |
| say “Y ” when the sample really came from Y . Neyman & Pearson (1933) showed that h∗ is the rule which returns “Y ”         |
| deterministically on the set S∗ = {z ∈ Rd : µY (z)                                                                          |
| µX (z) ≥ t} for whichever t makes P(X ∈ S∗) = α. In other words, to state                                                   |
| this in a form that looks like Part 2 of Lemma 3:                                                                           |
| if h is a different rule with P(h(X) = 1) ≤ α, then h∗ is more powerful                                                     |
| than h, i.e. P(h(Y ) = 1) ≤ P(Y ∈ S∗).                                                                                      |
| Now we state the special case of Lemma 3 for when X and Y are isotropic Gaussians.                                          |
| Lemma 4 (Neyman-Pearson for Gaussians with different means). Let X ∼ N (x, σ2I) and Y ∼ N (x + δ, σ2I). Let                 |
| h : Rd → {0, 1} be any deterministic or random function. Then:                                                              |
| 1.                                                                                                                          |
| If S = (cid:8)z ∈ Rd : δT z ≤ β(cid:9) for some β and P(h(X) = 1) ≥ P(X ∈ S), then P(h(Y ) = 1) ≥ P(Y ∈ S)                  |



## Page 13

Certiﬁed Adversarial Robustness via Randomized Smoothing
2. If S =

z ∈Rd : δT z ≥β
	
for some β and P(h(X) = 1) ≤P(X ∈S), then P(h(Y ) = 1) ≤P(Y ∈S)
Proof. This lemma is the special case of Lemma 3 when X and Y are isotropic Gaussians with means x and x + δ.
By Lemma 3 it sufﬁces to simply show that for any β, there is some t > 0 for which:
{z : δT z ≤β} =

z : µY (z)
µX(z) ≤t

and
{z : δT z ≥β} =

z : µY (z)
µX(z) ≥t

(5)
The likelihood ratio for this choice of X and Y turns out to be:
µY (z)
µX(z) =
exp

−1
2σ2
Pd
i=1(zi −(xi + δi))2)

exp

−1
2σ2
Pd
i=1(zi −xi)2

= exp
 
1
2σ2
d
X
i=1
2ziδi −δ2
i −2xiδi
!
= exp(aδT z + b)
where a > 0 and b are constants w.r.t z, speciﬁcally a =
1
σ2 and b = −(2δT x+∥δ∥2)
2σ2
.
Therefore, given any β we may take t = exp(aβ + b), noticing that
δT z ≤β ⇐⇒exp(aδT z + b) ≤t
δT z ≥β ⇐⇒exp(aδT z + b) ≥t
Finally, we prove Theorem 1 and Theorem 2.
Theorem 1 (restated).
Let f : Rd →Y be any deterministic or random function. Let ε ∼N(0, σ2I). Let g(x) =
arg maxc P(f(x + ε) = c). Suppose that for a speciﬁc x ∈Rd, there exist cA ∈Y and pA, pB ∈[0, 1] such that:
P(f(x + ε) = cA) ≥pA ≥pB ≥max
c̸=cA P(f(x + ε) = c)
(6)
Then g(x + δ) = cA for all ∥δ∥2 < R, where
R = σ
2 (Φ−1(pA) −Φ−1(pB))
(7)
Proof. To show that g(x + δ) = cA, it follows from the deﬁnition of g that we need to show that
P(f(x + δ + ε) = cA) > max
cB̸=cA P(f(x + δ + ε) = cB)
We will prove that P(f(x + δ + ε) = cA) > P(f(x + δ + ε) = cB) for every class cB ̸= cA. Fix one such class cB without
loss of generality.
For brevity, deﬁne the random variables
X := x + ε = N(x, σ2I)
Y := x + δ + ε = N(x + δ, σ2I)
In this notation, we know from (6) that
P(f(X) = cA) ≥pA
and
P(f(X) = cB) ≤pB
(8)


**Table 14 from page 13**

| 0                                                                                                             | 1                                                                                                                               |
|:--------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------|
| Certiﬁed Adversarial Robustness via Randomized Smoothing                                                      |                                                                                                                                 |
| 2.                                                                                                            |                                                                                                                                 |
| If S = (cid:8)z ∈ Rd : δT z ≥ β(cid:9) for some β and P(h(X) = 1) ≤ P(X ∈ S), then P(h(Y ) = 1) ≤ P(Y ∈ S)    |                                                                                                                                 |
| Proof. This lemma is the special case of Lemma 3 when X and Y are isotropic Gaussians with means x and x + δ. |                                                                                                                                 |
| By Lemma 3 it sufﬁces to simply show that for any β, there is some t > 0 for which:                           |                                                                                                                                 |
| (cid:26)                                                                                                      |                                                                                                                                 |
| (cid:27)                                                                                                      |                                                                                                                                 |
| (cid:26)                                                                                                      |                                                                                                                                 |
| (cid:27)                                                                                                      |                                                                                                                                 |
| µY (z)                                                                                                        |                                                                                                                                 |
| µY (z)                                                                                                        |                                                                                                                                 |
| {z : δT z ≤ β} =                                                                                              | (5)                                                                                                                             |
| z :                                                                                                           |                                                                                                                                 |
| ≤ t                                                                                                           |                                                                                                                                 |
| {z : δT z ≥ β} =                                                                                              |                                                                                                                                 |
| z :                                                                                                           |                                                                                                                                 |
| ≥ t                                                                                                           |                                                                                                                                 |
| and                                                                                                           |                                                                                                                                 |
| µX (z)                                                                                                        |                                                                                                                                 |
| µX (z)                                                                                                        |                                                                                                                                 |
| The likelihood ratio for this choice of X and Y turns out to be:                                              |                                                                                                                                 |
| (cid:16)                                                                                                      |                                                                                                                                 |
| (cid:17)                                                                                                      |                                                                                                                                 |
| (cid:80)d                                                                                                     |                                                                                                                                 |
| exp                                                                                                           |                                                                                                                                 |
| − 1                                                                                                           |                                                                                                                                 |
| i=1(zi − (xi + δi))2)                                                                                         |                                                                                                                                 |
| 2σ2                                                                                                           |                                                                                                                                 |
| µY (z)                                                                                                        |                                                                                                                                 |
| =                                                                                                             |                                                                                                                                 |
| (cid:17)                                                                                                      |                                                                                                                                 |
| (cid:16)                                                                                                      |                                                                                                                                 |
| (cid:80)d                                                                                                     |                                                                                                                                 |
| µX (z)                                                                                                        |                                                                                                                                 |
| exp                                                                                                           |                                                                                                                                 |
| − 1                                                                                                           |                                                                                                                                 |
| i=1(zi − xi)2                                                                                                 |                                                                                                                                 |
| 2σ2                                                                                                           |                                                                                                                                 |
| (cid:32)                                                                                                      |                                                                                                                                 |
| (cid:33)                                                                                                      |                                                                                                                                 |
| 1                                                                                                             |                                                                                                                                 |
| d(cid:88) i                                                                                                   |                                                                                                                                 |
| = exp                                                                                                         |                                                                                                                                 |
| 2ziδi − δ2                                                                                                    |                                                                                                                                 |
| i − 2xiδi                                                                                                     |                                                                                                                                 |
| 2σ2                                                                                                           |                                                                                                                                 |
| =1                                                                                                            |                                                                                                                                 |
| = exp(aδT z + b)                                                                                              |                                                                                                                                 |
| where a > 0 and b are constants w.r.t z, speciﬁcally a = 1                                                    |                                                                                                                                 |
| σ2 and b = −(2δT x+(cid:107)δ(cid:107)2)                                                                      |                                                                                                                                 |
| .                                                                                                             |                                                                                                                                 |
| Therefore, given any β we may take t = exp(aβ + b), noticing that                                             |                                                                                                                                 |
| δT z ≤ β ⇐⇒ exp(aδT z + b) ≤ t                                                                                |                                                                                                                                 |
| δT z ≥ β ⇐⇒ exp(aδT z + b) ≥ t                                                                                |                                                                                                                                 |
| Finally, we prove Theorem 1 and Theorem 2.                                                                    |                                                                                                                                 |
| Let f                                                                                                         | : Rd → Y be any deterministic or random function. Let ε ∼ N (0, σ2I). Let g(x) =                                                |
| Theorem 1 (restated).                                                                                         |                                                                                                                                 |
| P(f (x + ε) = c). Suppose that for a speciﬁc x ∈ Rd, there exist cA ∈ Y and pA, pB ∈ [0, 1] such that:        |                                                                                                                                 |
| arg maxc                                                                                                      |                                                                                                                                 |
| P(f (x + ε) = c)                                                                                              | (6)                                                                                                                             |
| P(f (x + ε) = cA) ≥ pA ≥ pB ≥ max                                                                             |                                                                                                                                 |
| c(cid:54)=cA                                                                                                  |                                                                                                                                 |
| Then g(x + δ) = cA for all (cid:107)δ(cid:107)2 < R, where                                                    |                                                                                                                                 |
| σ 2                                                                                                           | (7)                                                                                                                             |
| R =                                                                                                           |                                                                                                                                 |
| (Φ−1(pA) − Φ−1(pB))                                                                                           |                                                                                                                                 |
| Proof. To show that g(x + δ) = cA, it follows from the deﬁnition of g that we need to show that               |                                                                                                                                 |
| P(f (x + δ + ε) = cA) > max                                                                                   |                                                                                                                                 |
| P(f (x + δ + ε) = cB)                                                                                         |                                                                                                                                 |
| cB (cid:54)=cA                                                                                                |                                                                                                                                 |
|                                                                                                               | We will prove that P(f (x + δ + ε) = cA) > P(f (x + δ + ε) = cB) for every class cB (cid:54)= cA. Fix one such class cB without |
| loss of generality.                                                                                           |                                                                                                                                 |
| For brevity, deﬁne the random variables                                                                       |                                                                                                                                 |
| X := x + ε = N (x, σ2I)                                                                                       |                                                                                                                                 |
| Y := x + δ + ε = N (x + δ, σ2I)                                                                               |                                                                                                                                 |
| In this notation, we know from (6) that                                                                       |                                                                                                                                 |
| and                                                                                                           | (8)                                                                                                                             |
| P(f (X) = cA) ≥ pA                                                                                            |                                                                                                                                 |
| P(f (X) = cB) ≤ pB                                                                                            |                                                                                                                                 |



## Page 14

Certiﬁed Adversarial Robustness via Randomized Smoothing
x + δ
x
x + δ
x
Figure 9. Illustration of the proof of Theorem 1. The solid line concentric circles are the density level sets of X := x + ε; the dashed
line concentric circles are the level sets of Y := x + δ + ε. The set A is in blue and the set B is in red. The ﬁgure on the left depicts
a situation where P(Y ∈A) > P(Y ∈B), and hence g(x + δ) may equal cA. The ﬁgure on the right depicts a situation where
P(Y ∈A) < P(Y ∈B) and hence g(x + δ) ̸= cA.
and our goal is to show that
P(f(Y ) = cA) > P(f(Y ) = cB)
(9)
Deﬁne the half-spaces:
A := {z : δT (z −x) ≤σ∥δ∥Φ−1(pA)}
B := {z : δT (z −x) ≥σ∥δ∥Φ−1(1 −pB)}
Algebra (deferred to the end) shows that P(X ∈A) = pA. Therefore, by (8) we know that P(f(X) = cA) ≥P(X ∈A).
Hence we may apply Lemma 4 with h(z) := 1[f(z) = cA] to conclude:
P(f(Y ) = cA) ≥P(Y ∈A)
(10)
Similarly, algebra shows that P(X ∈B) = pB. Therefore, by (8) we know that P(f(X) = cB) ≤P(X ∈B). Hence we
may apply Lemma 4 with h(z) := 1[f(z) = cB] to conclude:
P(f(Y ) = cB) ≤P(Y ∈B)
(11)
To guarantee (9), we see from (10, 11) that it sufﬁces to show that P(Y ∈A) > P(Y ∈B), as this step completes the chain
of inequalities
P(f(Y ) = cA) ≥P(Y ∈A) > P(Y ∈B) ≥P(f(Y ) = cB)
(12)
We can compute the following:
P(Y ∈A) = Φ

Φ−1(pA) −∥δ∥
σ

(13)
P(Y ∈B) = Φ

Φ−1(pB) + ∥δ∥
σ

(14)
Finally, algebra shows that P(Y ∈A) > P(Y ∈B) if and only if:
∥δ∥< σ
2 (Φ−1(pA) −Φ−1(pB))
(15)
which recovers the theorem statement.


**Table 15 from page 14**

| 0                                                                                                                                        | 1                                                                                                                                         | 2    |
|:-----------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------|:-----|
|                                                                                                                                          | Certiﬁed Adversarial Robustness via Randomized Smoothing                                                                                  |      |
|                                                                                                                                          | x + δ                                                                                                                                     |      |
|                                                                                                                                          | x + δ                                                                                                                                     |      |
|                                                                                                                                          | x                                                                                                                                         |      |
|                                                                                                                                          | x                                                                                                                                         |      |
| Figure 9. Illustration of the proof of Theorem 1. The solid line concentric circles are the density level sets of X := x + ε; the dashed |                                                                                                                                           |      |
|                                                                                                                                          | line concentric circles are the level sets of Y := x + δ + ε. The set A is in blue and the set B is in red. The ﬁgure on the left depicts |      |
|                                                                                                                                          | a situation where P(Y ∈ A) > P(Y ∈ B), and hence g(x + δ) may equal cA. The ﬁgure on the right depicts a situation where                  |      |
| P(Y ∈ A) < P(Y ∈ B) and hence g(x + δ) (cid:54)= cA.                                                                                     |                                                                                                                                           |      |
| and our goal is to show that                                                                                                             |                                                                                                                                           |      |
|                                                                                                                                          | P(f (Y ) = cA) > P(f (Y ) = cB)                                                                                                           | (9)  |
| Deﬁne the half-spaces:                                                                                                                   |                                                                                                                                           |      |
|                                                                                                                                          | A := {z : δT (z − x) ≤ σ(cid:107)δ(cid:107)Φ−1(pA)}                                                                                       |      |
|                                                                                                                                          | B := {z : δT (z − x) ≥ σ(cid:107)δ(cid:107)Φ−1(1 − pB)}                                                                                   |      |
|                                                                                                                                          | Algebra (deferred to the end) shows that P(X ∈ A) = pA. Therefore, by (8) we know that P(f (X) = cA) ≥ P(X ∈ A).                          |      |
| Hence we may apply Lemma 4 with h(z) := 1[f (z) = cA] to conclude:                                                                       |                                                                                                                                           |      |
|                                                                                                                                          | P(f (Y ) = cA) ≥ P(Y ∈ A)                                                                                                                 | (10) |
|                                                                                                                                          | Similarly, algebra shows that P(X ∈ B) = pB. Therefore, by (8) we know that P(f (X) = cB) ≤ P(X ∈ B). Hence we                            |      |
| may apply Lemma 4 with h(z) := 1[f (z) = cB] to conclude:                                                                                |                                                                                                                                           |      |
|                                                                                                                                          | P(f (Y ) = cB) ≤ P(Y ∈ B)                                                                                                                 | (11) |
|                                                                                                                                          | To guarantee (9), we see from (10, 11) that it sufﬁces to show that P(Y ∈ A) > P(Y ∈ B), as this step completes the chain                 |      |
| of inequalities                                                                                                                          |                                                                                                                                           |      |
|                                                                                                                                          | P(f (Y ) = cA) ≥ P(Y ∈ A) > P(Y ∈ B) ≥ P(f (Y ) = cB)                                                                                     | (12) |
| We can compute the following:                                                                                                            |                                                                                                                                           |      |
|                                                                                                                                          | (cid:18)                                                                                                                                  |      |
|                                                                                                                                          | (cid:19)                                                                                                                                  |      |
|                                                                                                                                          | (cid:107)δ(cid:107)                                                                                                                       |      |
|                                                                                                                                          | P(Y ∈ A) = Φ                                                                                                                              | (13) |
|                                                                                                                                          | Φ−1(pA) −                                                                                                                                 |      |
|                                                                                                                                          | σ                                                                                                                                         |      |
|                                                                                                                                          | (cid:18)                                                                                                                                  |      |
|                                                                                                                                          | (cid:19)                                                                                                                                  |      |
|                                                                                                                                          | (cid:107)δ(cid:107)                                                                                                                       |      |
|                                                                                                                                          | P(Y ∈ B) = Φ                                                                                                                              | (14) |
|                                                                                                                                          | Φ−1(pB) +                                                                                                                                 |      |
|                                                                                                                                          | σ                                                                                                                                         |      |
| Finally, algebra shows that P(Y ∈ A) > P(Y ∈ B) if and only if:                                                                          |                                                                                                                                           |      |
|                                                                                                                                          | σ 2                                                                                                                                       | (15) |
|                                                                                                                                          | (cid:107)δ(cid:107) <                                                                                                                     |      |
|                                                                                                                                          | (Φ−1(pA) − Φ−1(pB))                                                                                                                       |      |
| which recovers the theorem statement.                                                                                                    |                                                                                                                                           |      |



## Page 15

Certiﬁed Adversarial Robustness via Randomized Smoothing
We now restate and prove Theorem 2, which shows that the bound in Theorem 1 is tight. The assumption below in Theorem
2 that pA + pB ≤1 is mild: given any pA and pB which do not satisfy this condition, one could have always redeﬁned
pB ←1 −pA to obtain a Theorem 1 guarantee with a larger certiﬁed radius, so there is no reason to invoke Theorem 1
unless pA + pB ≤1.
Theorem 2 (restated). Assune pA + pB ≤1. For any perturbation δ ∈Rd with ∥δ∥2 > R, there exists a base classiﬁer f ∗
consistent with the observed class probabilities (6) such that if f ∗is the base classiﬁer for g, then g(x + δ) ̸= cA.
Proof. We re-use notation from the preceding proof.
Pick any class cB arbitrarily. Deﬁne A and B as above, and consider the function
f ∗(x) :=





cA
if x ∈A
cB
if x ∈B
other classes
otherwise
This function is well-deﬁned, since A ∩B = ∅provided that pA + pB ≤1.
By construction, the function f ∗satisﬁes (6) with equalities, since
P(f ∗(x + ε) = cA) = P(X ∈A) = pA
P(f ∗(x + ε) = cB) = P(X ∈B) = pB
It follows from (13) and (14) that
P(Y ∈A) < P(Y ∈B) ⇐⇒∥δ∥2 > R
By assumption, ∥δ∥2 > R, so P(Y ∈A) < P(Y ∈B), or equivalently,
P(f ∗(x + δ + ε) = cA) < P(f ∗(x + δ + ε) = cB)
Therefore, if f ∗is the base classiﬁer for g, then g(x + δ) ̸= cA.
A.0.1. DEFERRED ALGEBRA
Claim. P(X ∈A) = pA
Proof. Recall that X ∼N(x, σ2I) and A = {z : δT (z −x) ≤σ∥δ∥Φ−1(pA)}.
P(X ∈A) = P(δT (X −x) ≤σ∥δ∥Φ−1(pA))
= P(δT N(0, σ2I) ≤σ∥δ∥Φ−1(pA))
= P(σ∥δ∥Z ≤σ∥δ∥Φ−1(pA))
(Z ∼N(0, 1))
= Φ(Φ−1(pA))
= pA
Claim. P(X ∈B) = pB
Proof. Recall that X ∼N(x, σ2I) and B = {z : δT (z −x) ≤σ∥δ∥Φ−1(1 −pB)}.
P(X ∈A) = P(δT (X −x) ≥σ∥δ∥Φ−1(1 −pB))
= P(δT N(0, σ2I) ≥σ∥δ∥Φ−1(1 −pB))
= P(σ∥δ∥Z ≥σ∥δ∥Φ−1(1 −pB))
(Z ∼N(0, 1))
= P(Z ≥Φ−1(1 −pB))
= 1 −Φ(Φ−1(1 −pB))
= pB


**Table 16 from page 15**

| 0                                                                                                                                      |
|:---------------------------------------------------------------------------------------------------------------------------------------|
| Certiﬁed Adversarial Robustness via Randomized Smoothing                                                                               |
| We now restate and prove Theorem 2, which shows that the bound in Theorem 1 is tight. The assumption below in Theorem                  |
| 2 that pA + pB ≤ 1 is mild: given any pA and pB which do not satisfy this condition, one could have always redeﬁned                    |
| pB ← 1 − pA to obtain a Theorem 1 guarantee with a larger certiﬁed radius, so there is no reason to invoke Theorem 1                   |
| unless pA + pB ≤ 1.                                                                                                                    |
| Theorem 2 (restated). Assune pA + pB ≤ 1. For any perturbation δ ∈ Rd with (cid:107)δ(cid:107)2 > R, there exists a base classiﬁer f ∗ |
| consistent with the observed class probabilities (6) such that if f ∗ is the base classiﬁer for g, then g(x + δ) (cid:54)= cA.         |
| Proof. We re-use notation from the preceding proof.                                                                                    |
| Pick any class cB arbitrarily. Deﬁne A and B as above, and consider the function                                                       |



## Page 16

Certiﬁed Adversarial Robustness via Randomized Smoothing
Claim. P(Y ∈A) = Φ

Φ−1(pA) −∥δ∥
σ

Proof. Recall that Y ∼N(x + δ, σ2I) and A = {z : δT (z −x) ≤σ∥δ∥Φ−1(pA)}.
P(Y ∈A) = P(δT (Y −x) ≤σ∥δ∥Φ−1(pA))
= P(δT N(0, σ2I) + ∥δ∥2 ≤σ∥δ∥Φ−1(pA))
= P(σ∥δ∥Z ≤σ∥δ∥Φ−1(pA) −∥δ∥2)
(Z ∼N(0, 1))
= P

Z ≤Φ−1(pA) −∥δ∥
σ

= Φ

Φ−1(pA) −∥δ∥
σ

Claim. P(Y ∈B) = Φ

Φ−1(pB) + ∥δ∥
σ

Proof. Recall that Y ∼N(x + δ, σ2I) and B = {z : δT (z −x) ≥σ∥δ∥Φ−1(1 −pB)}.
P(Y ∈B) = P(δT (Y −x) ≥σ∥δ∥Φ−1(1 −pB))
= P(δT N(0, σ2I) + ∥δ∥2 ≥σ∥δ∥Φ−1(1 −pB))
= P(σ∥δ∥Z + ∥δ∥2 ≥σ∥δ∥Φ−1(1 −pB))
(Z ∼N(0, 1))
= P

Z ≥Φ−1(1 −pB) −∥δ∥
σ

= P

Z ≤Φ−1(pB) + ∥δ∥
σ

= Φ

Φ−1(pB) + ∥δ∥
σ



**Table 17 from page 16**

| 0                                                                                                 | 1                                                                                   | 2              |
|:--------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:---------------|
|                                                                                                   | Certiﬁed Adversarial Robustness via Randomized Smoothing                            |                |
|                                                                                                   | (cid:17)                                                                            |                |
|                                                                                                   | (cid:16)                                                                            |                |
| Claim. P(Y ∈ A) = Φ                                                                               | Φ−1(pA) − (cid:107)δ(cid:107)                                                       |                |
|                                                                                                   | σ                                                                                   |                |
| Proof. Recall that Y ∼ N (x + δ, σ2I) and A = {z : δT (z − x) ≤ σ(cid:107)δ(cid:107)Φ−1(pA)}.     |                                                                                     |                |
|                                                                                                   | P(Y ∈ A) = P(δT (Y − x) ≤ σ(cid:107)δ(cid:107)Φ−1(pA))                              |                |
|                                                                                                   | = P(δT N (0, σ2I) + (cid:107)δ(cid:107)2 ≤ σ(cid:107)δ(cid:107)Φ−1(pA))             |                |
|                                                                                                   | = P(σ(cid:107)δ(cid:107)Z ≤ σ(cid:107)δ(cid:107)Φ−1(pA) − (cid:107)δ(cid:107)2)     | (Z ∼ N (0, 1)) |
|                                                                                                   | (cid:18)                                                                            |                |
|                                                                                                   | (cid:19)                                                                            |                |
|                                                                                                   | (cid:107)δ(cid:107)                                                                 |                |
|                                                                                                   | = P                                                                                 |                |
|                                                                                                   | Z ≤ Φ−1(pA) −                                                                       |                |
|                                                                                                   | σ                                                                                   |                |
|                                                                                                   | (cid:18)                                                                            |                |
|                                                                                                   | (cid:19)                                                                            |                |
|                                                                                                   | (cid:107)δ(cid:107)                                                                 |                |
|                                                                                                   | = Φ                                                                                 |                |
|                                                                                                   | Φ−1(pA) −                                                                           |                |
|                                                                                                   | σ                                                                                   |                |
|                                                                                                   | (cid:17)                                                                            |                |
|                                                                                                   | (cid:16)                                                                            |                |
| Claim. P(Y ∈ B) = Φ                                                                               | Φ−1(pB) + (cid:107)δ(cid:107)                                                       |                |
|                                                                                                   | σ                                                                                   |                |
| Proof. Recall that Y ∼ N (x + δ, σ2I) and B = {z : δT (z − x) ≥ σ(cid:107)δ(cid:107)Φ−1(1 − pB)}. |                                                                                     |                |
|                                                                                                   | P(Y ∈ B) = P(δT (Y − x) ≥ σ(cid:107)δ(cid:107)Φ−1(1 − pB))                          |                |
|                                                                                                   | = P(δT N (0, σ2I) + (cid:107)δ(cid:107)2 ≥ σ(cid:107)δ(cid:107)Φ−1(1 − pB))         |                |
|                                                                                                   | = P(σ(cid:107)δ(cid:107)Z + (cid:107)δ(cid:107)2 ≥ σ(cid:107)δ(cid:107)Φ−1(1 − pB)) | (Z ∼ N (0, 1)) |
|                                                                                                   | (cid:18)                                                                            |                |
|                                                                                                   | (cid:19)                                                                            |                |
|                                                                                                   | (cid:107)δ(cid:107)                                                                 |                |
|                                                                                                   | = P                                                                                 |                |
|                                                                                                   | Z ≥ Φ−1(1 − pB) −                                                                   |                |
|                                                                                                   | σ                                                                                   |                |
|                                                                                                   | (cid:18)                                                                            |                |
|                                                                                                   | (cid:19)                                                                            |                |
|                                                                                                   | (cid:107)δ(cid:107)                                                                 |                |
|                                                                                                   | = P                                                                                 |                |
|                                                                                                   | Z ≤ Φ−1(pB) +                                                                       |                |
|                                                                                                   | σ                                                                                   |                |
|                                                                                                   | (cid:18)                                                                            |                |
|                                                                                                   | (cid:19)                                                                            |                |
|                                                                                                   | (cid:107)δ(cid:107)                                                                 |                |
|                                                                                                   | = Φ                                                                                 |                |
|                                                                                                   | Φ−1(pB) +                                                                           |                |
|                                                                                                   | σ                                                                                   |                |



## Page 17

Certiﬁed Adversarial Robustness via Randomized Smoothing
B. Smoothing a two-class linear classiﬁer
In this appendix, we analyze what happens when the base classiﬁer f is a two-class linear classiﬁer f(x) = sign(wT x + b).
To match the deﬁnition of g, we take sign(·) to be undeﬁned when its argument is zero.
x
x
Figure 10. Illustration of Proposition 3. A binary linear classiﬁer f(x) = sign(wT x + b) partitions Rd into two half-spaces, drawn here
in blue and red. An isotropic Gaussian N(x, σ2I) will put more mass on whichever half-space its center x lies in: in the ﬁgure on
the left, x is in the blue half-space and N(x, σ2I) puts more mass on the blue than on red. In the ﬁgure on the right, x is in the red
half-space and N(x, σ2I) puts more mass on red than on blue. Since the smoothed classiﬁer’s prediction g(x) is deﬁned to be whichever
half-space N(x, σ2I) puts more mass in, and the base classiﬁer’s prediction f(x) is deﬁned to be whichever half-space x is in, we have
that g(x) = f(x) for all x.
Our ﬁrst result is that when f is a two-class linear classiﬁer, the smoothed classiﬁer g is identical to the base classiﬁer f.
Proposition 3. If f is a two-class linear classiﬁer f(x) = sign(wT x + b), and g is the smoothed version of f with any σ,
then g(x) = f(x) for any x (where f is deﬁned).
Proof. From the deﬁnition of g,
g(x) = 1 ⇐⇒Pε(f(x + ε) = 1) > 1
2
(ε ∼N(0, σ2I))
⇐⇒Pε
 sign(wT (x + ε) + b) = 1

> 1
2
⇐⇒Pε
 wT x + wT ε + b ≥0

> 1
2
⇐⇒P
 σ∥w∥Z ≥−wT x −b

> 1
2
(Z ∼N(0, 1))
⇐⇒P

Z ≤wT x + b
σ∥w∥

> 1
2
⇐⇒wT x + b
σ∥w∥
> 0
⇐⇒wT x + b > 0
⇐⇒f(x) = 1
A similar calculation shows that g(x) = −1 ⇐⇒f(x) = −1.
A two-class linear classiﬁer f(x) = sign(wT x + b) is already certiﬁable: the distance from any point x to the decision
boundary is (wT x+b)/∥w∥2, and no distance with ℓ2 norm strictly less than this distance can possibly change f’s prediction.
Let g be a smoothed version of f. By Proposition 3, g is identical to f, so it follows that g is truly robust around any
input x within the ℓ2 radius (wT x + b)/∥w∥2. We now show that Theorem 1 will certify this radius, rather than a smaller,
over-conservative radius.


**Table 18 from page 17**

| 0                                                      | 1   | 2                |
|:-------------------------------------------------------|:----|:-----------------|
| 1 2                                                    |     | (ε ∼ N (0, σ2I)) |
| g(x) = 1 ⇐⇒ Pε(f (x + ε) = 1) >                        |     |                  |
| (cid:0)sign(wT (x + ε) + b) = 1(cid:1) >               | 1 2 |                  |
| ⇐⇒ Pε                                                  |     |                  |
| 1 2                                                    |     |                  |
| (cid:0)wT x + wT ε + b ≥ 0(cid:1) >                    |     |                  |
| ⇐⇒ Pε                                                  |     |                  |
| 1 2                                                    |     | (Z ∼ N (0, 1))   |
| ⇐⇒ P (cid:0)σ(cid:107)w(cid:107)Z ≥ −wT x − b(cid:1) > |     |                  |
| (cid:18)                                               |     |                  |
| (cid:19)                                               |     |                  |
| wT x + b                                               |     |                  |
| 1 2                                                    |     |                  |
| ⇐⇒ P                                                   |     |                  |
| Z ≤                                                    |     |                  |
| >                                                      |     |                  |
| σ(cid:107)w(cid:107)                                   |     |                  |
| wT x + b                                               |     |                  |
| ⇐⇒                                                     |     |                  |
| > 0                                                    |     |                  |
| σ(cid:107)w(cid:107)                                   |     |                  |
| ⇐⇒ wT x + b > 0                                        |     |                  |
| ⇐⇒ f (x) = 1                                           |     |                  |



## Page 18

Certiﬁed Adversarial Robustness via Randomized Smoothing
Proposition 4. If f is a two-class linear classiﬁer f(x) = sign(wT x + b), and g is the smoothed version of f with any
σ, then invoking Theorem 1 at any x (where f is deﬁned) with pA = pA and pB = pB will yield the certiﬁed radius
R = |wT x+b|
∥w∥
.
Proof. In binary classiﬁcation, pA = 1 −pB, so Theorem 1 returns R = σΦ−1(pA).
We have:
pA = Pε(f(x + ε) = g(x))
= Pε(sign(wT (x + ε) + b) = sign(wT x + b))
(By Proposition 3, g(x) = f(x))
= Pε(sign(wT x + σ∥w∥Z + b) = sign(wT x + b))
There are two cases: if wT x + b > 0, then
pA = Pε(wT x + σ∥w∥Z + b > 0)
= Pε

Z > −wT x −b
σ∥w∥

= Pε

Z < wT x + b
σ∥w∥

= Φ
wT x + b
σ∥w∥

On the other hand, if wT x + b < 0, then
pA = Pε(wT x + σ∥w∥Z + b < 0)
= Pε

Z < −wT x −b
σ∥w∥

= Φ
−wT x −b
σ∥w∥

In either case, we have:
pA = Φ
|wT x + b|
σ∥w∥

Therefore, the bound in Theorem 1 returns a radius of
R = σΦ−1(pA)
= |wT x + b|
∥w∥
The previous two propositions imply that when f is a two-class linear classiﬁer, the Theorem 1 bound is “tight” in the sense
that there always exists a class-changing perturbation just beyond the certiﬁed radius.3
Proposition 5. Let f be a two-class linear classiﬁer f(x) = sign(wT x + b), let g be the smoothed version of f for some σ,
let x be any point (where f is deﬁned), and let R be the radius certiﬁed around x by Theorem 1. Then for any radius r > R,
there exists a perturbation δ with ∥δ∥2 = r for which g(x + δ) ̸= g(x).
3Note that this is a different sense of “tight” than the sense in which Theorem 2 proves that Theorem 1 is tight. Theorem 2 proves that
for any ﬁxed perturbation δ outside the radius certiﬁed by Theorem 1, there exists a base classiﬁer f for which g(x + δ) ̸= g(x). In
contrast, Proposition 5 proves that for any ﬁxed binary linear base classiﬁer f, there exists a perturbation δ just outside the radius certiﬁed
by Theorem 1 for which g(x + δ) ̸= g(x).


**Table 19 from page 18**

| 0                                                                                                                                                |
|:-------------------------------------------------------------------------------------------------------------------------------------------------|
| Certiﬁed Adversarial Robustness via Randomized Smoothing                                                                                         |
| If f is a two-class linear classiﬁer f (x) = sign(wT x + b), and g is the smoothed version of f with any                                         |
| Proposition 4.                                                                                                                                   |
| σ,                                                                                                                                               |
| then invoking Theorem 1 at any x (where f is deﬁned) with pA = pA and pB = pB will yield the certiﬁed radius                                     |
| .                                                                                                                                                |
| R = |wT x+b|                                                                                                                                     |
| (cid:107)w(cid:107)                                                                                                                              |
| Proof.                                                                                                                                           |
| In binary classiﬁcation, pA = 1 − pB, so Theorem 1 returns R = σΦ−1(pA).                                                                         |
| We have:                                                                                                                                         |
| pA = Pε(f (x + ε) = g(x))                                                                                                                        |
| (By Proposition 3, g(x) = f (x))                                                                                                                 |
| = Pε(sign(wT (x + ε) + b) = sign(wT x + b))                                                                                                      |
| = Pε(sign(wT x + σ(cid:107)w(cid:107)Z + b) = sign(wT x + b))                                                                                    |
| There are two cases:                                                                                                                             |
| if wT x + b > 0, then                                                                                                                            |
| pA = Pε(wT x + σ(cid:107)w(cid:107)Z + b > 0)                                                                                                    |
| (cid:19)                                                                                                                                         |
| (cid:18)                                                                                                                                         |
| −wT x − b                                                                                                                                        |
| Z >                                                                                                                                              |
| = Pε                                                                                                                                             |
| σ(cid:107)w(cid:107)                                                                                                                             |
| (cid:19)                                                                                                                                         |
| (cid:18)                                                                                                                                         |
| wT x + b                                                                                                                                         |
| Z <                                                                                                                                              |
| = Pε                                                                                                                                             |
| σ(cid:107)w(cid:107)                                                                                                                             |
| (cid:19)                                                                                                                                         |
| (cid:18) wT x + b                                                                                                                                |
| = Φ                                                                                                                                              |
| σ(cid:107)w(cid:107)                                                                                                                             |
| On the other hand, if wT x + b < 0, then                                                                                                         |
| pA = Pε(wT x + σ(cid:107)w(cid:107)Z + b < 0)                                                                                                    |
| (cid:19)                                                                                                                                         |
| (cid:18)                                                                                                                                         |
| −wT x − b                                                                                                                                        |
| Z <                                                                                                                                              |
| = Pε                                                                                                                                             |
| σ(cid:107)w(cid:107)                                                                                                                             |
| (cid:19)                                                                                                                                         |
| (cid:18) −wT x − b                                                                                                                               |
| = Φ                                                                                                                                              |
| σ(cid:107)w(cid:107)                                                                                                                             |
| In either case, we have:                                                                                                                         |
| (cid:19)                                                                                                                                         |
| (cid:18) |wT x + b|                                                                                                                              |
| pA = Φ                                                                                                                                           |
| σ(cid:107)w(cid:107)                                                                                                                             |
| Therefore, the bound in Theorem 1 returns a radius of                                                                                            |
| R = σΦ−1(pA)                                                                                                                                     |
| |wT x + b|                                                                                                                                       |
| =                                                                                                                                                |
| (cid:107)w(cid:107)                                                                                                                              |
| The previous two propositions imply that when f is a two-class linear classiﬁer, the Theorem 1 bound is “tight” in the sense                     |
| that there always exists a class-changing perturbation just beyond the certiﬁed radius.3                                                         |
| Proposition 5. Let f be a two-class linear classiﬁer f (x) = sign(wT x + b), let g be the smoothed version of f for some σ,                      |
| let x be any point (where f is deﬁned), and let R be the radius certiﬁed around x by Theorem 1. Then for any radius r > R,                       |
| there exists a perturbation δ with (cid:107)δ(cid:107)2 = r for which g(x + δ) (cid:54)= g(x).                                                   |
| 3Note that this is a different sense of “tight” than the sense in which Theorem 2 proves that Theorem 1 is tight. Theorem 2 proves that          |
| for any ﬁxed perturbation δ outside the radius certiﬁed by Theorem 1, there exists a base classiﬁer f for which g(x + δ) (cid:54)= g(x).         |
| In                                                                                                                                               |
| contrast, Proposition 5 proves that for any ﬁxed binary linear base classiﬁer f , there exists a perturbation δ just outside the radius certiﬁed |
| by Theorem 1 for which g(x + δ) (cid:54)= g(x).                                                                                                  |



## Page 19

Certiﬁed Adversarial Robustness via Randomized Smoothing
Proof. By Proposition 3 it sufﬁces to show that there exists some perturbation δ with ∥δ∥2 = r for which f(x + δ) ̸= f(x).
By Proposition 4, we know that R = |wT x+b|
∥w∥2 .
If wT x + b > 0, consider the perturbation δ = −
w
∥w∥2 r. This perturbation satisﬁes ∥δ∥2 = r and
wT (x + δ) + b = wT x + b + wT δ
= wT x + b −∥w∥2r
< wT x + b −∥w∥2R
= wT x + b −|wT x + b|
= wT x + b −(wT x + b)
= 0
implying that f(x + δ) = −1.
Likewise, if wT x+b < 0, then consider the perturbation δ =
w
∥w∥2 r. This perturbation satisﬁes ∥δ∥2 = r and f(x+δ) = −1.
x
x
x + δ
Figure 11. Left: Illustration of of Proposition 4. The red/blue half-spaces are the decision regions of both the base classiﬁer f and the
smoothed classiﬁer g. (Since the base classiﬁer is binary linear, g = f everywhere.) The black circle is the robustness radius R certiﬁed by
Theorem 1. Right: Illustration of Proposition 5. For any r > R, there exists a perturbation δ with ∥δ∥2 = r for which g(x + δ) ̸= g(x).
This special property of two-class linear classiﬁers is not true in general. In fact, it is possible to construct situations where
g’s prediction around some point x0 is robust at radius ∞, yet Theorem 1 only certiﬁes a radius of τ, where τ is arbitrarily
close to zero.
Proposition 6. For any τ > 0, there exists a base classiﬁer f and an input x0 for which the corresponding smoothed
classiﬁer g is robust around x0 at radius ∞, yet Theorem 1 only certiﬁes a radius of τ around x0.
Proof. Let t = −Φ−1( 1
2Φ(τ)) and consider the following base classiﬁer:
f(x) =





1
if x < −t
−1
if −t ≤x ≤t
1
if x > t
Let g be the smoothed version of f with σ = 1. We will show that g(x) = 1 everywhere, implying that g’s prediction is
robust around x0 = 0 with radius ∞. Yet Theorem 1 only certiﬁes a radius of τ around x0.


**Table 20 from page 19**

| 0                                                                                                                                                              |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Certiﬁed Adversarial Robustness via Randomized Smoothing                                                                                                       |
| Proof. By Proposition 3 it sufﬁces to show that there exists some perturbation δ with (cid:107)δ(cid:107)2 = r for which f (x + δ) (cid:54)= f (x).            |
| By Proposition 4, we know that R = |wT x+b|                                                                                                                    |
| .                                                                                                                                                              |
| (cid:107)w(cid:107)2                                                                                                                                           |
| If wT x + b > 0, consider the perturbation δ = − w                                                                                                             |
| r. This perturbation satisﬁes (cid:107)δ(cid:107)2 = r and                                                                                                     |
| (cid:107)w(cid:107)2                                                                                                                                           |
| wT (x + δ) + b = wT x + b + wT δ                                                                                                                               |
| = wT x + b − (cid:107)w(cid:107)2r                                                                                                                             |
| < wT x + b − (cid:107)w(cid:107)2R                                                                                                                             |
| = wT x + b − |wT x + b|                                                                                                                                        |
| = wT x + b − (wT x + b)                                                                                                                                        |
| = 0                                                                                                                                                            |
| implying that f (x + δ) = −1.                                                                                                                                  |
| w                                                                                                                                                              |
| Likewise, if wT x+b < 0, then consider the perturbation δ =                                                                                                    |
| r. This perturbation satisﬁes (cid:107)δ(cid:107)2 = r and f (x+δ) = −1.                                                                                       |
| (cid:107)w(cid:107)2                                                                                                                                           |
| x + δ                                                                                                                                                          |
| x                                                                                                                                                              |
| x                                                                                                                                                              |
| Figure 11. Left: Illustration of of Proposition 4. The red/blue half-spaces are the decision regions of both the base classiﬁer f and the                      |
| smoothed classiﬁer g. (Since the base classiﬁer is binary linear, g = f everywhere.) The black circle is the robustness radius R certiﬁed by                   |
| Theorem 1. Right: Illustration of Proposition 5. For any r > R, there exists a perturbation δ with (cid:107)δ(cid:107)2 = r for which g(x + δ) (cid:54)= g(x). |
| This special property of two-class linear classiﬁers is not true in general. In fact, it is possible to construct situations where                             |
| g’s prediction around some point x0 is robust at radius ∞, yet Theorem 1 only certiﬁes a radius of τ , where τ is arbitrarily                                  |
| close to zero.                                                                                                                                                 |
| Proposition 6. For any τ > 0,                                                                                                                                  |
| there exists a base classiﬁer f and an input x0 for which the corresponding smoothed                                                                           |
| classiﬁer g is robust around x0 at radius ∞, yet Theorem 1 only certiﬁes a radius of τ around x0.                                                              |
| Proof. Let t = −Φ−1( 1                                                                                                                                         |
| 2 Φ(τ )) and consider the following base classiﬁer:                                                                                                            |
| 1                                                                                                                                                              |
| if x < −t                                                                                                                                                      |
|                                                                                                                                                           |
| f (x) =                                                                                                                                                        |
| −1                                                                                                                                                             |
| if − t ≤ x ≤ t                                                                                                                                                 |
| 1                                                                                                                                                              |
| if x > t                                                                                                                                                       |
| Let g be the smoothed version of f with σ = 1. We will show that g(x) = 1 everywhere, implying that g’s prediction is                                          |
| robust around x0 = 0 with radius ∞. Yet Theorem 1 only certiﬁes a radius of τ around x0.                                                                       |



## Page 20

Certiﬁed Adversarial Robustness via Randomized Smoothing
Let Z ∼N(0, 1). For any x, we have:
P(f(x + ε) = −1) = P(−t ≤x + ε ≤t)
= P[−t −x ≤Z ≤t −x]
≤P[−t ≤Z ≤t]
(apply Lemma 5 below with ℓ= −t −x)
= 1 −2Φ(−t)
= 1 −Φ(τ)
< 1
2.
Therefore, g(x) = 1 for all x.
Meanwhile, at x0 = 0, we have:
P(f(x0 + ε) = 1) = P(f(ε) = 1)
= P(Z < −t or Z > t)
= 2Φ(−t)
= Φ(τ),
so by Theorem 1, the certiﬁed radius around x0 is R = τ.
The proof of Proposition 6 employed the following lemma, which formalizes the visually obvious fact that out of all intervals
of some ﬁxed width 2t, the interval with maximal mass under the standard normal distribution Z is the interval [−t, t].
Lemma 5. Let Z ∼N(0, 1). For any ℓ∈R, t > 0, we have P(ℓ≤Z ≤ℓ+ 2t) ≤P(−t ≤Z ≤t).
Proof. Let φ be the PDF of the standard normal distribution. Since φ is symmetric about the origin (i.e. φ(x) = φ(−x) ∀x),
P(−t ≤Z ≤t) = 2
Z t
0
φ(x)dx.
There are two cases to consider:
Case 1: The interval [ℓ, ℓ+ 2t] is entirely positive, i.e. ℓ≥0, or [ℓ, ℓ+ 2t] is entirely negative, i.e. ℓ+ 2t ≤0.
First, we use the fact that φ is symmetric about the origin to rewrite P(ℓ≤Z ≤ℓ+ 2t) as the probability that Z falls in a
non-negative interval [a, a + 2t] for some a.
Speciﬁcally, if ℓ≥0, then let a = ℓ. Else, if ℓ+ 2t ≤0, then let a = −(ℓ+ 2t). We therefore have:
P(ℓ≤Z ≤ℓ+ 2t) = P(a ≤Z ≤a + 2t).
Therefore:
P(−t ≤Z ≤t) −P(ℓ≤Z ≤ℓ+ 2t) =
Z t
0
φ(x)dx −
Z a+t
a
φ(x)dx +
Z t
0
φ(x)dx −
Z a+2t
a+t
φ(x)dx
=
Z a+t
a
φ(x −a)dx −
Z a+t
a
φ(x)dx +
Z a+2t
a+t
φ(x −a −t)dx −
Z a+2t
a+t
φ(x)dx
=
Z a+t
a
[φ(x −a) −φ(x)] dx +
Z a+2t
a+t
[φ(x −a −t) −φ(x)] dx
≥
Z a+t
a
0 dx +
Z a+2t
a+t
0 dx
= 0


**Table 21 from page 20**

| 0                                                         | 1                                                                                                                                                                  | 2                      | 3             | 4                                            |
|:----------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------|:--------------|:---------------------------------------------|
|                                                           | Certiﬁed Adversarial Robustness via Randomized Smoothing                                                                                                           |                        |               |                                              |
| Let Z ∼ N (0, 1). For any x, we have:                     |                                                                                                                                                                    |                        |               |                                              |
|                                                           | P(f (x + ε) = −1) = P(−t ≤ x + ε ≤ t)                                                                                                                              |                        |               |                                              |
|                                                           | = P[−t − x ≤ Z ≤ t − x]                                                                                                                                            |                        |               |                                              |
|                                                           | ≤ P[−t ≤ Z ≤ t]                                                                                                                                                    |                        |               | (apply Lemma 5 below with (cid:96) = −t − x) |
|                                                           | = 1 − 2Φ(−t)                                                                                                                                                       |                        |               |                                              |
|                                                           | = 1 − Φ(τ )                                                                                                                                                        |                        |               |                                              |
|                                                           | 1 2                                                                                                                                                                |                        |               |                                              |
|                                                           | <                                                                                                                                                                  |                        |               |                                              |
|                                                           | .                                                                                                                                                                  |                        |               |                                              |
| Therefore, g(x) = 1 for all x.                            |                                                                                                                                                                    |                        |               |                                              |
| Meanwhile, at x0 = 0, we have:                            |                                                                                                                                                                    |                        |               |                                              |
|                                                           | P(f (x0 + ε) = 1) = P(f (ε) = 1)                                                                                                                                   |                        |               |                                              |
|                                                           |                                                                                                                                                                    | = P(Z < −t or Z > t)   |               |                                              |
|                                                           |                                                                                                                                                                    | = 2Φ(−t)               |               |                                              |
|                                                           |                                                                                                                                                                    | = Φ(τ ),               |               |                                              |
| so by Theorem 1, the certiﬁed radius around x0 is R = τ . |                                                                                                                                                                    |                        |               |                                              |
|                                                           | The proof of Proposition 6 employed the following lemma, which formalizes the visually obvious fact that out of all intervals                                      |                        |               |                                              |
|                                                           | of some ﬁxed width 2t, the interval with maximal mass under the standard normal distribution Z is the interval [−t, t].                                            |                        |               |                                              |
|                                                           | Lemma 5. Let Z ∼ N (0, 1). For any (cid:96) ∈ R, t > 0, we have P((cid:96) ≤ Z ≤ (cid:96) + 2t) ≤ P(−t ≤ Z ≤ t).                                                   |                        |               |                                              |
|                                                           | Proof. Let φ be the PDF of the standard normal distribution. Since φ is symmetric about the origin (i.e. φ(x) = φ(−x) ∀x),                                         |                        |               |                                              |
|                                                           |                                                                                                                                                                    | (cid:90) t             |               |                                              |
|                                                           | P(−t ≤ Z ≤ t) = 2                                                                                                                                                  |                        | φ(x)dx.       |                                              |
|                                                           |                                                                                                                                                                    | 0                      |               |                                              |
| There are two cases to consider:                          |                                                                                                                                                                    |                        |               |                                              |
|                                                           | Case 1: The interval [(cid:96), (cid:96) + 2t] is entirely positive, i.e. (cid:96) ≥ 0, or [(cid:96), (cid:96) + 2t] is entirely negative, i.e. (cid:96) + 2t ≤ 0. |                        |               |                                              |
|                                                           | First, we use the fact that φ is symmetric about the origin to rewrite P((cid:96) ≤ Z ≤ (cid:96) + 2t) as the probability that Z falls in a                        |                        |               |                                              |
| non-negative interval [a, a + 2t] for some a.             |                                                                                                                                                                    |                        |               |                                              |
|                                                           | Speciﬁcally, if (cid:96) ≥ 0, then let a = (cid:96). Else, if (cid:96) + 2t ≤ 0, then let a = −((cid:96) + 2t). We therefore have:                                 |                        |               |                                              |
|                                                           | P((cid:96) ≤ Z ≤ (cid:96) + 2t) = P(a ≤ Z ≤ a + 2t).                                                                                                               |                        |               |                                              |
| Therefore:                                                |                                                                                                                                                                    |                        |               |                                              |
|                                                           | (cid:90) t                                                                                                                                                         |                        | (cid:90) t    | (cid:90) a+2t                                |
|                                                           | (cid:90) a+t                                                                                                                                                       |                        |               |                                              |
| P(−t ≤ Z ≤ t) − P((cid:96) ≤ Z ≤ (cid:96) + 2t) =         | φ(x)dx −                                                                                                                                                           | φ(x)dx +               | φ(x)dx −      | φ(x)dx                                       |
|                                                           | 0                                                                                                                                                                  |                        | 0             | a+t                                          |
|                                                           | a                                                                                                                                                                  |                        |               |                                              |
|                                                           | (cid:90) a+t                                                                                                                                                       | (cid:90) a+t           |               | (cid:90) a+2t                                |
|                                                           |                                                                                                                                                                    |                        |               | (cid:90) a+2t                                |
| =                                                         | φ(x − a)dx −                                                                                                                                                       |                        | φ(x)dx +      | φ(x − a − t)dx −                             |
|                                                           |                                                                                                                                                                    |                        |               | φ(x)dx                                       |
|                                                           | a                                                                                                                                                                  | a                      |               | a+t                                          |
|                                                           |                                                                                                                                                                    |                        |               | a+t                                          |
|                                                           | (cid:90) a+t                                                                                                                                                       |                        | (cid:90) a+2t |                                              |
| =                                                         |                                                                                                                                                                    | [φ(x − a) − φ(x)] dx + |               | [φ(x − a − t) − φ(x)] dx                     |
|                                                           | a                                                                                                                                                                  |                        | a+t           |                                              |
|                                                           | (cid:90) a+t                                                                                                                                                       |                        |               |                                              |
|                                                           | (cid:90) a+2t                                                                                                                                                      |                        |               |                                              |
| ≥                                                         | 0 dx +                                                                                                                                                             | 0 dx                   |               |                                              |
|                                                           | a                                                                                                                                                                  |                        |               |                                              |
|                                                           | a+t                                                                                                                                                                |                        |               |                                              |
|                                                           | = 0                                                                                                                                                                |                        |               |                                              |



## Page 21

Certiﬁed Adversarial Robustness via Randomized Smoothing
where the inequality is because φ is monotonically decreasing on [0, ∞).
Case 2: I is partly positive, partly negative, i.e. ℓ< 0 < ℓ+ 2t.
First, we use the fact that φ is symmetric about the origin to rewrite P(ℓ≤Z ≤ℓ+ 2t) as the sum of the probabilities that
Z falls in two non-negative intervals [0, a] and [0, b] for some a, b.
Speciﬁcally, let a = min(−ℓ, ℓ+ 2t) and b = max(−ℓ, ℓ+ 2t). We therefore have:
P(ℓ≤Z ≤ℓ+ 2t) = P(0 ≤Z ≤a) + P(0 ≤Z ≤b).
Note that by construction, a + b = 2t, and 0 ≤a ≤t and t ≤b ≤2t.
We have:
P(−t ≤Z ≤t) −P(ℓ≤Z ≤ℓ+ 2t) =
Z t
0
φ(x)dx −
Z a
0
φ(x)dx

−
"Z b
0
φ(x)dx −
Z t
0
φ(x)dx
!
=
Z t
a
φ(x)dx −
Z b
t
φ(x)dx
=
Z t
a
φ(x)dx −
Z 2t−a
t
φ(x)dx
=
Z t
a
φ(x)dx −
Z t
a
φ(x −a + t)dx
=
Z t
a
(φ(x) −φ(x −a + t))dx
≥
Z t
a
0 dx
= 0
where the inequality is again because φ is monotonically decreasing on [0, ∞).


**Table 22 from page 21**

| 0                                                                                                                                           |
|:--------------------------------------------------------------------------------------------------------------------------------------------|
| Certiﬁed Adversarial Robustness via Randomized Smoothing                                                                                    |
| where the inequality is because φ is monotonically decreasing on [0, ∞).                                                                    |
| Case 2: I is partly positive, partly negative, i.e. (cid:96) < 0 < (cid:96) + 2t.                                                           |
| First, we use the fact that φ is symmetric about the origin to rewrite P((cid:96) ≤ Z ≤ (cid:96) + 2t) as the sum of the probabilities that |
| Z falls in two non-negative intervals [0, a] and [0, b] for some a, b.                                                                      |
| Speciﬁcally, let a = min(−(cid:96), (cid:96) + 2t) and b = max(−(cid:96), (cid:96) + 2t). We therefore have:                                |
| P((cid:96) ≤ Z ≤ (cid:96) + 2t) = P(0 ≤ Z ≤ a) + P(0 ≤ Z ≤ b).                                                                              |
| Note that by construction, a + b = 2t, and 0 ≤ a ≤ t and t ≤ b ≤ 2t.                                                                        |
| We have:                                                                                                                                    |
| (cid:33)                                                                                                                                    |

**Table 23 from page 21**

| 0                                                                    | 1                  | 2                       | 3                  | 4        | 5          |
|:---------------------------------------------------------------------|:-------------------|:------------------------|:-------------------|:---------|:-----------|
| Note that by construction, a + b = 2t, and 0 ≤ a ≤ t and t ≤ b ≤ 2t. |                    |                         |                    |          |            |
|                                                                      |                    |                         |                    |          | (cid:33)   |
|                                                                      | (cid:18)(cid:90) t | (cid:21)                | (cid:34)(cid:90) b |          | (cid:90) t |
|                                                                      |                    | (cid:90) a              |                    |          |            |
| P(−t ≤ Z ≤ t) − P((cid:96) ≤ Z ≤ (cid:96) + 2t) =                    |                    | φ(x)dx                  | −                  | φ(x)dx − | φ(x)dx     |
|                                                                      |                    | φ(x)dx −                |                    |          |            |
|                                                                      |                    | 0                       |                    | 0        | 0          |
|                                                                      |                    | 0                       |                    |          |            |
|                                                                      | (cid:90) t         | (cid:90) b              |                    |          |            |
| =                                                                    |                    | φ(x)dx                  |                    |          |            |
|                                                                      |                    | φ(x)dx −                |                    |          |            |
|                                                                      | a                  | t                       |                    |          |            |
|                                                                      | (cid:90) t         | (cid:90) 2t−a           |                    |          |            |
| =                                                                    |                    | φ(x)dx −                |                    |          |            |
|                                                                      |                    | φ(x)dx                  |                    |          |            |
|                                                                      | a                  | t                       |                    |          |            |
|                                                                      | (cid:90) t         | (cid:90) t              |                    |          |            |
| =                                                                    |                    | φ(x)dx −                | φ(x − a + t)dx     |          |            |
|                                                                      | a                  | a                       |                    |          |            |
|                                                                      | (cid:90) t         |                         |                    |          |            |
| =                                                                    |                    | (φ(x) − φ(x − a + t))dx |                    |          |            |
|                                                                      | a                  |                         |                    |          |            |
|                                                                      | (cid:90) t         |                         |                    |          |            |
| ≥                                                                    |                    | 0 dx                    |                    |          |            |



## Page 22

Certiﬁed Adversarial Robustness via Randomized Smoothing
C. Practical algorithms
In this appendix, we elaborate on the prediction and certiﬁcation algorithms described in Section 3.2. The pseudocode in
Section 3.2 makes use of several helper functions:
• SAMPLEUNDERNOISE(f, x, num, σ) works as follows:
1. Draw num samples of noise, ε1 . . . εnum ∼N(0, σ2I).
2. Run the noisy images through the base classiﬁer f to obtain the predictions f(x + ε1), . . . , f(x + εnum).
3. Return the counts for each class, where the count for class c is deﬁned as Pnum
i=1 1[f(x + εi) = c].
• BINOMPVALUE(nA, nA+nB, p) returns the p-value of the two-sided hypothesis test that nA ∼Binomial(nA+nB, p).
Using scipy.stats.binom test, this can be implemented as: binom test(nA, nA + nB, p).
• LOWERCONFBOUND(k, n, 1 −α) returns a one-sided (1 −α) lower conﬁdence interval for the Binomial pa-
rameter p given that k ∼Binomial(n, p). In other words, it returns some number p for which p ≤p with prob-
ability at least 1 −α over the sampling of k ∼Binomial(n, p). Following Lecuyer et al. (2019), we chose to
use the Clopper-Pearson conﬁdence interval, which inverts the Binomial CDF (Clopper & Pearson, 1934). Using
statsmodels.stats.proportion.proportion confint, this can be implemented as
proportion_confint(k, n, alpha=2*alpha, method="beta")[0]
C.1. Prediction
The randomized algorithm given in pseudocode as PREDICT leverages the hypothesis test given in Hung & Fithian (2019)
for identifying the top category of a multinomial distribution. PREDICT has one tunable hyperparameter, α. When α is small,
PREDICT abstains frequently but rarely returns the wrong class. When α is large, PREDICT usually makes a prediction, but
may often return the wrong class.
We now prove that with high probability, PREDICT will either return g(x) or abstain.
Proposition 1 (restated). With probability at least 1 −α over the randomness in PREDICT, PREDICT will either abstain
or return g(x). (Equivalently: the probability that PREDICT returns a class other than g(x) is at most α.)
Proof. For notational convenience, deﬁne pc = P(f(x + ε) = c). Let cA = maxc pc. Notice that by deﬁnition, g(x) = cA.
We can describe the randomized procedure PREDICT as follows:
1. Sample a vector of class counts {nc}c∈Y from Multinomial({pc}c∈Y, n).
2. Let ˆcA = arg maxc nc be the class whose count is largest. Let nA and nB be the largest count and the second-largest
count, respectively.
3. If the p-value of the two-sided hypothesis test that nA is drawn from Binom
 nA + nB, 1
2

is less than α, then return
ˆcA. Else, abstain.
The quantities cA and the pc’s are ﬁxed but unknown, while the quantities ˆcA, the nc’s, nA, and nB are random.
We’d like to prove that the probability that PREDICT returns a class other than cA is at most α. PREDICT returns a class
other than cA if and only if (1) ˆcA ̸= cA and (2) PREDICT does not abstain.
We have:
P(PREDICT returns class ̸= cA) = P(ˆcA ̸= cA, PREDICT does not abstain)
= P(ˆcA ̸= cA) P(PREDICT does not abstain|ˆcA ̸= cA)
≤P(PREDICT does not abstain|ˆcA ̸= cA)


**Table 24 from page 22**

| 0                                                                                         | 1                                                                                                                | 2                                                   | 3                   |
|:------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------|:----------------------------------------------------|:--------------------|
|                                                                                           | 2. Run the noisy images through the base classiﬁer f to obtain the predictions f (x + ε1), . . . , f (x + εnum). |                                                     |                     |
| 3. Return the counts for each class, where the count for class c is deﬁned as (cid:80)num |                                                                                                                  |                                                     |                     |
|                                                                                           |                                                                                                                  | i=1 1[f (x + εi) = c].                              |                     |
|                                                                                           | • BINOMPVALUE(nA, nA +nB, p) returns the p-value of the two-sided hypothesis test that nA ∼ Binomial(nA +nB, p). |                                                     |                     |
| Using scipy.stats.binom test, this can be implemented as: binom test(nA,                  | nA                                                                                                               | +                                                   | p).                 |
|                                                                                           |                                                                                                                  | nB,                                                 |                     |
| • LOWERCONFBOUND(k, n, 1 − α)                                                             | returns a one-sided (1 − α) lower conﬁdence interval                                                             | for                                                 | the Binomial pa-    |
| rameter p given that k ∼ Binomial(n, p).                                                  | In other words,                                                                                                  | it returns some number p for which p ≤ p with prob- |                     |
| ability at                                                                                | the sampling of k ∼ Binomial(n, p).                                                                              |                                                     | (2019), we chose to |
| least 1 − α over                                                                          | Following Lecuyer et al.                                                                                         |                                                     |                     |
|                                                                                           | use the Clopper-Pearson conﬁdence interval, which inverts the Binomial CDF (Clopper & Pearson, 1934). Using      |                                                     |                     |
|                                                                                           | statsmodels.stats.proportion.proportion confint, this can be implemented as                                      |                                                     |                     |
| proportion_confint(k,                                                                     | method="beta")[0]                                                                                                |                                                     |                     |
| n,                                                                                        | alpha=2*alpha,                                                                                                   |                                                     |                     |



## Page 23

Certiﬁed Adversarial Robustness via Randomized Smoothing
Recall that PREDICT does not abstain if and only if the p-value of the two-sided hypothesis test that nA is drawn from
Binom(nA + nB, 1
2) is less than α. Theorem 1 in Hung & Fithian (2019) proves that the conditional probability that this
event occurs given that ˆcA ̸= cA is exactly α. That is,
P(PREDICT does not abstain|ˆcA ̸= cA) = α
Therefore, we have:
P(PREDICT returns class ̸= cA) ≤α
C.2. Certiﬁcation
The certiﬁcation task is: given some input x and a randomized smoothing classiﬁer described by (f, σ), return both (1) the
prediction g(x) and (2) a radius R in which this prediction is certiﬁed robust. This task requires identifying the class cA
with maximal weight in f(x + ε), estimating a lower bound pA on pA := P(f(x + ε) = cA) and estimating an upper bound
pB on pB := maxc̸=cA P(f(x + ε) = c) (Figure 1).
Suppose for simplicity that we already knew cA and needed to obtain pA. We could collect n samples of f(x + ε), count
how many times f(x + ε) = cA, and use a Binomial conﬁdence interval to obtain a lower bound on pA that holds with
probability at least 1 −α over the n samples.
However, estimating pA and pB while simultaneously identifying the top class cA is a little bit tricky, statistically speaking.
We propose a simple two-step procedure. First, use n0 samples from f(x + ε) to take a guess ˆcA at the identity of the top
class cA. In practice we observed that f(x + ε) tends to put most of its weight on the top class, so n0 can be set very small.
Second, use n samples from f(x + ε) to obtain some pA and pB for which pA ≤pA and pB ≥pB with probability at least
1 −α. We observed that it is much more typical for the mass of f(x + ε) not allocated to cA to be allocated entirely to one
runner-up class than to be allocated uniformly over all remaining classes. Therefore, the quantity 1 −pA is a reasonably
tight upper bound on pB. Hence, we simply set pB = 1 −pA, so our bound becomes
R = σ
2 (Φ−1(pA) −Φ−1(1 −pA))
= σ
2 (Φ−1(pA) + Φ−1(pA))
= σΦ−1(pA)
The full procedure is described in pseudocode as CERTIFY. If pA < 1
2, we abstain from making a certiﬁcation; this can
occur especially if ˆcA ̸= g(x), i.e. if we misidentify the top class using the ﬁrst n0 samples of f(x + ε).
Proposition 2 (restated). With probability at least 1 −α over the randomness in CERTIFY, if CERTIFY returns a class ˆcA
and a radius R (i.e. does not abstain), then we have the robustness guarantee
g(x + δ) = ˆcA
whenever
∥δ∥2 < R
Proof. From the contract of LOWERCONFBOUND, we know that with probability at least 1 −α over the sampling of
ε1 . . . εn, we have pA ≤P[f(x + ε) = ˆcA]. Notice that CERTIFY returns a class and radius only if pA > 1
2 (otherwise it
abstains). If pA ≤P[f(x + ε) = ˆcA] and 1
2 < pA, then we can invoke Theorem 1 with pB = 1 −pA to obtain the desired
guarantee.


**Table 25 from page 23**

| 0                                                                                                                               |
|:--------------------------------------------------------------------------------------------------------------------------------|
| Certiﬁed Adversarial Robustness via Randomized Smoothing                                                                        |
| Recall                                                                                                                          |
| that PREDICT does not abstain if and only if the p-value of the two-sided hypothesis test                                       |
| that nA is drawn from                                                                                                           |
| Binom(nA + nB, 1                                                                                                                |
| 2 ) is less than α. Theorem 1 in Hung & Fithian (2019) proves that the conditional probability that this                        |
| event occurs given that ˆcA (cid:54)= cA is exactly α. That is,                                                                 |
| P(PREDICT does not abstain|ˆcA (cid:54)= cA) = α                                                                                |
| Therefore, we have:                                                                                                             |
| P(PREDICT returns class                                                                                                         |
| (cid:54)= cA) ≤ α                                                                                                               |
| C.2. Certiﬁcation                                                                                                               |
| The certiﬁcation task is: given some input x and a randomized smoothing classiﬁer described by (f, σ), return both (1) the      |
| prediction g(x) and (2) a radius R in which this prediction is certiﬁed robust. This task requires identifying the class cA     |
| with maximal weight in f (x + ε), estimating a lower bound pA on pA := P(f (x + ε) = cA) and estimating an upper bound          |
| P(f (x + ε) = c) (Figure 1).                                                                                                    |
| pB on pB := maxc(cid:54)=cA                                                                                                     |
| Suppose for simplicity that we already knew cA and needed to obtain pA. We could collect n samples of f (x + ε), count          |
| how many times f (x + ε) = cA, and use a Binomial conﬁdence interval to obtain a lower bound on pA that holds with              |
| probability at least 1 − α over the n samples.                                                                                  |
| However, estimating pA and pB while simultaneously identifying the top class cA is a little bit tricky, statistically speaking. |
| We propose a simple two-step procedure. First, use n0 samples from f (x + ε) to take a guess ˆcA at the identity of the top     |
| class cA. In practice we observed that f (x + ε) tends to put most of its weight on the top class, so n0 can be set very small. |
| Second, use n samples from f (x + ε) to obtain some pA and pB for which pA ≤ pA and pB ≥ pB with probability at least           |
| 1 − α. We observed that it is much more typical for the mass of f (x + ε) not allocated to cA to be allocated entirely to one   |
| runner-up class than to be allocated uniformly over all remaining classes. Therefore, the quantity 1 − pA is a reasonably       |
| tight upper bound on pB. Hence, we simply set pB = 1 − pA, so our bound becomes                                                 |
| σ 2                                                                                                                             |
| R =                                                                                                                             |
| (Φ−1(pA) − Φ−1(1 − pA))                                                                                                         |
| σ 2                                                                                                                             |
| =                                                                                                                               |
| (Φ−1(pA) + Φ−1(pA))                                                                                                             |
| = σΦ−1(pA)                                                                                                                      |
| The full procedure is described in pseudocode as CERTIFY.                                                                       |
| If pA < 1                                                                                                                       |
| 2 , we abstain from making a certiﬁcation; this can                                                                             |
| occur especially if ˆcA (cid:54)= g(x), i.e.                                                                                    |
| if we misidentify the top class using the ﬁrst n0 samples of f (x + ε).                                                         |
| Proposition 2 (restated). With probability at least 1 − α over the randomness in CERTIFY, if CERTIFY returns a class ˆcA        |
| and a radius R (i.e. does not abstain), then we have the robustness guarantee                                                   |
| whenever                                                                                                                        |
| g(x + δ) = ˆcA                                                                                                                  |
| (cid:107)δ(cid:107)2 < R                                                                                                        |
| Proof. From the contract of LOWERCONFBOUND, we know that with probability at                                                    |
| least 1 − α over the sampling of                                                                                                |
| ε1 . . . εn, we have pA ≤ P[f (x + ε) = ˆcA]. Notice that CERTIFY returns a class and radius only if pA > 1                     |
| 2 (otherwise it                                                                                                                 |
| abstains). If pA ≤ P[f (x + ε) = ˆcA] and 1                                                                                     |
| 2 < pA, then we can invoke Theorem 1 with pB = 1 − pA to obtain the desired                                                     |
| guarantee.                                                                                                                      |



## Page 24

Certiﬁed Adversarial Robustness via Randomized Smoothing
D. Estimating the certiﬁed test-set accuracy
In this appendix, we show how to convert the “approximate certiﬁed test accuracy” considered in the main paper into a
lower bound on the true certiﬁed test accuracy that holds with high probability over the randomness in CERTIFY.
Consider a classiﬁer g, a test set S = {(x1, c1) . . . (xm, cm)}, and a radius r. For each example i ∈[m], let zi indicate
whether g’s prediction at xi is both correct and robust at radius r, i.e.
zi = 1[g(xi + δ) = ci ∀∥δ∥2 < r]
The certiﬁed test set accuracy of g at radius r is deﬁned as 1
m
Pm
i=1 zi. If g is a randomized smoothing classiﬁer, we cannot
compute this quantity exactly, but we can estimate a lower bound that holds with arbitrarily high probability over the
randomness in CERTIFY. In particular, suppose that we run CERTIFY with failure rate α on each example xi in the test set.
Let the Bernoulli random variable Yi denote the event that on example i, CERTIFY returns the correct label cA = ci and a
certiﬁed radius R which is greater than r. Let Y = Pm
i=1 Yi. In the main paper, we referred to Y/m as the “approximate
certiﬁed accuracy.” It is “approximate” because Yi = 1 does not mean that zi = 1. Rather, from Proposition 2, we know
the following: if zi = 0, then P(Yi = 1) ≤α. We now show how to exploit this fact to construct a one-sided conﬁdence
interval for the unobserved quantity 1
m
Pm
i=1 zi using the observed quantities Y and m.
Theorem 6. For any ρ > 0, with probability at least 1 −ρ over the randomness in CERTIFY,
1
m
m
X
i=1
zi ≥
1
1 −α
 
Y
m −α −
r
2α(1 −α) log(1/ρ)
m
−log(1/ρ)
3m
!
(16)
Proof. Let mgood = Pm
i=1 zi and mbad = Pm
i=1(1 −zi) be the number of test examples on which zi = 1 or zi = 0,
respectively. We model Yi ∼Bernoulli(pi), where pi is in general unknown. Let Ygood = P
i:zi=1 Yi and Ybad = P
i:zi=0 Yi.
The quantity of interest, the certiﬁed accuracy 1
m
Pm
i=1 zi, is equal to mgood/m. However, we only observe Y = Ygood+Ybad.
Note that if zi = 0, then pi ≤α, so we have E[Yi] = pi ≤α and assuming α ≤1
2, we have Var[Yi] = pi(1−pi) ≤α(1−α).
Since Ybad is a sum of mbad independent random variables each bounded between zero and one, with E[Ybad] ≤αmbad and
Var(Ybad) ≤mbadα(1 −α), Bernstein’s inequality (Blanchard, 2007) guarantees that with probability at least 1 −ρ over the
randomness in CERTIFY,
Ybad ≤αmbad +
p
2mbadα(1 −α) log(1/ρ) + log(1/ρ)
3
From now on, we manipulate this inequality — remember that it holds with probability at least 1 −ρ.
Since Y = Ygood + Ybad, may write
Ygood ≥Y −αmbad −
p
2mbadα(1 −α) log(1/ρ) −log(1/ρ)
3
Since mgood ≥Ygood, we may write
mgood ≥Y −αmbad −
p
2mbadα(1 −α) log(1/ρ) −log(1/ρ)
3
Since mgood + mbad = m, we may write
mgood ≥
1
1 −α

Y −α m −
p
2mbadα(1 −α) log(1/ρ) −log(1/ρ)
3

Finally, in order to make this conﬁdence interval depend only on observables, we use mbad ≤m to write
mgood ≥
1
1 −α

Y −α m −
p
2mα(1 −α) log(1/ρ) −log(1/ρ)
3

Dividing both sides of the inequality by m recovers the theorem statement.


**Table 26 from page 24**

| 0                                                                                                                        |
|:-------------------------------------------------------------------------------------------------------------------------|
| Certiﬁed Adversarial Robustness via Randomized Smoothing                                                                 |
| D. Estimating the certiﬁed test-set accuracy                                                                             |
| In this appendix, we show how to convert the “approximate certiﬁed test accuracy” considered in the main paper into a    |
| lower bound on the true certiﬁed test accuracy that holds with high probability over the randomness in CERTIFY.          |
| indicate                                                                                                                 |
| Consider a classiﬁer g, a test set S = {(x1, c1) . . . (xm, cm)}, and a radius r. For each example i ∈ [m],              |
| let zi                                                                                                                   |
| is both correct and robust at radius r, i.e.                                                                             |
| whether g’s prediction at xi                                                                                             |
| zi = 1[g(xi + δ) = ci                                                                                                    |
| ∀(cid:107)δ(cid:107)2 < r]                                                                                               |
| (cid:80)m                                                                                                                |
| The certiﬁed test set accuracy of g at radius r is deﬁned as                                                             |
| 1m                                                                                                                       |
| i=1 zi. If g is a randomized smoothing classiﬁer, we cannot                                                              |
| compute this quantity exactly, but we can estimate a lower bound that holds with arbitrarily high probability over the   |
| in the test set.                                                                                                         |
| randomness in CERTIFY. In particular, suppose that we run CERTIFY with failure rate α on each example xi                 |
| Let the Bernoulli random variable Yi denote the event that on example i, CERTIFY returns the correct label cA = ci and a |
| certiﬁed radius R which is greater than r. Let Y = (cid:80)m                                                             |
| i=1 Yi. In the main paper, we referred to Y /m as the “approximate                                                       |
| certiﬁed accuracy.” It is “approximate” because Yi = 1 does not mean that zi = 1. Rather, from Proposition 2, we know    |
| the following:                                                                                                           |
| if zi = 0, then P(Yi = 1) ≤ α. We now show how to exploit this fact to construct a one-sided conﬁdence                   |
| (cid:80)m                                                                                                                |
| 1m                                                                                                                       |
| interval for the unobserved quantity                                                                                     |
| i=1 zi using the observed quantities Y and m.                                                                            |
| Theorem 6. For any ρ > 0, with probability at least 1 − ρ over the randomness in CERTIFY,                                |
| (cid:32)                                                                                                                 |
| (cid:33)                                                                                                                 |
| (cid:114)                                                                                                                |



## Page 25

Certiﬁed Adversarial Robustness via Randomized Smoothing
E. ImageNet and CIFAR-10 Results
E.1. Certiﬁcation
Tables 2 and 3 show the approximate certiﬁed top-1 test set accuracy of randomized smoothing on ImageNet and CIFAR-10
with various noise levels σ. By “approximate certiﬁed accuracy,” we mean that we ran CERTIFY on a subsample of the
test set, and for each r we report the fraction of examples on which CERTIFY (a) did not abstain, (b) returned the correct
class, and (c) returned a radius R greater than r. There is some probability (at most α) that any example’s certiﬁcation is
inaccurate. We used α = 0.001 and n = 100000. On CIFAR-10 our base classiﬁer was a 110-layer residual network and
we certiﬁed the full test set; on ImageNet our base classiﬁer was a ResNet-50 and we certiﬁed a subsample of 500 points.
Note that the certiﬁed accuracy at r = 0 is just the standard accuracy of the smoothed classiﬁer. See Appendix J for more
experimental details.
r = 0.0
r = 0.5
r = 1.0
r = 1.5
r = 2.0
r = 2.5
r = 3.0
σ = 0.25
0.67
0.49
0.00
0.00
0.00
0.00
0.00
σ = 0.50
0.57
0.46
0.37
0.29
0.00
0.00
0.00
σ = 1.00
0.44
0.38
0.33
0.26
0.19
0.15
0.12
Table 2. Approximate certiﬁed test accuracy on ImageNet. Each row is a setting of the hyperparameter σ, each column is an ℓ2 radius.
The entry of the best σ for each radius is bolded. For comparison, random guessing would attain 0.001 accuracy.
r = 0.0
r = 0.25
r = 0.5
r = 0.75
r = 1.0
r = 1.25
r = 1.5
σ = 0.12
0.83
0.60
0.00
0.00
0.00
0.00
0.00
σ = 0.25
0.77
0.61
0.42
0.25
0.00
0.00
0.00
σ = 0.50
0.66
0.55
0.43
0.32
0.22
0.14
0.08
σ = 1.00
0.47
0.41
0.34
0.28
0.22
0.17
0.14
Table 3. Approximate certiﬁed test accuracy on CIFAR-10. Each row is a setting of the hyperparameter σ, each column is an ℓ2 radius.
The entry of the best σ for each radius is bolded. For comparison, random guessing would attain 0.1 accuracy.
E.2. Prediction
Table 4 shows the performance of PREDICT as the number of Monte Carlo samples n is varied between 100 and 10,000.
Suppose that for some test example (x, c), PREDICT returns the label ˆcA. We say that this prediction was correct if ˆcA = c
and we say that this prediction was accurate if ˆcA = g(x). For example, a prediction could be correct but inaccurate if g is
wrong at x, yet PREDICT accidentally returns the correct class. Ideally, we’d like PREDICT to be both correct and accurate.
With n = 100 Monte Carlo samples and a failure rate of α = 0.001, PREDICT is cheap to evaluate (0.15 seconds on our
hardware) yet it attains relatively high top-1 accuracy of 65% on the ImageNet test set, and only abstains 12% of the time.
When we use n = 10,000 Monte Carlo samples, PREDICT takes longer to evaluate (15 seconds), yet only abstains 4% of the
time. Interestingly, we observe from Table 4 that most of the abstentions when n = 100 were for examples on which g was
wrong, so in practice we would lose little accuracy by taking n to be as small as 100.
CORRECT, ACCURATE
CORRECT, INACCURATE
INCORRECT, ACCURATE
INCORRECT, INACCURATE
ABSTAIN
N
100
0.65
0.00
0.23
0.00
0.12
1000
0.68
0.00
0.28
0.00
0.04
10000
0.69
0.00
0.30
0.00
0.01
Table 4. Performance of PRECICT as n is varied. The dataset was ImageNet and σ = 0.25, α = 0.001. Each column shows the fraction
of test examples which ended up in one of ﬁve categories; the prediction at x is “correct” if PREDICT returned the true label, while the
prediction is “accurate” if PREDICT returned g(x). Computing g(x) exactly is not possible, so in order to determine whether PREDICT
was accurate, we took the gold standard to be the top class over n =100,000 Monte Carlo samples.


**Table 27 from page 25**

| 0                                                                                                                           |
|:----------------------------------------------------------------------------------------------------------------------------|
| Certiﬁed Adversarial Robustness via Randomized Smoothing                                                                    |
| E. ImageNet and CIFAR-10 Results                                                                                            |
| E.1. Certiﬁcation                                                                                                           |
| Tables 2 and 3 show the approximate certiﬁed top-1 test set accuracy of randomized smoothing on ImageNet and CIFAR-10       |
| with various noise levels σ. By “approximate certiﬁed accuracy,” we mean that we ran CERTIFY on a subsample of the          |
| test set, and for each r we report the fraction of examples on which CERTIFY (a) did not abstain, (b) returned the correct  |
| class, and (c) returned a radius R greater than r. There is some probability (at most α) that any example’s certiﬁcation is |
| inaccurate. We used α = 0.001 and n = 100000. On CIFAR-10 our base classiﬁer was a 110-layer residual network and           |
| we certiﬁed the full test set; on ImageNet our base classiﬁer was a ResNet-50 and we certiﬁed a subsample of 500 points.    |
| Note that the certiﬁed accuracy at r = 0 is just the standard accuracy of the smoothed classiﬁer. See Appendix J for more   |
| experimental details.                                                                                                       |



## Page 26

Certiﬁed Adversarial Robustness via Randomized Smoothing
F. Training with Noise
As mentioned in section 3.3, in the experiments for this paper, we followed Lecuyer et al. (2019) and trained the base
classiﬁer by minimizing the cross-entropy loss with Gaussian data augmentation. We now provide some justiﬁcation for this
idea.
Let {(x1, c1), . . . , (xn, cn)} be a training dataset.
We assume that the base classiﬁer takes the form f(x)
=
arg maxc∈Y fc(x), where each fc is the scoring function for class c.
Suppose that our goal is to maximize the sum of of the log-probabilities that f will classify each xi + ε as ci:
n
X
i=1
log Pε(f(xi + ε) = ci) =
n
X
i=1
log Eε 1

arg max
c
fc(xi + ε) = ci

(17)
Recall that the softmax function can be interpreted as a continuous, differentiable approximation to arg max:
1

arg max
c
fc(xi + ε) = ci

≈
exp(fci(xi + ε))
P
c∈Y exp(fc(xi + ε))
Therefore, our objective is approximately equal to:
n
X
i=1
log Eε

exp(fci(xi + ε))
P
c∈Y exp(fc(xi + ε))

(18)
By Jensen’s inequality and the concavity of log, this quantity is lower-bounded by:
n
X
i=1
Eε

log
exp(fci(xi + ε))
P
c∈Y exp(fc(xi + ε))

which is the negative of the cross-entropy loss under Gaussian data augmentation.
Therefore, minimizing the cross-entropy loss under Gaussian data augmentation will maximize (18), which will approxi-
mately maximize (17).


**Table 28 from page 26**

| 0                            | 1                                                                                                                         | 2   | 3                                                                                         | 4        | 5        | 6   | 7      | 8    | 9   | 10   | 11        | 12    | 13   | 14         | 15   |
|:-----------------------------|:--------------------------------------------------------------------------------------------------------------------------|:----|:------------------------------------------------------------------------------------------|:---------|:---------|:----|:-------|:-----|:----|:-----|:----------|:------|:-----|:-----------|:-----|
|                              |                                                                                                                           |     | Certiﬁed Adversarial Robustness via Randomized Smoothing                                  |          |          |     |        |      |     |      |           |       |      |            |      |
| F. Training with Noise       |                                                                                                                           |     |                                                                                           |          |          |     |        |      |     |      |           |       |      |            |      |
| As mentioned in section 3.3, |                                                                                                                           |     | in the experiments for this paper, we followed Lecuyer et al. (2019) and trained the base |          |          |     |        |      |     |      |           |       |      |            |      |
|                              | classiﬁer by minimizing the cross-entropy loss with Gaussian data augmentation. We now provide some justiﬁcation for this |     |                                                                                           |          |          |     |        |      |     |      |           |       |      |            |      |
| idea.                        |                                                                                                                           |     |                                                                                           |          |          |     |        |      |     |      |           |       |      |            |      |
| Let                          | {(x1, c1), . . . , (xn, cn)}                                                                                              | be  | a                                                                                         | training | dataset. | We  | assume | that | the | base | classiﬁer | takes | the  | form f (x) | =    |
|                              | arg maxc∈Y fc(x), where each fc is the scoring function for class c.                                                      |     |                                                                                           |          |          |     |        |      |     |      |           |       |      |            |      |
|                              | Suppose that our goal is to maximize the sum of of the log-probabilities that f will classify each xi + ε as ci:          |     |                                                                                           |          |          |     |        |      |     |      |           |       |      |            |      |



## Page 27

Certiﬁed Adversarial Robustness via Randomized Smoothing
G. Noise Level can Scale with Input Resolution
Since our robustness guarantee (3) in Theorem 1 does not explicitly depend on the data dimension d, one might worry that
randomized smoothing is less effective for images in high resolution — certifying a ﬁxed ℓ2 radius is “less impressive” for,
say, 224 × 224 image than for a 56 × 56 image. However, it turns out that in high resolution, images can be corrupted
with larger levels of isotropic Gaussian noise while still preserving their content. This fact is made clear by Figure 12,
which shows an image at high and low resolution corrupted by Gaussian noise with the same variance.full The class
(“hummingbird”) is easy to discern from the high-resolution noisy image, but not from the low-resolution noisy image. As a
consequence, in high resolution one can take σ to be larger while still being able to obtain a base classiﬁer that classiﬁes
noisy images accurately. Since our Theorem 1 robustness guarantee scales linearly with σ, this means that in high resolution
one can certify larger radii.
Figure 12. Top: An ImageNet image from class “hummingbird” in resolutions 56x56 (left) and 224x224 (right). Bottom: the same
images corrupted by isotropic Gaussian noise at σ = 0.5. On noiseless images the class is easy to distinguish no matter the resolution, but
on noisy data the class is much easier to distinguish when the resolution is high.
The argument above can be made rigorous, though we ﬁrst need to decide what it means for two images to be high- and
low-resolution versions of each other. Here we present one solution:
Let X denote the space of “high-resolution” images in dimension 2k×2k×3, and let X ′ denote the space of “low-resolution”
images in dimension k × k × 3. Let AVGPOOL : X →X ′ be the function which takes as input an image x in dimension
2k × 2k × 3, averages together every 2x2 square of pixels, and outputs an image in dimension k × k × 3.
Equipped with these deﬁnitions, we can say that (x, x′) ∈X ×X ′ are a high/low resolution image pair if x′ = AVGPOOL(x).
Proposition 7. Given any smoothing classiﬁer g′ : X ′ →Y, one can construct a smoothing classiﬁer g : X →Y with
the following property: for any x ∈X and x′ = AVGPOOL(x), g predicts the same class at x that g′ predicts at x′, but is
certiﬁably robust at twice the radius.
Proof. Given some smoothing classiﬁer g′ = (f ′, σ′) from X ′ to Y, deﬁne g to be the smoothing classiﬁer (f, σ) from X
to Y with noise level σ = 2σ′ and base classiﬁer f(x) = f ′(AVGPOOL(x)). Note that the average of four independent
copies of N(0, (2σ)2) is distributed as N(0, σ2). Therefore, for any high/low-resolution image pair x′ = AVGPOOL(x),
the random variable AVGPOOL(x + ε), where ε ∼N(0, (2σ)2I2k×2k×3), is equal in distribution to the random variable
x′ + ε′, where ε′ ∼N(0, σ2Ik×k×3). Hence, f(x + ε) = f ′(AVGPOOL(x + ε)) has the same distribution as f ′(x′ + ε′).
By the deﬁnition of g, this means that g(x) = g′(x′), Additionally, by Theorem 1, since σ = 2σ′, this means that g’s
prediction at x is certiﬁably robust at twice the radius as g′’s prediction at x′.


**Table 29 from page 27**

| 0                                                                                                                            | 1                                                                                                                                   |
|:-----------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------|
| Certiﬁed Adversarial Robustness via Randomized Smoothing                                                                     |                                                                                                                                     |
| G. Noise Level can Scale with Input Resolution                                                                               |                                                                                                                                     |
| Since our robustness guarantee (3) in Theorem 1 does not explicitly depend on the data dimension d, one might worry that     |                                                                                                                                     |
|                                                                                                                              | randomized smoothing is less effective for images in high resolution — certifying a ﬁxed (cid:96)2 radius is “less impressive” for, |
| say, 224 × 224 image than for a 56 × 56 image. However,                                                                      | images can be corrupted                                                                                                             |
| it                                                                                                                           |                                                                                                                                     |
| turns out                                                                                                                    |                                                                                                                                     |
| that                                                                                                                         |                                                                                                                                     |
| in high resolution,                                                                                                          |                                                                                                                                     |
| with larger levels of isotropic Gaussian noise while still preserving their content. This fact                               | is made clear by Figure 12,                                                                                                         |
| which shows an image at high and low resolution corrupted by Gaussian noise with the same variance.full The class            |                                                                                                                                     |
| (“hummingbird”) is easy to discern from the high-resolution noisy image, but not from the low-resolution noisy image. As a   |                                                                                                                                     |
| consequence, in high resolution one can take σ to be larger while still being able to obtain a base classiﬁer that classiﬁes |                                                                                                                                     |
| noisy images accurately. Since our Theorem 1 robustness guarantee scales linearly with σ, this means that in high resolution |                                                                                                                                     |
| one can certify larger radii.                                                                                                |                                                                                                                                     |



## Page 28

Certiﬁed Adversarial Robustness via Randomized Smoothing
H. Additional Experiments
H.1. Comparisons to baselines
Figure 13 compares the certiﬁed accuracy of a smoothed 20-layer resnet to that of the released models from two recent works
on certiﬁed ℓ2 robustness: the Lipschitz approach from Tsuzuku et al. (2018) and the approach from Zhang et al. (2018).
Note that in these experiments, the base classiﬁer for smoothing was larger than the networks of competing approaches. The
comparison to Zhang et al. (2018) is on CIFAR-10, while the comparison to Tsuzuku et al. (2018) is on SVHN. Note that
for each comparison, we preprocessed the dataset to follow the preprocessing used when the baseline was trained; therefore,
the radii reported for CIFAR-10 here are not comparable to the radii reported elsewhere in this paper. Full experimental
details are in Appendix J.
0.00
0.05
0.10
0.15
0.20
0.25
0.30
0.35
0.40
radius
0.0
0.2
0.4
0.6
0.8
1.0
certified accuracy
ours
(Tsuzuku et al)
(a) Tsuzuku et al. (2018)
0.0
0.1
0.2
0.3
0.4
0.5
radius
0.0
0.2
0.4
0.6
0.8
1.0
certified accuracy
ours
(Zhang et al)
(b) Zhang et al. (2018)
Figure 13. Randomized smoothing with a 20-layer resnet base classiﬁer attains higher certiﬁed accuracy than the released models from
two recent works on certiﬁed ℓ2 robustness.
H.2. High-probability guarantees
Appendix D details how to use CERTIFY to obtain a lower bound on the certiﬁed test accuracy at radius r of a randomized
smoothing classiﬁer that holds with high probability over the randomness in CERTIFY. In the main paper, we declined to do
this and simply reported the approximate certiﬁed test accuracy, deﬁned as the fraction of test examples for which CERTIFY
gives the correct prediction and certiﬁes it at radius r. Of course, with some probability (guaranteed to be less than α), each
of these certiﬁcations is wrong.
However, we now demonstrate empirically that there is a negligible difference between a proper high-probability lower
bound on the certiﬁed accuracy and the approximate version that we reported in the paper. We created a randomized
smoothing classiﬁer g on ImageNet with a ResNet-50 base classiﬁer and noise level σ = 0.25. We used CERTIFY with
α = 0.001 to certify a subsample of 500 examples from the ImageNet test set. From this we computed the approximate
certiﬁed test accuracy at each radius r. Then we used the correction from Appendix D with ρ = 0.001 to obtain a lower
bound on the certiﬁed test accuracy at r that holds pointwise with probability at least 1 −ρ over the randomness in CERTIFY.
Figure 14 plots both quantities as a function of r. Observe that the difference is so negligible that the lines almost overlap.
H.3. How much noise to use when training the base classiﬁer?
In the main paper, whenever we created a randomized smoothing classiﬁer g at noise level σ, we always trained the
corresponding base classiﬁer f with Gaussian data augmentation at noise level σ. In Figure 15, we show the effects of
training the base classiﬁer with a different level of Gaussian noise. Observe that g has a lower certiﬁed accuracy if f was
trained using a different noise level. It seems to be worse to train with noise < σ than to train with noise > σ.


**Table 30 from page 28**

| 0                                                                                                                           | 1                                                                                            |
|:----------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------|
|                                                                                                                             | Certiﬁed Adversarial Robustness via Randomized Smoothing                                     |
| H. Additional Experiments                                                                                                   |                                                                                              |
| H.1. Comparisons to baselines                                                                                               |                                                                                              |
| Figure 13 compares the certiﬁed accuracy of a smoothed 20-layer resnet to that of the released models from two recent works |                                                                                              |
| on certiﬁed (cid:96)2 robustness:                                                                                           | the Lipschitz approach from Tsuzuku et al. (2018) and the approach from Zhang et al. (2018). |
| Note that in these experiments, the base classiﬁer for smoothing was larger than the networks of competing approaches. The  |                                                                                              |
| comparison to Zhang et al. (2018) is on CIFAR-10, while the comparison to Tsuzuku et al. (2018) is on SVHN. Note that       |                                                                                              |
| for each comparison, we preprocessed the dataset to follow the preprocessing used when the baseline was trained; therefore, |                                                                                              |
| the radii reported for CIFAR-10 here are not comparable to the radii reported elsewhere in this paper. Full experimental    |                                                                                              |
| details are in Appendix J.                                                                                                  |                                                                                              |

**Table 31 from page 28**

| 0                          | 1    | 2    | 3    | 4    | 5    | 6               | 7    | 8                  | 9             |
|:---------------------------|:-----|:-----|:-----|:-----|:-----|:----------------|:-----|:-------------------|:--------------|
| details are in Appendix J. |      |      |      |      |      |                 |      |                    |               |
| 1.0                        |      |      |      |      |      |                 |      | 1.0                |               |
|                            |      |      |      |      |      | ours            |      |                    | ours          |
|                            |      |      |      |      |      | (Tsuzuku et al) |      |                    | (Zhang et al) |
| 0.8                        |      |      |      |      |      |                 |      | 0.8                |               |
| certified accuracy         |      |      |      |      |      |                 |      | certified accuracy |               |
| 0.6                        |      |      |      |      |      |                 |      | 0.6                |               |
| 0.4                        |      |      |      |      |      |                 |      | 0.4                |               |
| 0.2                        |      |      |      |      |      |                 |      | 0.2                |               |
| 0.0                        |      |      |      |      |      |                 |      | 0.0                |               |
| 0.00                       | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30            | 0.40 | 0.0                | 0.2           |
|                            |      |      |      |      |      | 0.35            |      | 0.1                | 0.3           |
|                            |      |      |      |      |      |                 |      |                    | 0.4           |
|                            |      |      |      |      |      |                 |      |                    | 0.5           |



## Page 29

Certiﬁed Adversarial Robustness via Randomized Smoothing
0.00
0.25
0.50
0.75
1.00
1.25
1.50
1.75
2.00
radius
0.0
0.2
0.4
0.6
0.8
1.0
certified accuracy
Approximate
High-Prob
Figure 14. The difference between the approximate certiﬁed accuracy, and a high-probability lower bound on the certiﬁed accuracy, is
negligible.
0.0
0.2
0.4
0.6
0.8
1.0
1.2
1.4
radius
0.0
0.2
0.4
0.6
0.8
1.0
certified accuracy
train 
= 0.25
train 
= 0.50
train 
= 1.00
(a) CIFAR-10
0.0
0.2
0.4
0.6
0.8
1.0
1.2
1.4
radius
0.0
0.2
0.4
0.6
0.8
1.0
certified accuracy
train 
= 0.25
train 
= 0.50
train 
= 1.00
(b) ImageNet
Figure 15. Vary training noise while holding prediction noise ﬁxed at σ = 0.50.
I. Derivation of Prior Randomized Smoothing Guarantees
In this appendix, we derive the randomized smoothing guarantees of Lecuyer et al. (2019) and Li et al. (2018) using the
notation of our paper. Both guarantees take same general form as ours, except with a different expression for R:
Theorem (generic guarantee): Let f : Rd →Y be any deterministic or random function, and let ε ∼N(0, σ2I). Let g
be deﬁned as in (1). Suppose cA ∈Y and pA, pB ∈[0, 1] satisfy:
P(f(x + ε) = cA) ≥pA ≥pB ≥max
c̸=cA P(f(x + ε) = c)
(19)
Then g(x + δ) = cA for all ∥δ∥2 < R.
For convenience, deﬁne the notation X ∼N(x, σ2I) and Y ∼N(x + δ, σ2I).
I.1. Lecuyer et al. (2019)
Lecuyer et al. (2019) proved a version of the generic robustness guarantee in which
R =
sup
0<β≤min

1, 1
2 log
pA
pB

σβ
r
2 log

1.25(1+exp(β))
pA−exp(2β)pB

Proof. In order to avoid notation that conﬂicts with the rest of this paper, we use β and γ where Lecuyer et al. (2019) used ϵ
and δ.


**Table 32 from page 29**

| 0                                                        | 1    | 2    | 3    | 4    | 5      | 6    | 7         | 8           | 9    |
|:---------------------------------------------------------|:-----|:-----|:-----|:-----|:-------|:-----|:----------|:------------|:-----|
| Certiﬁed Adversarial Robustness via Randomized Smoothing |      |      |      |      |        |      |           |             |      |
|                                                          | 1.0  |      |      |      |        |      |           |             |      |
|                                                          |      |      |      |      |        |      |           | Approximate |      |
|                                                          |      |      |      |      |        |      | High-Prob |             |      |
|                                                          | 0.8  |      |      |      |        |      |           |             |      |
| certified accuracy                                       | 0.6  |      |      |      |        |      |           |             |      |
|                                                          | 0.4  |      |      |      |        |      |           |             |      |
|                                                          | 0.2  |      |      |      |        |      |           |             |      |
|                                                          | 0.0  |      |      |      |        |      |           |             |      |
|                                                          | 0.00 | 0.25 | 0.50 | 0.75 | 1.00   | 1.25 | 1.50      | 1.75        | 2.00 |
|                                                          |      |      |      |      | radius |      |           |             |      |

**Table 33 from page 29**

| 0                  | 1   | 2   | 3   | 4      | 5             | 6                  | 7   | 8   | 9   | 10     | 11            |
|:-------------------|:----|:----|:----|:-------|:--------------|:-------------------|:----|:----|:----|:-------|:--------------|
| 1.0                |     |     |     |        |               | 1.0                |     |     |     |        |               |
|                    |     |     |     |        | train  = 0.25 |                    |     |     |     |        | train  = 0.25 |
|                    |     |     |     |        | train  = 0.50 |                    |     |     |     |        | train  = 0.50 |
| 0.8                |     |     |     |        |               | 0.8                |     |     |     |        |               |
|                    |     |     |     |        | train  = 1.00 |                    |     |     |     |        | train  = 1.00 |
| certified accuracy |     |     |     |        |               | certified accuracy |     |     |     |        |               |
| 0.6                |     |     |     |        |               | 0.6                |     |     |     |        |               |
| 0.4                |     |     |     |        |               | 0.4                |     |     |     |        |               |
| 0.2                |     |     |     |        |               | 0.2                |     |     |     |        |               |
| 0.0                |     |     |     |        |               | 0.0                |     |     |     |        |               |
| 0.0                | 0.2 | 0.4 | 0.6 | 0.8    | 1.0           | 0.0                | 0.2 | 0.4 | 0.6 | 0.8    | 1.0           |
|                    |     |     |     |        | 1.2           |                    |     |     |     |        | 1.2           |
|                    |     |     |     |        | 1.4           |                    |     |     |     |        | 1.4           |
|                    |     |     |     | radius |               |                    |     |     |     | radius |               |



## Page 30

Certiﬁed Adversarial Robustness via Randomized Smoothing
Suppose that we have some 0 < β ≤1 and γ > 0 such that
σ2 = ∥δ∥2
β2 2 log 1.25
γ
(20)
The “Gaussian mechanism” from differential privacy guarantees that:
P(f(X) = cA) ≤exp(β)P(f(Y ) = cA) + γ
(21)
and, symmetrically,
P(f(Y ) = cB) ≤exp(β)P(f(X) = cB) + γ
(22)
See Lecuyer et al. (2019), Lemma 2 for how to obtain this form from the standard form of the (β, γ) DP deﬁnition.
Fix a perturbation δ. To guarantee that g(x + δ) = cA, we need to show that P(f(Y ) = cA) > P(f(Y ) = cB) for each
cB ̸= cA.
Together, (21) and (22) imply that to guarantee P(f(Y ) = cA) > P(f(Y ) = cB) for any cB, it sufﬁces to show that:
P(f(X) = cA) > exp(2β)P(f(X) = cB) + γ(1 + exp(β))
(23)
Therefore, in order to guarantee thatP(f(Y ) = cA) > P(f(Y ) = cB) for each cB ̸= cA, by (19) it sufﬁces to show:
pA > exp(2β)pB + γ(1 + exp(β))
(24)
Now, inverting (20), we obtain:
γ = 1.25 exp

−σ2β2
2∥δ∥2

(25)
Plugging (25) into (24), we see that to guarantee P(f(Y ) = cA) ≥P(f(Y ) = cB) it sufﬁces to show that:
pA > exp(2β)pB + 1.25 exp

−σ2β2
2∥δ∥2

(1 + exp(β))
(26)
which rearranges to:
pA −exp(2β)pB
1.25(1 + exp(β)) > exp

−σ2β2
2∥δ∥2

(27)
Since the RHS is always positive, and the denominator on the LHS is always positive, this condition can only possibly hold
if the numerator on the LHS is positive. Therefore, we need to restrict β to
0 < β ≤min

1, 1
2 log pA
pB

(28)
The condition (27) is equivalent to:
∥δ∥2 log 1.25(1 + exp(β))
pA −exp(2β)pB
< σ2β2
2
(29)
Since pA ≤1 and pB ≥0, the denominator in the LHS is ≤1 which is in turn ≤the numerator on the LHS. Therefore, the
term inside the log in the LHS is greater than 1, so the log term on the LHS is greater than zero. Therefore, we may divide
both sides of the inequality by the log term on the LHS to obtain:
∥δ∥2 <
σ2β2
2 log

1.25(1+exp(β))
pA−exp(2β)pB

(30)
Finally, we take the square root and maximize the bound over all valid β (28) to yield:
∥δ∥<
sup
0<β≤min

1, 1
2 log
pA
pB

σβ
r
2 log

1.25(1+exp(β))
pA−exp(2β)pB

(31)


**Table 34 from page 30**

| 0                                                                            | 1                                                                                                                           | 2    |
|:-----------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------|:-----|
|                                                                              | Certiﬁed Adversarial Robustness via Randomized Smoothing                                                                    |      |
| Suppose that we have some 0 < β ≤ 1 and γ > 0 such that                      |                                                                                                                             |      |
|                                                                              | 1.25                                                                                                                        |      |
|                                                                              | (cid:107)δ(cid:107)2                                                                                                        |      |
|                                                                              | 2 log                                                                                                                       | (20) |
|                                                                              | σ2 =                                                                                                                        |      |
|                                                                              | β2                                                                                                                          |      |
|                                                                              | γ                                                                                                                           |      |
| The “Gaussian mechanism” from differential privacy guarantees that:          |                                                                                                                             |      |
|                                                                              | P(f (X) = cA) ≤ exp(β)P(f (Y ) = cA) + γ                                                                                    | (21) |
| and, symmetrically,                                                          |                                                                                                                             |      |
|                                                                              | P(f (Y ) = cB) ≤ exp(β)P(f (X) = cB) + γ                                                                                    | (22) |
|                                                                              | See Lecuyer et al. (2019), Lemma 2 for how to obtain this form from the standard form of the (β, γ) DP deﬁnition.           |      |
|                                                                              | Fix a perturbation δ. To guarantee that g(x + δ) = cA, we need to show that P(f (Y ) = cA) > P(f (Y ) = cB) for each        |      |
| cB (cid:54)= cA.                                                             |                                                                                                                             |      |
|                                                                              | Together, (21) and (22) imply that to guarantee P(f (Y ) = cA) > P(f (Y ) = cB) for any cB, it sufﬁces to show that:        |      |
|                                                                              | P(f (X) = cA) > exp(2β)P(f (X) = cB) + γ(1 + exp(β))                                                                        | (23) |
|                                                                              | Therefore, in order to guarantee thatP(f (Y ) = cA) > P(f (Y ) = cB) for each cB (cid:54)= cA, by (19) it sufﬁces to show:  |      |
|                                                                              | pA > exp(2β)pB + γ(1 + exp(β))                                                                                              | (24) |
| Now, inverting (20), we obtain:                                              |                                                                                                                             |      |
|                                                                              | (cid:18)                                                                                                                    |      |
|                                                                              | (cid:19)                                                                                                                    |      |
|                                                                              | σ2β2                                                                                                                        |      |
|                                                                              | γ = 1.25 exp                                                                                                                | (25) |
|                                                                              | −                                                                                                                           |      |
|                                                                              | 2(cid:107)δ(cid:107)2                                                                                                       |      |
|                                                                              | Plugging (25) into (24), we see that to guarantee P(f (Y ) = cA) ≥ P(f (Y ) = cB) it sufﬁces to show that:                  |      |
|                                                                              | (cid:18)                                                                                                                    |      |
|                                                                              | (cid:19)                                                                                                                    |      |
|                                                                              | σ2β2                                                                                                                        |      |
|                                                                              | −                                                                                                                           | (26) |
|                                                                              | (1 + exp(β))                                                                                                                |      |
|                                                                              | pA > exp(2β)pB + 1.25 exp                                                                                                   |      |
|                                                                              | 2(cid:107)δ(cid:107)2                                                                                                       |      |
| which rearranges to:                                                         |                                                                                                                             |      |
|                                                                              | (cid:18)                                                                                                                    |      |
|                                                                              | (cid:19)                                                                                                                    |      |
|                                                                              | pA − exp(2β)pB                                                                                                              |      |
|                                                                              | σ2β2                                                                                                                        |      |
|                                                                              | > exp                                                                                                                       | (27) |
|                                                                              | −                                                                                                                           |      |
|                                                                              | 1.25(1 + exp(β))                                                                                                            |      |
|                                                                              | 2(cid:107)δ(cid:107)2                                                                                                       |      |
|                                                                              | Since the RHS is always positive, and the denominator on the LHS is always positive, this condition can only possibly hold  |      |
| if the numerator on the LHS is positive. Therefore, we need to restrict β to |                                                                                                                             |      |
|                                                                              | (cid:18)                                                                                                                    |      |
|                                                                              | (cid:19)                                                                                                                    |      |
|                                                                              | pA                                                                                                                          |      |
|                                                                              | 1 2                                                                                                                         | (28) |
|                                                                              | log                                                                                                                         |      |
|                                                                              | 0 < β ≤ min                                                                                                                 |      |
|                                                                              | 1,                                                                                                                          |      |
|                                                                              | pB                                                                                                                          |      |
| The condition (27) is equivalent to:                                         |                                                                                                                             |      |
|                                                                              | σ2β2                                                                                                                        |      |
|                                                                              | 1.25(1 + exp(β))                                                                                                            |      |
|                                                                              | <                                                                                                                           | (29) |
|                                                                              | (cid:107)δ(cid:107)2 log                                                                                                    |      |
|                                                                              | 2                                                                                                                           |      |
|                                                                              | pA − exp(2β)pB                                                                                                              |      |
|                                                                              | Since pA ≤ 1 and pB ≥ 0, the denominator in the LHS is ≤ 1 which is in turn ≤ the numerator on the LHS. Therefore, the      |      |
|                                                                              | term inside the log in the LHS is greater than 1, so the log term on the LHS is greater than zero. Therefore, we may divide |      |
| both sides of the inequality by the log term on the LHS to obtain:           |                                                                                                                             |      |
|                                                                              | σ2β2                                                                                                                        |      |
|                                                                              | (cid:107)δ(cid:107)2 <                                                                                                      | (30) |
|                                                                              | (cid:17)                                                                                                                    |      |
|                                                                              | (cid:16) 1.25(1+exp(β))                                                                                                     |      |
|                                                                              | 2 log                                                                                                                       |      |
|                                                                              | pA−exp(2β)pB                                                                                                                |      |



## Page 31

Certiﬁed Adversarial Robustness via Randomized Smoothing
Figure 16a plots this bound at varying settings of the tuning parameter β, while Figure 16c plots how the bound varies with
β for a ﬁxed pA and pB.
I.2. Li et al. (2018)
Li et al. (2018) proved a version of the generic robustness guarantee in which
R = sup
α>0
σ
s
−2
α log

1 −pA −pB + 2
1
2(pA1−α + pB1−α)1−α

Proof. A generalization of KL divergence, the α-Renyi divergence is an information theoretic measure of distance between
two distributions. It is parameterized by some α > 0. The α-Renyi divergence between two discrete distributions P and Q
is deﬁned as:
Dα(P||Q) :=
1
α −1 log
 k
X
i=1
pα
i
qα−1
i
!
(32)
In the continuous case, this sum is replaced with an integral. The divergence is undeﬁned when α = 1 since a division by
zero occurs, but the limit of Dα(P||Q) as α →1 is the KL divergence between P and Q.
Li et al. (2018) prove that if P is a discrete distribution for which the highest probability class has probability ≥pA and all
other classes have probability ≤pB, then for any other discrete distribution Q for which
Dα(P||Q) < −log

1 −pA −pB + 2
1
2(pA
1−α + pB
1−α)1−α

(33)
the highest-probability class in Q is guaranteed to be the same as the highest-probability class in P.
We now apply this result to the discrete distributions P = f(X) and Q = f(Y ). If Dα(f(X)||f(Y )) satisﬁes (33), then it
is guaranteed that g(x) = g(x + δ).
The data processing inequality states that applying a function to two random variables can only decrease the α-Renyi
divergence between them. In particular,
Dα(f(X)||f(Y )) ≤Dα(X||Y )
(34)
There is a closed-form expression for the α-Renyi divergence between two Gaussians:
Dα(X||Y ) = α∥δ∥2
2σ2
(35)
Therefore, we can guarantee that g(x + δ) = cA so long as
α∥δ∥2
2σ2
< −log

1 −pA −pB + 2
1
2(pA
1−α + pB
1−α)1−α

(36)
which simpliﬁes to
∥δ∥< σ
s
−2
α log

1 −pA −pB + 2
1
2(pA1−α + pB1−α)1−α

(37)
Finally, since this result holds for any α > 0, we may maximize over α to obtain the largest possible certiﬁed radius:
∥δ∥< sup
α>0
σ
s
−2
α log

1 −pA −pB + 2
1
2(pA1−α + pB1−α)1−α

(38)
Figure 16b plots this bound at varying settings of the tuning parameter α, while ﬁgure 16d plots how the bound varies with
α for a ﬁxed pA and pB.


**Table 35 from page 31**

| 0                                                                                                                            | 1                                                                                                                                | 2       | 3                             | 4        | 5                | 6    |
|:-----------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------|:--------|:------------------------------|:---------|:-----------------|:-----|
|                                                                                                                              | Certiﬁed Adversarial Robustness via Randomized Smoothing                                                                         |         |                               |          |                  |      |
|                                                                                                                              | Figure 16a plots this bound at varying settings of the tuning parameter β, while Figure 16c plots how the bound varies with      |         |                               |          |                  |      |
| β for a ﬁxed pA and pB.                                                                                                      |                                                                                                                                  |         |                               |          |                  |      |
| I.2. Li et al. (2018)                                                                                                        |                                                                                                                                  |         |                               |          |                  |      |
|                                                                                                                              | Li et al. (2018) proved a version of the generic robustness guarantee in which                                                   |         |                               |          |                  |      |
|                                                                                                                              | (cid:115)                                                                                                                        |         |                               |          |                  |      |
|                                                                                                                              |                                                                                                                                  |         | (cid:18)                      |          | (cid:19)(cid:19) |      |
|                                                                                                                              |                                                                                                                                  |         | (cid:18) 1                    |          |                  |      |
| R = sup                                                                                                                      | σ                                                                                                                                | 2 α     | log                           | 1−α + pB | 1−α)1−α          |      |
|                                                                                                                              |                                                                                                                                  | −       | 1 − pA − pB + 2               |          |                  |      |
|                                                                                                                              |                                                                                                                                  |         | (pA                           |          |                  |      |
| α>0                                                                                                                          |                                                                                                                                  |         | 2                             |          |                  |      |
|                                                                                                                              | Proof. A generalization of KL divergence, the α-Renyi divergence is an information theoretic measure of distance between         |         |                               |          |                  |      |
|                                                                                                                              | two distributions. It is parameterized by some α > 0. The α-Renyi divergence between two discrete distributions P and Q          |         |                               |          |                  |      |
| is deﬁned as:                                                                                                                |                                                                                                                                  |         |                               |          |                  |      |
|                                                                                                                              |                                                                                                                                  |         | (cid:32) k                    | (cid:33) |                  |      |
|                                                                                                                              |                                                                                                                                  |         | 1                             |          |                  |      |
|                                                                                                                              |                                                                                                                                  |         | pα                            |          |                  |      |
|                                                                                                                              |                                                                                                                                  |         | i                             |          |                  |      |
|                                                                                                                              |                                                                                                                                  |         | (cid:88) i                    |          |                  | (32) |
|                                                                                                                              |                                                                                                                                  |         | log                           |          |                  |      |
|                                                                                                                              |                                                                                                                                  |         | Dα(P ||Q) :=                  |          |                  |      |
|                                                                                                                              |                                                                                                                                  |         | α − 1                         |          |                  |      |
|                                                                                                                              |                                                                                                                                  |         | qα−1                          |          |                  |      |
|                                                                                                                              |                                                                                                                                  |         | i                             |          |                  |      |
|                                                                                                                              |                                                                                                                                  |         | =1                            |          |                  |      |
|                                                                                                                              | In the continuous case, this sum is replaced with an integral. The divergence is undeﬁned when α = 1 since a division by         |         |                               |          |                  |      |
|                                                                                                                              | zero occurs, but the limit of Dα(P ||Q) as α → 1 is the KL divergence between P and Q.                                           |         |                               |          |                  |      |
|                                                                                                                              | Li et al. (2018) prove that if P is a discrete distribution for which the highest probability class has probability ≥ pA and all |         |                               |          |                  |      |
|                                                                                                                              | other classes have probability ≤ pB, then for any other discrete distribution Q for which                                        |         |                               |          |                  |      |
|                                                                                                                              |                                                                                                                                  |         | (cid:18)                      |          | (cid:19)(cid:19) |      |
|                                                                                                                              |                                                                                                                                  |         | (cid:18) 1                    |          |                  |      |
|                                                                                                                              | Dα(P ||Q) < − log                                                                                                                |         | (pA                           | 1−α + pB | 1−α)1−α          | (33) |
|                                                                                                                              |                                                                                                                                  |         | 1 − pA − pB + 2               |          |                  |      |
|                                                                                                                              |                                                                                                                                  |         | 2                             |          |                  |      |
|                                                                                                                              | the highest-probability class in Q is guaranteed to be the same as the highest-probability class in P .                          |         |                               |          |                  |      |
| We now apply this result to the discrete distributions P = f (X) and Q = f (Y ). If Dα(f (X)||f (Y )) satisﬁes (33), then it |                                                                                                                                  |         |                               |          |                  |      |
| is guaranteed that g(x) = g(x + δ).                                                                                          |                                                                                                                                  |         |                               |          |                  |      |
|                                                                                                                              | The data processing inequality states that applying a function to two random variables can only decrease the α-Renyi             |         |                               |          |                  |      |
|                                                                                                                              | divergence between them. In particular,                                                                                          |         |                               |          |                  |      |
|                                                                                                                              |                                                                                                                                  |         | Dα(f (X)||f (Y )) ≤ Dα(X||Y ) |          |                  | (34) |
|                                                                                                                              | There is a closed-form expression for the α-Renyi divergence between two Gaussians:                                              |         |                               |          |                  |      |
|                                                                                                                              |                                                                                                                                  |         | α(cid:107)δ(cid:107)2         |          |                  |      |
|                                                                                                                              |                                                                                                                                  |         | Dα(X||Y ) =                   |          |                  | (35) |
|                                                                                                                              |                                                                                                                                  |         | 2σ2                           |          |                  |      |
|                                                                                                                              | Therefore, we can guarantee that g(x + δ) = cA so long as                                                                        |         |                               |          |                  |      |
|                                                                                                                              |                                                                                                                                  |         | (cid:18)                      |          | (cid:19)(cid:19) |      |
| α(cid:107)δ(cid:107)2                                                                                                        |                                                                                                                                  |         | (cid:18) 1                    |          |                  |      |
|                                                                                                                              |                                                                                                                                  | < − log | (pA                           | 1−α)1−α  |                  | (36) |
|                                                                                                                              |                                                                                                                                  |         | 1 − pA − pB + 2               | 1−α + pB |                  |      |
| 2σ2                                                                                                                          |                                                                                                                                  |         | 2                             |          |                  |      |
| which simpliﬁes to                                                                                                           |                                                                                                                                  |         |                               |          |                  |      |
|                                                                                                                              | (cid:115)                                                                                                                        |         |                               |          |                  |      |
|                                                                                                                              |                                                                                                                                  |         | (cid:18)                      |          | (cid:19)(cid:19) |      |
|                                                                                                                              |                                                                                                                                  |         | (cid:18) 1                    |          |                  |      |
| (cid:107)δ(cid:107) < σ                                                                                                      | −                                                                                                                                | 2 α     | 1 − pA − pB + 2               | 1−α)1−α  |                  | (37) |
|                                                                                                                              |                                                                                                                                  | log     | (pA                           | 1−α + pB |                  |      |
|                                                                                                                              |                                                                                                                                  |         | 2                             |          |                  |      |
|                                                                                                                              | Finally, since this result holds for any α > 0, we may maximize over α to obtain the largest possible certiﬁed radius:           |         |                               |          |                  |      |
|                                                                                                                              | (cid:115)                                                                                                                        |         |                               |          |                  |      |
|                                                                                                                              |                                                                                                                                  |         | (cid:18)                      |          | (cid:19)(cid:19) |      |
|                                                                                                                              |                                                                                                                                  |         | (cid:18) 1                    |          |                  |      |
| (cid:107)δ(cid:107) < sup                                                                                                    | σ                                                                                                                                | 2 α     | log                           | 1−α + pB | 1−α)1−α          | (38) |
|                                                                                                                              |                                                                                                                                  | −       | (pA                           |          |                  |      |
|                                                                                                                              |                                                                                                                                  |         | 1 − pA − pB + 2               |          |                  |      |
| α>0                                                                                                                          |                                                                                                                                  |         | 2                             |          |                  |      |
|                                                                                                                              | Figure 16b plots this bound at varying settings of the tuning parameter α, while ﬁgure 16d plots how the bound varies with       |         |                               |          |                  |      |
| α for a ﬁxed pA and pB.                                                                                                      |                                                                                                                                  |         |                               |          |                  |      |



## Page 32

Certiﬁed Adversarial Robustness via Randomized Smoothing
(a) The Lecuyer et al. (2019) bound over several settings of β. The
brown line is the pointwise supremum over all eligible β, computed
numerically.
(b) The Li et al. (2018) bound over several settings of α. The
purple line is the pointwise supremum over all eligible α, computed
numerically.
(c) Tuning the Lecuyer et al. (2019) bound wrt β when pA =
0.8, pB = 0.2
(d) Tuning the Li et al. (2018) bound wrt α when pA = 0.999, pB =
0.0001
J. Experiment Details
J.1. Comparison to baselines
We compared randomized smoothing against three recent approaches for ℓ2-robust classiﬁcation (Tsuzuku et al., 2018;
Wong et al., 2018; Zhang et al., 2018). Tsuzuku et al. (2018) and Wong et al. (2018) propose both a robust training method
and a complementary certiﬁcation mechanism, while Zhang et al. (2018) propose a method to certify generically trained
networks. In all cases we compared against networks provided by the authors. We compared against Wong et al. (2018) and
Zhang et al. (2018) on CIFAR-10, and we compared against Tsuzuku et al. (2018) on SVHN.
In image classiﬁcation it is common practice to preprocess a dataset by subtracting from each channel the mean over the
dataset, and dividing each channel by the standard deviation over the dataset. However, we wanted to report certiﬁed radii
in the original image coordinates rather than in the standardized coordinates. Therefore, throughout most of this work we
ﬁrst added the Gaussian noise, and then standardized the channels, before feeding the image to the base classiﬁer. (In the
practical PyTorch implementation, the ﬁrst layer of the base classiﬁer was a layer that standardized the input.) However, all
of the baselines we compared against provided pre-trained networks which assumed that the dataset was ﬁrst preprocessed
in a speciﬁc way. Therefore, when comparing against the baselines we also preprocessed the datasets ﬁrst, so that we could
report certiﬁed radii that were directly comparable to the radii reported by the baseline methods.
Comparison to Wong et al. (2018)
Following Wong et al. (2018), the CIFAR-10 dataset was preprocessed by subtracting
(0.485, 0.456, 0.406) and dividing by (0.225, 0.225, 0.225).
While the body of the Wong et al. (2018) paper focuses on ℓ∞certiﬁed robustness, their algorithm naturally extends to
ℓ2 certiﬁed robustness, as developed in the appendix of the paper. We used three ℓ2-trained residual networks publicly
released by the authors, each trained with a different setting of their hyperparameter ϵ ∈{0.157, 0.628, 2.51}. We used code
publicly released by the authors at https://github.com/locuslab/convex_adversarial/blob/master/


**Table 36 from page 32**

| 0                                                                                                                                    |
|:-------------------------------------------------------------------------------------------------------------------------------------|
| Certiﬁed Adversarial Robustness via Randomized Smoothing                                                                             |
| (b) The Li et al.                                                                                                                    |
| (2018) bound over several settings of α.                                                                                             |
| The                                                                                                                                  |
| (a) The Lecuyer et al. (2019) bound over several settings of β. The                                                                  |
| purple line is the pointwise supremum over all eligible α, computed                                                                  |
| brown line is the pointwise supremum over all eligible β, computed                                                                   |
| numerically.                                                                                                                         |
| numerically.                                                                                                                         |
| (c) Tuning the Lecuyer et al.                                                                                                        |
| (2019) bound wrt β when pA =                                                                                                         |
| (d) Tuning the Li et al. (2018) bound wrt α when pA = 0.999, pB =                                                                    |
| 0.8, pB = 0.2                                                                                                                        |
| 0.0001                                                                                                                               |
| J. Experiment Details                                                                                                                |
| J.1. Comparison to baselines                                                                                                         |
| We compared randomized smoothing against                                                                                             |
| three recent approaches for (cid:96)2-robust classiﬁcation (Tsuzuku et al., 2018;                                                    |
| Wong et al., 2018; Zhang et al., 2018). Tsuzuku et al. (2018) and Wong et al. (2018) propose both a robust training method           |
| and a complementary certiﬁcation mechanism, while Zhang et al. (2018) propose a method to certify generically trained                |
| networks. In all cases we compared against networks provided by the authors. We compared against Wong et al. (2018) and              |
| Zhang et al. (2018) on CIFAR-10, and we compared against Tsuzuku et al. (2018) on SVHN.                                              |
| In image classiﬁcation it is common practice to preprocess a dataset by subtracting from each channel the mean over the              |
| dataset, and dividing each channel by the standard deviation over the dataset. However, we wanted to report certiﬁed radii           |
| in the original image coordinates rather than in the standardized coordinates. Therefore, throughout most of this work we            |
| ﬁrst added the Gaussian noise, and then standardized the channels, before feeding the image to the base classiﬁer. (In the           |
| practical PyTorch implementation, the ﬁrst layer of the base classiﬁer was a layer that standardized the input.) However, all        |
| of the baselines we compared against provided pre-trained networks which assumed that the dataset was ﬁrst preprocessed              |
| in a speciﬁc way. Therefore, when comparing against the baselines we also preprocessed the datasets ﬁrst, so that we could           |
| report certiﬁed radii that were directly comparable to the radii reported by the baseline methods.                                   |
| Comparison to Wong et al. (2018)                                                                                                     |
| Following Wong et al. (2018), the CIFAR-10 dataset was preprocessed by subtracting                                                   |
| (0.485, 0.456, 0.406) and dividing by (0.225, 0.225, 0.225).                                                                         |
| While the body of the Wong et al. (2018) paper focuses on (cid:96)∞ certiﬁed robustness, their algorithm naturally extends to        |
| (cid:96)2 certiﬁed robustness, as developed in the appendix of the paper. We used three (cid:96)2-trained residual networks publicly |
| released by the authors, each trained with a different setting of their hyperparameter (cid:15) ∈ {0.157, 0.628, 2.51}. We used code |
| publicly released by the authors at https://github.com/locuslab/convex_adversarial/blob/master/                                      |



## Page 33

Certiﬁed Adversarial Robustness via Randomized Smoothing
examples/cifar_evaluate.py to compute the robustness radius of test images. The code accepts a radius and
returns TRUE (robust) or FALSE (not robust); we incorporated this subroutine into a binary search procedure to ﬁnd the
largest radius for which the code returned TRUE.
For randomized smoothing we used σ = 0.6 and a 20-layer residual network base classiﬁer. We ran CERTIFY with n0 = 100,
n = 100,000 and α = 0.001.
For both methods, we certiﬁed the full CIFAR-10 test set.
Comparison to Tsuzuku et al. (2018)
Following Tsuzuku et al. (2018), the SVHN dataset was not preprocessed except
that pixels were divided by 255 so as to lie within [0, 1].
We compared against a pretrained network provided to us by the authors in which the hyperparameter of their method was
set to c = 0.1. The network was a wide residual network with 16 layers and a width factor of 4. We used the authors’ code
at https://github.com/ytsmiling/lmt to compute the robustness radius of test images.
For randomized smoothing we used σ = 0.1 and a 20-layer residual network base classiﬁer. We ran CERTIFY with n0 = 100,
n = 100,000 and α = 0.001.
For both methods, we certiﬁed the whole SVHN test set.
Comparison to Zhang et al. (2018)
Following Zhang et al. (2018), the CIFAR-10 dataset was preprocessed by subtracting
0.5 from each pixel.
We compared against the cifar 7 1024 vanilla network released by the authors, which is a 7-layer MLP. We used the
authors’ code at https://github.com/IBM/CROWN-Robustness-Certification to compute the robustness
radius of test images.
For randomized smoothing we used σ = 1.2 and a 20-layer residual network base classiﬁer. We ran CERTIFY with n0 = 100,
n = 100,000 and α = 0.001.
For randomized smoothing, we certiﬁed the whole CIFAR-10 test set. For Zhang et al. (2018), we certiﬁed every fourth
image in the CIFAR-10 test set.
J.2. ImageNet and CIFAR-10 Experiments
Our code is available at http://github.com/locuslab/smoothing.
In order to report certiﬁed radii in the original coordinates, we ﬁrst added Gaussian noise, and then standardized the data.
Speciﬁcally, in our PyTorch implementation, the ﬁrst layer of the base classiﬁer was a normalization layer that performed
a channel-wise standardization of its input. For CIFAR-10 we subtracted the dataset mean (0.4914, 0.4822, 0.4465)
and divided by the dataset standard deviation (0.2023, 0.1994, 0.2010). For ImageNet we subtracted the dataset mean
(0.485, 0.456, 0.406) and divided by the standard deviation (0.229, 0.224, 0.225).
For both ImageNet and CIFAR-10, we trained the base classiﬁer with random horizontal ﬂips and random crops (in addition
to the Gaussian data augmentation discussed explicitly in the paper). On ImageNet we trained with synchronous SGD on
four NVIDIA RTX 2080 Ti GPUs; training took approximately three days.
On ImageNet our base classiﬁer used the ResNet-50 architecture provided in torchvision. On CIFAR-10 we used a
110-layer residual network from https://github.com/bearpaw/pytorch-classification.
On ImageNet we certiﬁed every 100-th image in the validation set, for 500 images total. On CIFAR-10 we certiﬁed the
whole test set.
In Figure 8 (middle) we ﬁxed σ = 0.25 and α = 0.001 while varying the number of samples n. We did not actually vary
the number of samples n that we simulated: we kept this number ﬁxed at 100,000 but varied the number that we fed the
Clopper-Pearson conﬁdence interval.
In Figure 8 (right), we ﬁxed σ = 0.25 and n =100,000 while varying α.


**Table 37 from page 33**

| 0                                                                                                                            |
|:-----------------------------------------------------------------------------------------------------------------------------|
| Certiﬁed Adversarial Robustness via Randomized Smoothing                                                                     |
| examples/cifar_evaluate.py to compute the robustness radius of test                                                          |
| images. The code accepts a radius and                                                                                        |
| returns TRUE (robust) or FALSE (not robust); we incorporated this subroutine into a binary search procedure to ﬁnd the       |
| largest radius for which the code returned TRUE.                                                                             |
| For randomized smoothing we used σ = 0.6 and a 20-layer residual network base classiﬁer. We ran CERTIFY with n0 = 100,       |
| n = 100,000 and α = 0.001.                                                                                                   |
| For both methods, we certiﬁed the full CIFAR-10 test set.                                                                    |
| Comparison to Tsuzuku et al. (2018)                                                                                          |
| Following Tsuzuku et al. (2018), the SVHN dataset was not preprocessed except                                                |
| that pixels were divided by 255 so as to lie within [0, 1].                                                                  |
| We compared against a pretrained network provided to us by the authors in which the hyperparameter of their method was       |
| set to c = 0.1. The network was a wide residual network with 16 layers and a width factor of 4. We used the authors’ code    |
| at https://github.com/ytsmiling/lmt to compute the robustness radius of test images.                                         |
| For randomized smoothing we used σ = 0.1 and a 20-layer residual network base classiﬁer. We ran CERTIFY with n0 = 100,       |
| n = 100,000 and α = 0.001.                                                                                                   |
| For both methods, we certiﬁed the whole SVHN test set.                                                                       |
| Comparison to Zhang et al. (2018)                                                                                            |
| Following Zhang et al. (2018), the CIFAR-10 dataset was preprocessed by subtracting                                          |
| 0.5 from each pixel.                                                                                                         |
| We compared against the cifar 7 1024 vanilla network released by the authors, which is a 7-layer MLP. We used the            |
| authors’ code at https://github.com/IBM/CROWN-Robustness-Certification to compute the robustness                             |
| radius of test images.                                                                                                       |
| For randomized smoothing we used σ = 1.2 and a 20-layer residual network base classiﬁer. We ran CERTIFY with n0 = 100,       |
| n = 100,000 and α = 0.001.                                                                                                   |
| For randomized smoothing, we certiﬁed the whole CIFAR-10 test set. For Zhang et al. (2018), we certiﬁed every fourth         |
| image in the CIFAR-10 test set.                                                                                              |
| J.2. ImageNet and CIFAR-10 Experiments                                                                                       |
| Our code is available at http://github.com/locuslab/smoothing.                                                               |
| In order to report certiﬁed radii in the original coordinates, we ﬁrst added Gaussian noise, and then standardized the data. |
| Speciﬁcally, in our PyTorch implementation, the ﬁrst layer of the base classiﬁer was a normalization layer that performed    |
| a channel-wise standardization of                                                                                            |
| its input.                                                                                                                   |
| For CIFAR-10 we subtracted the dataset mean (0.4914, 0.4822, 0.4465)                                                         |
| and divided by the dataset standard deviation (0.2023, 0.1994, 0.2010). For ImageNet we subtracted the dataset mean          |
| (0.485, 0.456, 0.406) and divided by the standard deviation (0.229, 0.224, 0.225).                                           |
| For both ImageNet and CIFAR-10, we trained the base classiﬁer with random horizontal ﬂips and random crops (in addition      |
| to the Gaussian data augmentation discussed explicitly in the paper). On ImageNet we trained with synchronous SGD on         |
| four NVIDIA RTX 2080 Ti GPUs; training took approximately three days.                                                        |
| On ImageNet our base classiﬁer used the ResNet-50 architecture provided in torchvision. On CIFAR-10 we used a                |
| 110-layer residual network from https://github.com/bearpaw/pytorch-classification.                                           |
| On ImageNet we certiﬁed every 100-th image in the validation set, for 500 images total. On CIFAR-10 we certiﬁed the          |
| whole test set.                                                                                                              |
| In Figure 8 (middle) we ﬁxed σ = 0.25 and α = 0.001 while varying the number of samples n. We did not actually vary          |
| the number of samples n that we simulated: we kept this number ﬁxed at 100,000 but varied the number that we fed the         |
| Clopper-Pearson conﬁdence interval.                                                                                          |
| In Figure 8 (right), we ﬁxed σ = 0.25 and n =100,000 while varying α.                                                        |



## Page 34

Certiﬁed Adversarial Robustness via Randomized Smoothing
J.3. Adversarial Attacks
As discussed in Section 4, we subjected smoothed classiﬁers to a projected gradient descent-style adversarial attack. We
now describe the details of this attack.
Let f be the base classiﬁer and let σ be the noise level. Following Li et al. (2018), given an example (x, c) ∈Rd × Y and a
radius r, we used a projected gradient descent style adversarial attack to optimize the objective:
arg max
δ:∥δ∥2<r
Eε∼N(0,σ2I) [ℓ(f(x + δ + ε), c)]
(39)
where ℓis the softmax loss function. (Breaking notation with the rest of the paper in which f returns a class, the function f
here refers to the function that maps an image in Rd to a vector of classwise scores.)
At each iteration of the attack, we drew k samples of noise, ε1 . . . εk ∼N(0, σ2I), and followed the stochastic gradient
gt = Pk
i=1 ∇δtℓ(f(x + δt + εk), c).
As is typical (Kolter & Madry, 2018), we used a “steepest ascent” update rule, which, for the ℓ2 norm, means that we
normalized the gradient before applying the update. The overall PGD update is: δt+1 = projr

δt + η
gt
∥gt∥

where the
function projr that projects its input onto the ball {z : ∥z∥2 ≤r} is given by projr(z) =
rz
max(r,∥z∥2). We used a constant
step size η and a ﬁxed number T of PGD iterations.
In practice, our step size was η = 0.1, we used T = 20 steps of PGD, and we computed the stochastic gradient using
k = 1000 Monte Carlo samples.
Unfortunately, the objective we optimize (39) is not actually the attack objective of interest. To force a misclassiﬁcation, an
attacker needs to ﬁnd some perturbation δ with ∥δ∥2 < r and some class cB for which
Pε∼N(0,σ2I)(f(x + δ + ε) = cB) ≥Pε∼N(0,σ2I)(f(x + δ + ε) = c)
Effective adversarial attacks against randomized smoothing are outside the scope of this paper.


**Table 38 from page 34**

| 0                                                                                                                                     | 1                                                                                                                               |
|:--------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------|
| Certiﬁed Adversarial Robustness via Randomized Smoothing                                                                              |                                                                                                                                 |
| J.3. Adversarial Attacks                                                                                                              |                                                                                                                                 |
| As discussed in Section 4, we subjected smoothed classiﬁers to a projected gradient descent-style adversarial attack. We              |                                                                                                                                 |
| now describe the details of this attack.                                                                                              |                                                                                                                                 |
|                                                                                                                                       | Let f be the base classiﬁer and let σ be the noise level. Following Li et al. (2018), given an example (x, c) ∈ Rd × Y and a    |
| radius r, we used a projected gradient descent style adversarial attack to optimize the objective:                                    |                                                                                                                                 |
| arg max                                                                                                                               | (39)                                                                                                                            |
| Eε∼N (0,σ2I) [(cid:96)(f (x + δ + ε), c)]                                                                                             |                                                                                                                                 |
| δ:(cid:107)δ(cid:107)2<r                                                                                                              |                                                                                                                                 |
| where (cid:96) is the softmax loss function. (Breaking notation with the rest of the paper in which f returns a class, the function f |                                                                                                                                 |
| here refers to the function that maps an image in Rd to a vector of classwise scores.)                                                |                                                                                                                                 |
| At each iteration of the attack, we drew k samples of noise, ε1 . . . εk ∼ N (0, σ2I), and followed the stochastic gradient           |                                                                                                                                 |
| gt = (cid:80)k                                                                                                                        |                                                                                                                                 |
| i=1 ∇δt(cid:96)(f (x + δt + εk), c).                                                                                                  |                                                                                                                                 |
| As is typical (Kolter & Madry, 2018), we used a “steepest ascent” update rule, which, for the (cid:96)2 norm, means that we           |                                                                                                                                 |
|                                                                                                                                       | (cid:16)                                                                                                                        |
|                                                                                                                                       | (cid:17)                                                                                                                        |
|                                                                                                                                       | gt                                                                                                                              |
| normalized the gradient before applying the update. The overall PGD update is: δt+1 = projr                                           | where the                                                                                                                       |
|                                                                                                                                       | δt + η                                                                                                                          |
|                                                                                                                                       | (cid:107)gt(cid:107)                                                                                                            |
|                                                                                                                                       | rz                                                                                                                              |
| function projr that projects its input onto the ball {z : (cid:107)z(cid:107)2 ≤ r} is given by projr(z) =                            | max(r,(cid:107)z(cid:107)2) . We used a constant                                                                                |
| step size η and a ﬁxed number T of PGD iterations.                                                                                    |                                                                                                                                 |
|                                                                                                                                       | In practice, our step size was η = 0.1, we used T = 20 steps of PGD, and we computed the stochastic gradient using              |
| k = 1000 Monte Carlo samples.                                                                                                         |                                                                                                                                 |
|                                                                                                                                       | Unfortunately, the objective we optimize (39) is not actually the attack objective of interest. To force a misclassiﬁcation, an |
| attacker needs to ﬁnd some perturbation δ with (cid:107)δ(cid:107)2 < r and some class cB for which                                   |                                                                                                                                 |
| Pε∼N (0,σ2I)(f (x + δ + ε) = cB) ≥ Pε∼N (0,σ2I)(f (x + δ + ε) = c)                                                                    |                                                                                                                                 |
| Effective adversarial attacks against randomized smoothing are outside the scope of this paper.                                       |                                                                                                                                 |



## Page 35

Certiﬁed Adversarial Robustness via Randomized Smoothing
K. Examples of Noisy Images
We now show examples of CIFAR-10 and ImageNet images corrupted with varying levels of noise.
σ = 0.00
σ = 0.25
σ = 0.50
σ = 1.00
Figure 17. CIFAR-10 images additively corrupted by varying levels of Gaussian noise N(0, σ2I). Pixel values greater than 1.0 (=255) or
less than 0.0 (=0) were clipped to 1.0 or 0.0.


**Table 39 from page 35**

| 0                                                                                                                                       | 1                                                                                            | 2                                                        | 3        |
|:----------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------|:---------------------------------------------------------|:---------|
|                                                                                                                                         |                                                                                              | Certiﬁed Adversarial Robustness via Randomized Smoothing |          |
| K. Examples of Noisy Images                                                                                                             |                                                                                              |                                                          |          |
|                                                                                                                                         | We now show examples of CIFAR-10 and ImageNet images corrupted with varying levels of noise. |                                                          |          |
| σ = 0.00                                                                                                                                | σ = 0.25                                                                                     | σ = 0.50                                                 | σ = 1.00 |
| Figure 17. CIFAR-10 images additively corrupted by varying levels of Gaussian noise N (0, σ2I). Pixel values greater than 1.0 (=255) or |                                                                                              |                                                          |          |
| less than 0.0 (=0) were clipped to 1.0 or 0.0.                                                                                          |                                                                                              |                                                          |          |



## Page 36

Certiﬁed Adversarial Robustness via Randomized Smoothing
σ = 0.00
σ = 0.25
σ = 0.50
σ = 1.00
Figure 18. ImageNet images additively corrupted by varying levels of Gaussian noise N(0, σ2I). Pixel values greater than 1.0 (=255) or
less than 0.0 (=0) were clipped to 1.0 or 0.0.


**Table 40 from page 36**

| 0                                                                                                                                       | 1        | 2                                                        | 3        |
|:----------------------------------------------------------------------------------------------------------------------------------------|:---------|:---------------------------------------------------------|:---------|
|                                                                                                                                         |          | Certiﬁed Adversarial Robustness via Randomized Smoothing |          |
| σ = 0.00                                                                                                                                | σ = 0.25 | σ = 0.50                                                 | σ = 1.00 |
| Figure 18. ImageNet images additively corrupted by varying levels of Gaussian noise N (0, σ2I). Pixel values greater than 1.0 (=255) or |          |                                                          |          |
| less than 0.0 (=0) were clipped to 1.0 or 0.0.                                                                                          |          |                                                          |          |

