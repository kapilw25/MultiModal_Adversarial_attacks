# 2_Adversarial_Examples_Are_Not_Bugs_They_Are_Features

## Document Information

- **Source**: attack_models/research_papers/2_Adversarial_Examples_Are_Not_Bugs_They_Are_Features.pdf
- **Pages**: 37
- **Tables**: 42



## Page 1

Adversarial Examples Are Not Bugs, They Are Features
Andrew Ilyas∗
MIT
ailyas@mit.edu
Shibani Santurkar∗
MIT
shibani@mit.edu
Dimitris Tsipras∗
MIT
tsipras@mit.edu
Logan Engstrom∗
MIT
engstrom@mit.edu
Brandon Tran
MIT
btran115@mit.edu
Aleksander M ˛adry
MIT
madry@mit.edu
Abstract
Adversarial examples have attracted signiﬁcant attention in machine learning, but the reasons for their
existence and pervasiveness remain unclear. We demonstrate that adversarial examples can be directly at-
tributed to the presence of non-robust features: features (derived from patterns in the data distribution) that
are highly predictive, yet brittle and (thus) incomprehensible to humans. After capturing these features
within a theoretical framework, we establish their widespread existence in standard datasets. Finally, we
present a simple setting where we can rigorously tie the phenomena we observe in practice to a misalign-
ment between the (human-speciﬁed) notion of robustness and the inherent geometry of the data.
1
Introduction
The pervasive brittleness of deep neural networks [Sze+14; Eng+19b; HD19; Ath+18] has attracted signif-
icant attention in recent years. Particularly worrisome is the phenomenon of adversarial examples [Big+13;
Sze+14], imperceptibly perturbed natural inputs that induce erroneous predictions in state-of-the-art clas-
siﬁers. Previous work has proposed a variety of explanations for this phenomenon, ranging from theoreti-
cal models [Sch+18; BPR18] to arguments based on concentration of measure in high-dimensions [Gil+18;
MDM18; Sha+19a]. These theories, however, are often unable to fully capture behaviors we observe in
practice (we discuss this further in Section 5).
More broadly, previous work in the ﬁeld tends to view adversarial examples as aberrations arising either
from the high dimensional nature of the input space or statistical ﬂuctuations in the training data [Sze+14;
GSS15; Gil+18]. From this point of view, it is natural to treat adversarial robustness as a goal that can
be disentangled and pursued independently from maximizing accuracy [Mad+18; SHS19; Sug+19], ei-
ther through improved standard regularization methods [TG16] or pre/post-processing of network in-
puts/outputs [Ues+18; CW17a; He+17].
In this work, we propose a new perspective on the phenomenon of adversarial examples. In contrast
to the previous models, we cast adversarial vulnerability as a fundamental consequence of the dominant
supervised learning paradigm. Speciﬁcally, we claim that:
Adversarial vulnerability is a direct result of our models’ sensitivity to well-generalizing features in the data.
Recall that we usually train classiﬁers to solely maximize (distributional) accuracy. Consequently, classiﬁers
tend to use any available signal to do so, even those that look incomprehensible to humans. After all, the
presence of “a tail” or “ears” is no more natural to a classiﬁer than any other equally predictive feature. In
fact, we ﬁnd that standard ML datasets do admit highly predictive yet imperceptible features. We posit that
∗Equal contribution
1
arXiv:1905.02175v4  [stat.ML]  12 Aug 2019


**Table 1 from page 1**

| 0                                                                                                               |
|:----------------------------------------------------------------------------------------------------------------|
| Abstract                                                                                                        |
| Adversarial examples have attracted signiﬁcant attention in machine learning, but the reasons for their         |
| existence and pervasiveness remain unclear. We demonstrate that adversarial examples can be directly at-        |
| tributed to the presence of non-robust features: features (derived from patterns in the data distribution) that |
| are highly predictive, yet brittle and (thus) incomprehensible to humans. After capturing these features        |
| within a theoretical framework, we establish their widespread existence in standard datasets. Finally, we       |
| present a simple setting where we can rigorously tie the phenomena we observe in practice to a misalign-        |
| ment between the (human-speciﬁed) notion of robustness and the inherent geometry of the data.                   |

**Table 2 from page 1**

| 0                                                                                                                  |
|:-------------------------------------------------------------------------------------------------------------------|
| ment between the (human-speciﬁed) notion of robustness and the inherent geometry of the data.                      |
| 1                                                                                                                  |
| Introduction                                                                                                       |
| The pervasive brittleness of deep neural networks [Sze+14; Eng+19b; HD19; Ath+18] has attracted signif-            |
| icant attention in recent years. Particularly worrisome is the phenomenon of adversarial examples [Big+13;         |
| Sze+14], imperceptibly perturbed natural inputs that induce erroneous predictions in state-of-the-art clas-        |
| siﬁers. Previous work has proposed a variety of explanations for this phenomenon, ranging from theoreti-           |
| cal models [Sch+18; BPR18] to arguments based on concentration of measure in high-dimensions [Gil+18;              |
| MDM18; Sha+19a].                                                                                                   |
| These theories, however, are often unable to fully capture behaviors we observe in                                 |
| practice (we discuss this further in Section 5).                                                                   |
| More broadly, previous work in the ﬁeld tends to view adversarial examples as aberrations arising either           |
| from the high dimensional nature of the input space or statistical ﬂuctuations in the training data [Sze+14;       |
| GSS15; Gil+18].                                                                                                    |
| From this point of view,                                                                                           |
| it                                                                                                                 |
| is natural                                                                                                         |
| to treat adversarial robustness as a goal                                                                          |
| that can                                                                                                           |
| be disentangled and pursued independently from maximizing accuracy [Mad+18; SHS19; Sug+19], ei-                    |
| ther                                                                                                               |
| through improved standard regularization methods [TG16] or pre/post-processing of network in-                      |
| puts/outputs [Ues+18; CW17a; He+17].                                                                               |
| In this work, we propose a new perspective on the phenomenon of adversarial examples.                              |
| In contrast                                                                                                        |
| to the previous models, we cast adversarial vulnerability as a fundamental consequence of the dominant             |
| supervised learning paradigm. Speciﬁcally, we claim that:                                                          |
| Adversarial vulnerability is a direct result of our models’ sensitivity to well-generalizing features in the data. |
| Recall that we usually train classiﬁers to solely maximize (distributional) accuracy. Consequently, classiﬁers     |
| tend to use any available signal to do so, even those that look incomprehensible to humans. After all, the         |
| presence of “a tail” or “ears” is no more natural to a classiﬁer than any other equally predictive feature.        |
| In                                                                                                                 |
| fact, we ﬁnd that standard ML datasets do admit highly predictive yet imperceptible features. We posit that        |



## Page 2

our models learn to rely on these “non-robust” features, leading to adversarial perturbations that exploit
this dependence.1
Our hypothesis also suggests an explanation for adversarial transferability: the phenomenon that adver-
sarial perturbations computed for one model often transfer to other, independently trained models. Since
any two models are likely to learn similar non-robust features, perturbations that manipulate such fea-
tures will apply to both. Finally, this perspective establishes adversarial vulnerability as a human-centric
phenomenon, since, from the standard supervised learning point of view, non-robust features can be as
important as robust ones. It also suggests that approaches aiming to enhance the interpretability of a given
model by enforcing “priors” for its explanation [MV15; OMS17; Smi+17] actually hide features that are
“meaningful” and predictive to standard models. As such, producing human-meaningful explanations that
remain faithful to underlying models cannot be pursued independently from the training of the models
themselves.
To corroborate our theory, we show that it is possible to disentangle robust from non-robust features in
standard image classiﬁcation datasets. Speciﬁcally, given any training dataset, we are able to construct:
1. A “robustiﬁed” version for robust classiﬁcation (Figure 1a)2 . We demonstrate that it is possible to
effectively remove non-robust features from a dataset. Concretely, we create a training set (seman-
tically similar to the original) on which standard training yields good robust accuracy on the original,
unmodiﬁed test set. This ﬁnding establishes that adversarial vulnerability is not necessarily tied to the
standard training framework, but is also a property of the dataset.
2. A “non-robust” version for standard classiﬁcation (Figure 1b)2. We are also able to construct a
training dataset for which the inputs are nearly identical to the originals, but all appear incorrectly
labeled. In fact, the inputs in the new training set are associated to their labels only through small
adversarial perturbations (and hence utilize only non-robust features). Despite the lack of any predictive
human-visible information, training on this dataset yields good accuracy on the original, unmodiﬁed
test set. This demonstrates that adversarial perturbations can arise from ﬂipping features in the data
that are useful for classiﬁcation of correct inputs (hence not being purely aberrations).
Finally, we present a concrete classiﬁcation task where the connection between adversarial examples and
non-robust features can be studied rigorously. This task consists of separating Gaussian distributions, and
is loosely based on the model presented in Tsipras et al. [Tsi+19], while expanding upon it in a few ways.
First, adversarial vulnerability in our setting can be precisely quantiﬁed as a difference between the intrinsic
data geometry and that of the adversary’s perturbation set. Second, robust training yields a classiﬁer which
utilizes a geometry corresponding to a combination of these two. Lastly, the gradients of standard models
can be signiﬁcantly more misaligned with the inter-class direction, capturing a phenomenon that has been
observed in practice in more complex scenarios [Tsi+19].
2
The Robust Features Model
We begin by developing a framework, loosely based on the setting proposed by Tsipras et al. [Tsi+19],
that enables us to rigorously refer to “robust” and “non-robust” features. In particular, we present a set of
deﬁnitions which allow us to formally describe our setup, theoretical results, and empirical evidence.
Setup.
We consider binary classiﬁcation3, where input-label pairs (x, y) ∈X × {±1} are sampled from a
(data) distribution D; the goal is to learn a classiﬁer C : X →{±1} which predicts a label y corresponding
to a given input x.
1It is worth emphasizing that while our ﬁndings demonstrate that adversarial vulnerability does arise from non-robust features, they
do not preclude the possibility of adversarial vulnerability also arising from other phenomena [TG16; Sch+18]. For example, Nakkiran
[Nak19a] constructs adversarial examples that do not exploit non-robust features (and hence do not allow one to learn a generalizing
model from them). Still, the mere existence of useful non-robust features sufﬁces to establish that without explicitly discouraging
models from utilizing these features, adversarial vulnerability will remain an issue.
2The corresponding datasets for CIFAR-10 are publicly available at http://git.io/adv-datasets.
3Our framework can be straightforwardly adapted though to the multi-class setting.
2


**Table 3 from page 2**

| 0                                                                                                                |
|:-----------------------------------------------------------------------------------------------------------------|
| our models learn to rely on these “non-robust” features,                                                         |
| leading to adversarial perturbations that exploit                                                                |
| this dependence.1                                                                                                |
| Our hypothesis also suggests an explanation for adversarial transferability:                                     |
| the phenomenon that adver-                                                                                       |
| sarial perturbations computed for one model often transfer to other,                                             |
| independently trained models. Since                                                                              |
| any two models are likely to learn similar non-robust                                                            |
| features, perturbations that manipulate such fea-                                                                |
| tures will apply to both. Finally, this perspective establishes adversarial vulnerability as a human-centric     |
| phenomenon, since,                                                                                               |
| from the standard supervised learning point of view, non-robust                                                  |
| features can be as                                                                                               |
| important as robust ones. It also suggests that approaches aiming to enhance the interpretability of a given     |
| model by enforcing “priors” for its explanation [MV15; OMS17; Smi+17] actually hide features that are            |
| “meaningful” and predictive to standard models. As such, producing human-meaningful explanations that            |
| remain faithful                                                                                                  |
| to underlying models cannot be pursued independently from the training of                                        |
| the models                                                                                                       |
| themselves.                                                                                                      |
| To corroborate our theory, we show that it is possible to disentangle robust from non-robust features in         |
| standard image classiﬁcation datasets. Speciﬁcally, given any training dataset, we are able to construct:        |
| 1. A “robustiﬁed” version for robust classiﬁcation (Figure 1a)2 . We demonstrate that it is possible to          |
| effectively remove non-robust features from a dataset. Concretely, we create a training set (seman-              |
| tically similar to the original) on which standard training yields good robust accuracy on the original,         |
| unmodiﬁed test set. This ﬁnding establishes that adversarial vulnerability is not necessarily tied to the        |
| standard training framework, but is also a property of the dataset.                                              |
| 2. A “non-robust” version for standard classiﬁcation (Figure 1b)2. We are also able to construct a               |
| training dataset for which the inputs are nearly identical to the originals, but all appear incorrectly          |
| labeled.                                                                                                         |
| In fact,                                                                                                         |
| the inputs in the new training set are associated to their labels only through small                             |
| adversarial perturbations (and hence utilize only non-robust features). Despite the lack of any predictive       |
| human-visible information,                                                                                       |
| training on this dataset yields good accuracy on the original, unmodiﬁed                                         |
| test set. This demonstrates that adversarial perturbations can arise from ﬂipping features in the data           |
| that are useful for classiﬁcation of correct inputs (hence not being purely aberrations).                        |
| Finally, we present a concrete classiﬁcation task where the connection between adversarial examples and          |
| non-robust features can be studied rigorously. This task consists of separating Gaussian distributions, and      |
| is loosely based on the model presented in Tsipras et al. [Tsi+19], while expanding upon it in a few ways.       |
| First, adversarial vulnerability in our setting can be precisely quantiﬁed as a difference between the intrinsic |
| data geometry and that of the adversary’s perturbation set. Second, robust training yields a classiﬁer which     |
| utilizes a geometry corresponding to a combination of these two. Lastly, the gradients of standard models        |
| can be signiﬁcantly more misaligned with the inter-class direction, capturing a phenomenon that has been         |
| observed in practice in more complex scenarios [Tsi+19].                                                         |



## Page 3

Robust dataset
Train
good standard accuracy 
good robust accuracy
good standard accuracy 
bad robust accuracy
Unmodiﬁed 
test set
Training image
frog
frog
frog
Non-robust dataset
Train
(a)
Evaluate on 
original test set
Training image
Robust Features: dog 
Non-Robust Features: dog
dog
Relabel as cat
Robust Features: dog 
Non-Robust Features: cat
cat
cat
max P(cat) 
Adversarial example 
towards “cat” 
Train
good accuracy
(b)
Figure 1: A conceptual diagram of the experiments of Section 3. In (a) we disentangle features into combi-
nations of robust/non-robust features (Section 3.1). In (b) we construct a dataset which appears mislabeled
to humans (via adversarial examples) but results in good accuracy on the original test set (Section 3.2).
We deﬁne a feature to be a function mapping from the input space X to the real numbers, with the set
of all features thus being F = { f : X →R}. For convenience, we assume that the features in F are
shifted/scaled to be mean-zero and unit-variance (i.e., so that E(x,y)∼D[ f (x)] = 0 and E(x,y)∼D[ f (x)2] = 1),
in order to make the following deﬁnitions scale-invariant4. Note that this formal deﬁnition also captures
what we abstractly think of as features (e.g., we can construct an f that captures how “furry” an image is).
Useful, robust, and non-robust features.
We now deﬁne the key concepts required for formulating our
framework. To this end, we categorize features in the following manner:
• ρ-useful features: For a given distribution D, we call a feature f ρ-useful (ρ > 0) if it is correlated with
the true label in expectation, that is if
E(x,y)∼D[y · f (x)] ≥ρ.
(1)
We then deﬁne ρD( f ) as the largest ρ for which feature f is ρ-useful under distribution D. (Note that
if a feature f is negatively correlated with the label, then −f is useful instead.) Crucially, a linear
classiﬁer trained on ρ-useful features can attain non-trivial generalization performance.
• γ-robustly useful features: Suppose we have a ρ-useful feature f (ρD( f ) > 0). We refer to f as a
robust feature (formally a γ-robustly useful feature for γ > 0) if, under adversarial perturbation (for
some speciﬁed set of valid perturbations ∆), f remains γ-useful. Formally, if we have that
E(x,y)∼D

inf
δ∈∆(x) y · f (x + δ)

≥γ.
(2)
• Useful, non-robust features: A useful, non-robust feature is a feature which is ρ-useful for some ρ
bounded away from zero, but is not a γ-robust feature for any γ ≥0. These features help with classi-
ﬁcation in the standard setting, but may hinder accuracy in the adversarial setting, as the correlation
with the label can be ﬂipped.
Classiﬁcation.
In our framework, a classiﬁer C = (F, w, b) is comprised of a set of features F ⊆F, a
weight vector w, and a scalar bias b. For a given input x, the classiﬁer predicts the label y as
C(x) = sgn
 
b + ∑
f ∈F
w f · f (x)
!
.
For convenience, we denote the set of features learned by a classiﬁer C as FC.
4This restriction can be straightforwardly removed by simply shifting/scaling the deﬁnitions.
3


**Table 4 from page 3**

| 0                                                                                                           |
|:------------------------------------------------------------------------------------------------------------|
| frog                                                                                                        |
| Evaluate on                                                                                                 |
| Non-robust dataset                                                                                          |
| original test set                                                                                           |
| (a)                                                                                                         |
| (b)                                                                                                         |
| Figure 1: A conceptual diagram of the experiments of Section 3.                                             |
| In (a) we disentangle features into combi-                                                                  |
| nations of robust/non-robust features (Section 3.1). In (b) we construct a dataset which appears mislabeled |
| to humans (via adversarial examples) but results in good accuracy on the original test set (Section 3.2).   |
| We deﬁne a feature to be a function mapping from the input space                                            |
| to the real numbers, with the set                                                                           |
| X                                                                                                           |
| R                                                                                                           |
| =                                                                                                           |
| f                                                                                                           |
| of all                                                                                                      |
| features thus being                                                                                         |
| :                                                                                                           |
| .                                                                                                           |
| For convenience, we assume that                                                                             |
| the features in                                                                                             |
| are                                                                                                         |
| F                                                                                                           |
| {                                                                                                           |
| X →                                                                                                         |
| }                                                                                                           |
| F                                                                                                           |
| shifted/scaled to be mean-zero and unit-variance (i.e., so that E                                           |
| [ f (x)] = 0 and E                                                                                          |
| [ f (x)2] = 1),                                                                                             |
| (x,y)                                                                                                       |
| (x,y)                                                                                                       |
| ∼D                                                                                                          |
| ∼D                                                                                                          |
| in order to make the following deﬁnitions scale-invariant4. Note that this formal deﬁnition also captures   |
| what we abstractly think of as features (e.g., we can construct an f                                        |
| that captures how “furry” an image is).                                                                     |
| Useful, robust, and non-robust features. We now deﬁne the key concepts required for formulating our         |
| framework. To this end, we categorize features in the following manner:                                     |
| ρ-useful features: For a given distribution                                                                 |
| , we call a feature f ρ-useful (ρ > 0) if it is correlated with                                             |
| •                                                                                                           |
| D                                                                                                           |
| the true label in expectation, that is if                                                                   |
| E                                                                                                           |
| [y                                                                                                          |
| f (x)]                                                                                                      |
| ρ.                                                                                                          |
| (1)                                                                                                         |
| (x,y)                                                                                                       |
| ·                                                                                                           |
| ≥                                                                                                           |
| ∼D                                                                                                          |
| We then deﬁne ρ                                                                                             |
| ( f ) as the largest ρ for which feature f                                                                  |
| is ρ-useful under distribution                                                                              |
| . (Note that                                                                                                |
| D                                                                                                           |
| D                                                                                                           |
| f                                                                                                           |
| f                                                                                                           |
| if a feature                                                                                                |
| is negatively correlated with the label,                                                                    |
| then                                                                                                        |
| is useful                                                                                                   |
| instead.) Crucially, a linear                                                                               |
| −                                                                                                           |
| classiﬁer trained on ρ-useful features can attain non-trivial generalization performance.                   |
| f                                                                                                           |
| γ-robustly useful features: Suppose we have a ρ-useful                                                      |
| feature                                                                                                     |
| (ρ                                                                                                          |
| ( f ) > 0). We refer to f as a                                                                              |
| D                                                                                                           |
| •                                                                                                           |
| robust feature (formally a γ-robustly useful feature for γ > 0) if, under adversarial perturbation (for     |
| f                                                                                                           |
| some speciﬁed set of valid perturbations ∆),                                                                |
| remains γ-useful. Formally, if we have that                                                                 |
| E                                                                                                           |
| y                                                                                                           |
| f (x + δ)                                                                                                   |
| inf                                                                                                         |
| γ.                                                                                                          |
| (2)                                                                                                         |
| (x,y)                                                                                                       |
| ∆(x)                                                                                                        |
| δ                                                                                                           |
| ·                                                                                                           |
| ≥                                                                                                           |
| ∈                                                                                                           |
| ∼D (cid:20)                                                                                                 |
| (cid:21)                                                                                                    |
| Useful, non-robust features: A useful, non-robust                                                           |
| feature is a feature which is ρ-useful                                                                      |
| for some ρ                                                                                                  |
| •                                                                                                           |
| bounded away from zero, but is not a γ-robust feature for any γ                                             |
| 0. These features help with classi-                                                                         |
| ≥                                                                                                           |
| ﬁcation in the standard setting, but may hinder accuracy in the adversarial setting, as the correlation     |
| with the label can be ﬂipped.                                                                               |
| Classiﬁcation.                                                                                              |
| In our framework, a classiﬁer C = (F, w, b) is comprised of a set of                                        |
| features F                                                                                                  |
| , a                                                                                                         |
| ⊆ F                                                                                                         |
| weight vector w, and a scalar bias b. For a given input x, the classiﬁer predicts the label y as            |



## Page 4

Standard Training.
Training a classiﬁer is performed by minimizing a loss function (via empirical risk
minimization (ERM)) that decreases with the correlation between the weighted combination of the features
and the label. The simplest example of such a loss is 5
E(x,y)∼D [Lθ(x, y)] = −E(x,y)∼D
"
y ·
 
b + ∑
f ∈F
w f · f (x)
!#
.
(3)
When minimizing classiﬁcation loss, no distinction exists between robust and non-robust features: the only
distinguishing factor of a feature is its ρ-usefulness. Furthermore, the classiﬁer will utilize any ρ-useful
feature in F to decrease the loss of the classiﬁer.
Robust training.
In the presence of an adversary, any useful but non-robust features can be made anti-
correlated with the true label, leading to adversarial vulnerability. Therefore, ERM is no longer sufﬁcient
to train classiﬁers that are robust, and we need to explicitly account for the effect of the adversary on the
classiﬁer. To do so, we use an adversarial loss function that can discern between robust and non-robust
features [Mad+18]:
E(x,y)∼D

max
δ∈∆(x) Lθ(x + δ, y)

,
(4)
for an appropriately deﬁned set of perturbations ∆. Since the adversary can exploit non-robust features to
degrade classiﬁcation accuracy, minimizing this adversarial loss (as in adversarial training [GSS15; Mad+18])
can be viewed as explicitly preventing the classiﬁer from learning a useful but non-robust combination of
features.
Remark.
We want to note that even though the framework above enables us to formally describe and
predict the outcome of our experiments, it does not necessarily capture the notion of non-robust features
exactly as we intuitively might think of them. For instance, in principle, our theoretical framework would
allow for useful non-robust features to arise as combinations of useful robust features and useless non-
robust features [Goh19b]. These types of constructions, however, are actually precluded by our experi-
mental results (in particular, the classiﬁers trained in Section 3 would not generalize). This shows that our
experimental ﬁndings capture a stronger, more ﬁne-grained statement than our formal deﬁnitions are able
to express. We view bridging this gap as an interesting direction for future work.
3
Finding Robust (and Non-Robust) Features
The central premise of our proposed framework is that there exist both robust and non-robust features that
constitute useful signals for standard classiﬁcation. We now provide evidence in support of this hypothesis
by disentangling these two sets of features.
On one hand, we will construct a “robustiﬁed” dataset, consisting of samples that primarily contain
robust features. Using such a dataset, we are able to train robust classiﬁers (with respect to the standard
test set) using standard (i.e., non-robust) training. This demonstrates that robustness can arise by removing
certain features from the dataset (as, overall, the new dataset contains less information about the original
training set). Moreover, it provides evidence that adversarial vulnerability is caused by non-robust features
and is not inherently tied to the standard training framework.
On the other hand, we will construct datasets where the input-label association is based purely on non-
robust features (and thus the corresponding dataset appears completely mislabeled to humans). We show
that this dataset sufﬁces to train a classiﬁer with good performance on the standard test set. This indicates
that natural models use non-robust features to make predictions, even in the presence of robust features.
These features alone are actually sufﬁcient for non-trivial generalizations performance on natural images,
which indicates that they are indeed valuable features, rather than artifacts of ﬁnite-sample overﬁtting.
A conceptual description of these experiments can be found in Figure 1.
5Just as for the other parts of this model, we use this loss for simplicity only—it is straightforward to generalize to more practical
loss function such as logistic or hinge loss.
4


**Table 5 from page 4**

| 0                                                                                                             |
|:--------------------------------------------------------------------------------------------------------------|
| risk                                                                                                          |
| Standard Training.                                                                                            |
| Training a classiﬁer is performed by minimizing a loss function (via empirical                                |
| minimization (ERM)) that decreases with the correlation between the weighted combination of the features      |
| and the label. The simplest example of such a loss is 5                                                       |
| E                                                                                                             |
| E                                                                                                             |
| b + ∑                                                                                                         |
| [                                                                                                             |
| y                                                                                                             |
| f (x)                                                                                                         |
| .                                                                                                             |
| (3)                                                                                                           |
| w f                                                                                                           |
| (x,y)                                                                                                         |
| (x,y)                                                                                                         |
| Lθ(x, y)] =                                                                                                   |
| −                                                                                                             |
| ·                                                                                                             |
| ∼D                                                                                                            |
| ∼D (cid:34)                                                                                                   |
| · (cid:32)                                                                                                    |
| (cid:33)(cid:35)                                                                                              |
| f                                                                                                             |
| F                                                                                                             |
| ∈                                                                                                             |
| When minimizing classiﬁcation loss, no distinction exists between robust and non-robust features:             |
| the only                                                                                                      |
| distinguishing factor of a feature is its ρ-usefulness.                                                       |
| Furthermore,                                                                                                  |
| the classiﬁer will utilize any ρ-useful                                                                       |
| feature in F to decrease the loss of the classiﬁer.                                                           |
| Robust                                                                                                        |
| training.                                                                                                     |
| In the presence of an adversary, any useful but non-robust                                                    |
| features can be made anti-                                                                                    |
| correlated with the true label,                                                                               |
| leading to adversarial vulnerability. Therefore, ERM is no longer sufﬁcient                                   |
| to train classiﬁers that are robust, and we need to explicitly account for the effect of the adversary on the |
| classiﬁer.                                                                                                    |
| To do so, we use an adversarial                                                                               |
| loss function that can discern between robust and non-robust                                                  |
| features [Mad+18]:                                                                                            |
| E                                                                                                             |
| max                                                                                                           |
| ,                                                                                                             |
| (4)                                                                                                           |
| (x,y)                                                                                                         |
| ∆(x) Lθ(x + δ, y)                                                                                             |
| δ                                                                                                             |
| ∈                                                                                                             |
| ∼D (cid:20)                                                                                                   |
| (cid:21)                                                                                                      |
| for an appropriately deﬁned set of perturbations ∆. Since the adversary can exploit non-robust features to    |
| degrade classiﬁcation accuracy, minimizing this adversarial loss (as in adversarial training [GSS15; Mad+18]) |
| can be viewed as explicitly preventing the classiﬁer from learning a useful but non-robust combination of     |
| features.                                                                                                     |
| Remark. We want                                                                                               |
| to note that even though the framework above enables us to formally describe and                              |
| predict the outcome of our experiments,                                                                       |
| it does not necessarily capture the notion of non-robust features                                             |
| exactly as we intuitively might think of them. For instance, in principle, our theoretical framework would    |
| allow for useful non-robust                                                                                   |
| features to arise as combinations of useful robust                                                            |
| features and useless non-                                                                                     |
| robust                                                                                                        |
| features [Goh19b].                                                                                            |
| These types of constructions, however, are actually precluded by our experi-                                  |
| mental results (in particular, the classiﬁers trained in Section 3 would not generalize). This shows that our |
| experimental ﬁndings capture a stronger, more ﬁne-grained statement than our formal deﬁnitions are able       |
| to express. We view bridging this gap as an interesting direction for future work.                            |



## Page 5

“airplane’’
“ship’’
“dog’’
“frog’’
“truck’’
D
!DNR
!DR
(a)
Std Training 
 using 
Adv Training 
 using 
Std Training 
 using 
R
Std Training 
 using 
NR
0
20
40
60
80
100
Test Accuracy on 
 (%)
Std accuracy
Adv accuracy ( = 0.25)
(b)
Figure 2: Left: Random samples from our variants of the CIFAR-10 [Kri09] training set: the original training
set; the robust training set bDR, restricted to features used by a robust model; and the non-robust training
set bDNR, restricted to features relevant to a standard model (labels appear incorrect to humans). Right:
Standard and robust accuracy on the CIFAR-10 test set (D) for models trained with: (i) standard training
(on D) ; (ii) standard training on bDNR; (iii) adversarial training (on D); and (iv) standard training on bDR.
Models trained on bDR and bDNR reﬂect the original models used to create them: notably, standard training
on bDR yields nontrivial robust accuracy. Results for Restricted-ImageNet [Tsi+19] are in D.8 Figure 12.
3.1
Disentangling robust and non-robust features
Recall that the features a classiﬁer learns to rely on are based purely on how useful these features are
for (standard) generalization. Thus, under our conceptual framework, if we can ensure that only robust
features are useful, standard training should result in a robust classiﬁer. Unfortunately, we cannot directly
manipulate the features of very complex, high-dimensional datasets. Instead, we will leverage a robust
model and modify our dataset to contain only the features that are relevant to that model.
In terms of our formal framework (Section 2), given a robust (i.e., adversarially trained [Mad+18]) model
C we aim to construct a distribution bDR which satisﬁes:
E(x,y)∼bDR [ f (x) · y] =
(
E(x,y)∼D [ f (x) · y]
if f ∈FC
0
otherwise,
(5)
where FC again represents the set of features utilized by C. Conceptually, we want features used by C to
be as useful as they were on the original distribution D while ensuring that the rest of the features are not
useful under bDNR.
We will construct a training set for bDR via a one-to-one mapping x 7→xr from the original training set for
D. In the case of a deep neural network, FC corresponds to exactly the set of activations in the penultimate
layer (since these correspond to inputs to a linear classiﬁer). To ensure that features used by the model are
equally useful under both training sets, we (approximately) enforce all features in FC to have similar values
for both x and xr through the following optimization:
min
xr ∥g(xr) −g(x)∥2,
(6)
where x is the original input and g is the mapping from x to the representation layer. We optimize this
objective using gradient descent in input space6.
Since we don’t have access to features outside FC, there is no way to ensure that the expectation in (5) is
zero for all f ̸∈FC. To approximate this condition, we choose the starting point of gradient descent for the
optimization in (6) to be an input x0 which is drawn from D independently of the label of x (we also explore
sampling x0 from noise in Appendix D.1). This choice ensures that any feature present in that input will
6We follow [Mad+18] and normalize gradient steps during this optimization. Experimental details are provided in Appendix C.
5


**Table 6 from page 5**

| 0                                                                                                            | 1                                                                                                             | 2   |
|:-------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------|:----|
| 0                                                                                                            |                                                                                                               |     |
| Std Training                                                                                                 | Std Training                                                                                                  |     |
| Adv Training                                                                                                 |                                                                                                               |     |
| Std Training                                                                                                 |                                                                                                               |     |
| using                                                                                                        |                                                                                                               |     |
|  using                                                                                                       |                                                                                                               |     |
| R                                                                                                            |                                                                                                               |     |
| NR                                                                                                           |                                                                                                               |     |
|  using                                                                                                       |                                                                                                               |     |
|  using                                                                                                       |                                                                                                               |     |
| (a)                                                                                                          |                                                                                                               |     |
| (b)                                                                                                          |                                                                                                               |     |
| Figure 2: Left: Random samples from our variants of the CIFAR-10 [Kri09] training set: the original training |                                                                                                               |     |
| training set                                                                                                 | training                                                                                                      |     |
| set;                                                                                                         |                                                                                                               |     |
| the robust                                                                                                   |                                                                                                               |     |
| restricted to features used by a robust model; and the non-robust                                            |                                                                                                               |     |
| R,                                                                                                           |                                                                                                               |     |
| (cid:98)D                                                                                                    | to humans). Right:                                                                                            |     |
| set                                                                                                          |                                                                                                               |     |
| to a standard model                                                                                          |                                                                                                               |     |
| (labels appear incorrect                                                                                     |                                                                                                               |     |
| NR, restricted to features relevant                                                                          |                                                                                                               |     |
| Standard and robust accuracy on the CIFAR-10 test set (                                                      | (i) standard training                                                                                         |     |
| ) for models trained with:                                                                                   |                                                                                                               |     |
| (cid:98)D                                                                                                    |                                                                                                               |     |
| D                                                                                                            |                                                                                                               |     |
| (on                                                                                                          |                                                                                                               | R.  |
| ) ; (ii) standard training on                                                                                |                                                                                                               |     |
| ); and (iv) standard training on                                                                             |                                                                                                               |     |
| NR; (iii) adversarial training (on                                                                           |                                                                                                               |     |
| D                                                                                                            |                                                                                                               |     |
| D                                                                                                            |                                                                                                               |     |
| (cid:98)D                                                                                                    | (cid:98)D                                                                                                     |     |
| Models trained on                                                                                            | NR reﬂect the original models used to create them: notably, standard training                                 |     |
| R and                                                                                                        |                                                                                                               |     |
| (cid:98)D                                                                                                    |                                                                                                               |     |
| (cid:98)D                                                                                                    |                                                                                                               |     |
| (cid:98)D                                                                                                    |                                                                                                               |     |
| on                                                                                                           |                                                                                                               |     |
| R yields nontrivial robust accuracy. Results for Restricted-ImageNet [Tsi+19] are in D.8 Figure 12.          |                                                                                                               |     |
| 3.1                                                                                                          |                                                                                                               |     |
| Disentangling robust and non-robust features                                                                 |                                                                                                               |     |
| Recall                                                                                                       | these features are                                                                                            |     |
| that                                                                                                         |                                                                                                               |     |
| the features a classiﬁer                                                                                     |                                                                                                               |     |
| learns to rely on are based purely on how useful                                                             |                                                                                                               |     |
| for (standard) generalization. Thus, under our conceptual                                                    | if we can ensure that only robust                                                                             |     |
| framework,                                                                                                   |                                                                                                               |     |
|                                                                                                              | features are useful, standard training should result in a robust classiﬁer. Unfortunately, we cannot directly |     |
| manipulate the features of very complex, high-dimensional datasets.                                          | leverage a robust                                                                                             |     |
| Instead, we will                                                                                             |                                                                                                               |     |
| model and modify our dataset to contain only the features that are relevant to that model.                   |                                                                                                               |     |
|                                                                                                              | In terms of our formal framework (Section 2), given a robust (i.e., adversarially trained [Mad+18]) model     |     |
| (cid:98)D                                                                                                    |                                                                                                               |     |
| C we aim to construct a distribution                                                                         |                                                                                                               |     |
| R which satisﬁes:                                                                                            |                                                                                                               |     |



## Page 6

not be useful since they are not correlated with the label in expectation over x0. The underlying assumption
here is that, when performing the optimization in (6), features that are not being directly optimized (i.e.,
features outside FC) are not affected. We provide pseudocode for the construction in Figure 5 (Appendix C).
Given the new training set for bDR (a few random samples are visualized in Figure 2a), we train a clas-
siﬁer using standard (non-robust) training. We then test this classiﬁer on the original test set (i.e. D). The
results (Figure 2b) indicate that the classiﬁer learned using the new dataset attains good accuracy in both
standard and adversarial settings 7 8.
As a control, we repeat this methodology using a standard (non-robust) model for C in our construction
of the dataset. Sample images from the resulting “non-robust dataset” bDNR are shown in Figure 2a—they
tend to resemble more the source image of the optimization x0 than the target image x. We ﬁnd that training
on this dataset leads to good standard accuracy, yet yields almost no robustness (Figure 2b). We also verify
that this procedure is not simply a matter of encoding the weights of the original model—we get the same
results for both bDR and bDNR if we train with different architectures than that of the original models.
Overall, our ﬁndings corroborate the hypothesis that adversarial examples can arise from (non-robust)
features of the data itself. By ﬁltering out non-robust features from the dataset (e.g. by restricting the set of
available features to those used by a robust model), one can train a signiﬁcantly more robust model using
standard training.
3.2
Non-robust features sufﬁce for standard classiﬁcation
The results of the previous section show that by restricting the dataset to only contain features that are used
by a robust model, standard training results in classiﬁers that are signiﬁcantly more robust. This suggests
that when training on the standard dataset, non-robust features take on a large role in the resulting learned
classiﬁer. Here we set out to show that this role is not merely incidental or due to ﬁnite-sample overﬁtting.
In particular, we demonstrate that non-robust features alone sufﬁce for standard generalization— i.e., a
model trained solely on non-robust features can perform well on the standard test set.
To show this, we construct a dataset where the only features that are useful for classiﬁcation are non-
robust features (or in terms of our formal model from Section 2, all features f that are ρ-useful are non-
robust). To accomplish this, we modify each input-label pair (x, y) as follows. We select a target class t
either (a) uniformly at random among classes (hence features become uncorrelated with the labels) or (b)
deterministically according to the source class (e.g. using a ﬁxed permutation of labels). Then, we add a
small adversarial perturbation to x in order to ensure it is classiﬁed as t by a standard model. Formally:
xadv = arg min
∥x′−x∥≤ε
LC(x′, t),
(7)
where LC is the loss under a standard (non-robust) classiﬁer C and ε is a small constant. The resulting
inputs are nearly indistinguishable from the originals (Appendix D Figure 9)—to a human observer, it thus
appears that the label t assigned to the modiﬁed input is simply incorrect. The resulting input-label pairs
(xadv, t) make up the new training set (pseudocode in Appendix C Figure 6).
Now, since ∥xadv −x∥is small, by deﬁnition the robust features of xadv are still correlated with class
y (and not t) in expectation over the dataset. After all, humans still recognize the original class. On the
other hand, since every xadv is strongly classiﬁed as t by a standard classiﬁer, it must be that some of the
non-robust features are now strongly correlated with t (in expectation).
In the case where t is chosen at random, the robust features are originally uncorrelated with the label t (in
expectation), and after the adversarial perturbation can be only slightly correlated (hence being signiﬁcantly
7In an attempt to explain the gap in accuracy between the model trained on bDR and the original robust classiﬁer C, we test
distributional shift, by reporting results on the “robustiﬁed” test set in Appendix D.3.
8In order to gain more conﬁdence in the robustness of the resulting model, we attempt several diverse attacks in Appendix D.2.
6


**Table 7 from page 6**

| 0                                                                                                                 |
|:------------------------------------------------------------------------------------------------------------------|
| not be useful since they are not correlated with the label in expectation over x0. The underlying assumption      |
| here is that, when performing the optimization in (6),                                                            |
| features that are not being directly optimized (i.e.,                                                             |
| features outside FC) are not affected. We provide pseudocode for the construction in Figure 5 (Appendix C).       |
| Given the new training set for                                                                                    |
| R (a few random samples are visualized in Figure 2a), we train a clas-                                            |
| siﬁer using standard (non-robust) training. We then test this classiﬁer on the original test set (i.e.            |
| ). The                                                                                                            |
| (cid:98)D                                                                                                         |
| D                                                                                                                 |
| results (Figure 2b) indicate that the classiﬁer learned using the new dataset attains good accuracy in both       |
| standard and adversarial settings 7 8.                                                                            |
| As a control, we repeat this methodology using a standard (non-robust) model for C in our construction            |
| of the dataset. Sample images from the resulting “non-robust dataset”                                             |
| NR are shown in Figure 2a—they                                                                                    |
| (cid:98)D                                                                                                         |
| tend to resemble more the source image of the optimization x0 than the target image x. We ﬁnd that training       |
| on this dataset leads to good standard accuracy, yet yields almost no robustness (Figure 2b). We also verify      |
| that this procedure is not simply a matter of encoding the weights of the original model—we get the same          |
| results for both                                                                                                  |
| R and                                                                                                             |
| NR if we train with different architectures than that of the original models.                                     |
| (cid:98)D                                                                                                         |
| (cid:98)D                                                                                                         |
| Overall, our ﬁndings corroborate the hypothesis that adversarial examples can arise from (non-robust)             |
| features of the data itself. By ﬁltering out non-robust features from the dataset (e.g. by restricting the set of |
| available features to those used by a robust model), one can train a signiﬁcantly more robust model using         |
| standard training.                                                                                                |
| 3.2                                                                                                               |
| Non-robust features sufﬁce for standard classiﬁcation                                                             |
| The results of the previous section show that by restricting the dataset to only contain features that are used   |
| by a robust model, standard training results in classiﬁers that are signiﬁcantly more robust. This suggests       |
| that when training on the standard dataset, non-robust features take on a large role in the resulting learned     |
| classiﬁer. Here we set out to show that this role is not merely incidental or due to ﬁnite-sample overﬁtting.     |
| In particular, we demonstrate that non-robust                                                                     |
| features alone sufﬁce for standard generalization— i.e., a                                                        |
| model trained solely on non-robust features can perform well on the standard test set.                            |
| To show this, we construct a dataset where the only features that are useful for classiﬁcation are non-           |
| robust                                                                                                            |
| f                                                                                                                 |
| features (or in terms of our formal model                                                                         |
| from Section 2, all                                                                                               |
| features                                                                                                          |
| that are ρ-useful are non-                                                                                        |
| robust). To accomplish this, we modify each input-label pair (x, y) as follows. We select a target class t        |
| either (a) uniformly at random among classes (hence features become uncorrelated with the labels) or (b)          |
| deterministically according to the source class (e.g. using a ﬁxed permutation of labels). Then, we add a         |
| small adversarial perturbation to x in order to ensure it is classiﬁed as t by a standard model. Formally:        |



## Page 7

less useful for classiﬁcation than before) 9. Formally, we aim to construct a dataset bDrand where 10 :
E(x,y)∼bDrand [y · f (x)]
(
> 0
if f non-robustly useful under D,
≃0
otherwise.
(8)
In contrast, when t is chosen deterministically based on y, the robust features actually point away from
the assigned label t. In particular, all of the inputs labeled with class t exhibit non-robust features correlated
with t, but robust features correlated with the original class y. Thus, robust features on the original training
set provide signiﬁcant predictive power on the training set, but will actually hurt generalization on the
standard test set. Viewing this case again using the formal model, our goal is to construct bDdet such that
E(x,y)∼bDdet [y · f (x)]





> 0
if f non-robustly useful under D,
< 0
if f robustly useful under D
∈R
otherwise (f not useful under D)11
(9)
We ﬁnd that standard training on these datasets actually generalizes to the original test set, as shown in
Table 1). This indicates that non-robust features are indeed useful for classiﬁcation in the standard setting.
Remarkably, even training on bDdet (where all the robust features are correlated with the wrong class), results
in a well-generalizing classiﬁer. This indicates that non-robust features can be picked up by models during
standard training, even in the presence of robust features that are predictive 1213.
25
30
35
40
45
50
Test accuracy (%; trained on Dy + 1)
60
70
80
90
100
Transfer success rate (%)
VGG-16
Inception-v3
ResNet-18
DenseNet
ResNet-50
Figure 3:
Transfer rate of adversarial exam-
ples from a ResNet-50 to different architectures
alongside test set performance of these archi-
tecture when trained on the dataset generated
in Section 3.2. Architectures more susceptible
to transfer attacks also performed better on the
standard test set supporting our hypothesis that
adversarial transferability arises from utilizing
similar non-robust features.
Source Dataset
Dataset
CIFAR-10
ImageNetR
D
95.3%
96.6%
bDrand
63.3%
87.9%
bDdet
43.7%
64.4%
Table 1:
Test accuracy (on D) of classiﬁers
trained on the D, bDrand, and bDdet training sets
created using a standard (non-robust) model.
For both bDrand and bDdet, only non-robust fea-
tures correspond to useful features on both the
train set and D. These datasets are constructed
using adversarial perturbations of x towards a
class t (random for bDrand and deterministic for
bDdet); the resulting images are relabeled as t.
3.3
Transferability can arise from non-robust features
One of the most intriguing properties of adversarial examples is that they transfer across models with dif-
ferent architectures and independently sampled training sets [Sze+14; PMG16; CRP19]. Here, we show
9Goh [Goh19a] provides an approach to quantifying this “robust feature leakage” and ﬁnds that one can obtain a (small) amount
of test accuracy by leveraging robust feature leakage on bDrand.
10Note that the optimization procedure we describe aims to merely approximate this condition, where we once again use trained
models to simulate access to robust and non-robust features.
11 Note that regardless how useful a feature is on bDdet, since it is useless on D it cannot provide any generalization beneﬁt on the
unaltered test set.
12Additional results and analysis (e.g. training curves, generating bDrand and bDdet with a robust model, etc.) are in App. D.6 and D.5
13We also show that the models trained on bDrand and bDdet generalize to CIFAR-10.1 [Rec+19] in Appendix D.7.
7


**Table 8 from page 7**

| 0                                                                                                                |
|:-----------------------------------------------------------------------------------------------------------------|
| less useful for classiﬁcation than before) 9. Formally, we aim to construct a dataset                            |
| rand where 10 :                                                                                                  |
| (cid:98)D                                                                                                        |
| > 0                                                                                                              |
| if                                                                                                               |
| f non-robustly useful under                                                                                      |
| ,                                                                                                                |
| E                                                                                                                |
| [y                                                                                                               |
| f (x)]                                                                                                           |
| (8)                                                                                                              |
| D                                                                                                                |
| (x,y)                                                                                                            |
| rand                                                                                                             |
| ·                                                                                                                |
| 0                                                                                                                |
| otherwise.                                                                                                       |
| ∼                                                                                                                |
| (cid:40)                                                                                                         |
| (cid:98)D                                                                                                        |
| (cid:39)                                                                                                         |
| In contrast, when t is chosen deterministically based on y, the robust features actually point away from         |
| the assigned label t.                                                                                            |
| In particular, all of the inputs labeled with class t exhibit non-robust features correlated                     |
| with t, but robust features correlated with the original class y. Thus, robust features on the original training |
| set provide signiﬁcant predictive power on the training set, but will actually hurt generalization on the        |
| standard test set. Viewing this case again using the formal model, our goal is to construct                      |
| det such that                                                                                                    |
| (cid:98)D                                                                                                        |
| > 0                                                                                                              |
| if                                                                                                               |
| f non-robustly useful under                                                                                      |
| ,                                                                                                                |
| D                                                                                                                |
| E                                                                                                                |
| [y                                                                                                               |
| (9)                                                                                                              |
| f                                                                                                                |
| < 0                                                                                                              |
| if                                                                                                               |
| robustly useful under                                                                                            |
| f (x)]                                                                                                          |
| (x,y)                                                                                                            |
| det                                                                                                              |
| ·                                                                                                                |
| D                                                                                                                |
| ∼                                                                                                                |
| R                                                                                                                |
|                                                                                                                 |
| (cid:98)D                                                                                                        |
| otherwise ( f not useful under                                                                                   |
| )11                                                                                                              |
|                                                                                                               |
| ∈                                                                                                                |
| D                                                                                                                |
| We ﬁnd that standard training on these datasets actually generalizes to the original test set, as shown in       |
| Table 1). This indicates that non-robust features are indeed useful for classiﬁcation in the standard setting.   |
| Remarkably, even training on                                                                                     |
| det (where all the robust features are correlated with the wrong class), results                                 |
| (cid:98)D                                                                                                        |
| in a well-generalizing classiﬁer. This indicates that non-robust features can be picked up by models during      |
| standard training, even in the presence of robust features that are predictive 1213.                             |



## Page 8

that this phenomenon can in fact be viewed as a natural consequence of the existence of non-robust fea-
tures. Recall that, according to our main thesis, adversarial examples can arise as a result of perturbing
well-generalizing, yet brittle features. Given that such features are inherent to the data distribution, differ-
ent classiﬁers trained on independent samples from that distribution are likely to utilize similar non-robust
features. Consequently, an adversarial example constructed by exploiting the non-robust features learned
by one classiﬁer will transfer to any other classiﬁer utilizing these features in a similar manner.
In order to illustrate and corroborate this hypothesis, we train ﬁve different architectures on the dataset
generated in Section 3.2 (adversarial examples with deterministic labels) for a standard ResNet-50 [He+16].
Our hypothesis would suggest that architectures which learn better from this training set (in terms of per-
formance on the standard test set) are more likely to learn similar non-robust features to the original clas-
siﬁer. Indeed, we ﬁnd that the test accuracy of each architecture is predictive of how often adversarial
examples transfer from the original model to standard classiﬁers with that architecture (Figure 3). In a sim-
ilar vein, Nakkiran [Nak19a] constructs a set of adversarial perturbations that is explicitly non-transferable
and ﬁnds that these perturbations cannot be used to learn a good classiﬁer. These ﬁndings thus corrobo-
rate our hypothesis that adversarial transferability arises when models learn similar brittle features of the
underlying dataset.
4
A Theoretical Framework for Studying (Non)-Robust Features
The experiments from the previous section demonstrate that the conceptual framework of robust and non-
robust features is strongly predictive of the empirical behavior of state-of-the-art models on real-world
datasets. In order to further strengthen our understanding of the phenomenon, we instantiate the frame-
work in a concrete setting that allows us to theoretically study various properties of the corresponding
model. Our model is similar to that of Tsipras et al. [Tsi+19] in the sense that it contains a dichotomy
between robust and non-robust features, but extends upon it in a number of ways:
1. The adversarial vulnerability can be explicitly expressed as a difference between the inherent data
metric and the ℓ2 metric.
2. Robust learning corresponds exactly to learning a combination of these two metrics.
3. The gradients of adversarially trained models align better with the adversary’s metric.
Setup.
We study a simple problem of maximum likelihood classiﬁcation between two Gaussian distributions.
In particular, given samples (x, y) sampled from D according to
y u.a.r.
∼{−1, +1},
x ∼N (y · µ∗, Σ∗),
(10)
our goal is to learn parameters Θ = (µ, Σ) such that
Θ = arg min
µ,Σ E(x,y)∼D [ℓ(x; y · µ, Σ)] ,
(11)
where ℓ(x; µ, Σ) represents the Gaussian negative log-likelihood (NLL) function. Intuitively, we ﬁnd the
parameters µ, Σ which maximize the likelihood of the sampled data under the given model. Classiﬁcation
under this model can be accomplished via likelihood test: given an unlabeled sample x, we predict y as
y = arg max
y
ℓ(x; y · µ, Σ) = sign

x⊤Σ−1µ

.
In turn, the robust analogue of this problem arises from replacing ℓ(x; y · µ, Σ) with the NLL under adversarial
perturbation. The resulting robust parameters Θr can be written as
Θr = arg min
µ,Σ E(x,y)∼D

max
∥δ∥2≤ε ℓ(x + δ; y · µ, Σ)

,
(12)
A detailed analysis of this setting is in Appendix E—here we present a high-level overview of the results.
8


**Table 9 from page 8**

| 0                                                                                                                |
|:-----------------------------------------------------------------------------------------------------------------|
| that                                                                                                             |
| this phenomenon can in fact be viewed as a natural consequence of the existence of non-robust fea-               |
| tures. Recall                                                                                                    |
| that, according to our main thesis, adversarial examples can arise as a result of perturbing                     |
| well-generalizing, yet brittle features. Given that such features are inherent to the data distribution, differ- |
| ent classiﬁers trained on independent samples from that distribution are likely to utilize similar non-robust    |
| features. Consequently, an adversarial example constructed by exploiting the non-robust features learned         |
| by one classiﬁer will transfer to any other classiﬁer utilizing these features in a similar manner.              |
| In order to illustrate and corroborate this hypothesis, we train ﬁve different architectures on the dataset      |
| generated in Section 3.2 (adversarial examples with deterministic labels) for a standard ResNet-50 [He+16].      |
| Our hypothesis would suggest that architectures which learn better from this training set (in terms of per-      |
| formance on the standard test set) are more likely to learn similar non-robust features to the original clas-    |
| siﬁer.                                                                                                           |
| Indeed, we ﬁnd that                                                                                              |
| the test accuracy of each architecture is predictive of how often adversarial                                    |
| examples transfer from the original model to standard classiﬁers with that architecture (Figure 3). In a sim-    |
| ilar vein, Nakkiran [Nak19a] constructs a set of adversarial perturbations that is explicitly non-transferable   |
| and ﬁnds that these perturbations cannot be used to learn a good classiﬁer. These ﬁndings thus corrobo-          |
| rate our hypothesis that adversarial transferability arises when models learn similar brittle features of the    |
| underlying dataset.                                                                                              |



## Page 9

(1) Vulnerability from metric misalignment (non-robust features).
Note that in this model, one can rig-
orously make reference to an inner product (and thus a metric) induced by the features. In particular, one
can view the learned parameters of a Gaussian Θ = (µ, Σ) as deﬁning an inner product over the input space
given by ⟨x, y⟩Θ = (x −µ)⊤Σ−1(y −µ). This in turn induces the Mahalanobis distance, which represents
how a change in the input affects the features learned by the classiﬁer. This metric is not necessarily aligned
with the metric in which the adversary is constrained, the ℓ2-norm. Actually, we show that adversarial
vulnerability arises exactly as a misalignment of these two metrics.
Theorem 1 (Adversarial vulnerability from misalignment). Consider an adversary whose perturbation is deter-
mined by the “Lagrangian penalty” form of (12), i.e.
max
δ
ℓ(x + δ; y · µ, Σ) −C · ∥δ∥2,
where C ≥
1
σmin(Σ∗) is a constant trading off NLL minimization and the adversarial constraint14. Then, the adversarial
loss Ladv incurred by the non-robustly learned (µ, Σ) is given by:
Ladv(Θ) −L(Θ) = tr

I + (C · Σ∗−I)−12
−d,
and, for a ﬁxed tr(Σ∗) = k the above is minimized by Σ∗= k
d I.
In fact, note that such a misalignment corresponds precisely to the existence of non-robust features, as it
indicates that “small” changes in the adversary’s metric along certain directions can cause large changes
under the data-dependent notion of distance established by the parameters. This is illustrated in Figure 4,
where misalignment in the feature-induced metric is responsible for the presence of a non-robust feature in
the corresponding classiﬁcation problem.
(2) Robust Learning.
The optimal (non-robust) maximum likelihood estimate is Θ = Θ∗, and thus the
vulnerability for the standard MLE estimate is governed entirely by the true data distribution. The follow-
ing theorem characterizes the behaviour of the learned parameters in the robust problem. 15. In fact, we can
prove (Section E.3.4) that performing (sub)gradient descent on the inner maximization (also known as ad-
versarial training [GSS15; Mad+18]) yields exactly Θr. We ﬁnd that as the perturbation budget ε is increased,
the metric induced by the learned features mixes ℓ2 and the metric induced by the features.
Theorem 2 (Robustly Learned Parameters). Just as in the non-robust case, µr = µ∗, i.e. the true mean is learned.
For the robust covariance Σr, there exists an ε0 > 0, such that for any ε ∈[0, ε0),
Σr = 1
2Σ∗+ 1
λ · I +
r
1
λ · Σ∗+ 1
4Σ2∗,
where
Ω
 
1 + ε1/2
ε1/2 + ε3/2
!
≤λ ≤O
 
1 + ε1/2
ε1/2
!
.
The effect of robust optimization under an ℓ2-constrained adversary is visualized in Figure 4. As ϵ
grows, the learned covariance becomes more aligned with identity. For instance, we can see that the classi-
ﬁer learns to be less sensitive in certain directions, despite their usefulness for natural classiﬁcation.
(3) Gradient Interpretability.
Tsipras et al. [Tsi+19] observe that gradients of robust models tend to look
more semantically meaningful. It turns out that under our model, this behaviour arises as a natural con-
sequence of Theorem 2. In particular, we show that the resulting robustly learned parameters cause the
gradient of the linear classiﬁer and the vector connecting the means of the two distributions to better align
(in a worst-case sense) under the ℓ2 inner product.
Theorem 3 (Gradient alignment). Let f (x) and fr(x) be monotonic classiﬁers based on the linear separator induced
by standard and ℓ2-robust maximum likelihood classiﬁcation, respectively. The maximum angle formed between the
gradient of the classiﬁer (wrt input) and the vector connecting the classes can be smaller for the robust model:
min
µ
⟨µ, ∇x fr(x)⟩
∥µ∥· ∥∇x fr(x)∥> min
µ
⟨µ, ∇x f (x)⟩
∥µ∥· ∥∇x f (x)∥.
14The constraint on C is to ensure the problem is concave.
15Note: as discussed in Appendix E.3.3, we study a slight relaxation of (12) that approaches exactness exponentially fast as d →∞
9


**Table 10 from page 9**

| 0                                                                                                               |
|:----------------------------------------------------------------------------------------------------------------|
| (1) Vulnerability from metric misalignment (non-robust features).                                               |
| Note that in this model, one can rig-                                                                           |
| orously make reference to an inner product (and thus a metric) induced by the features.                         |
| In particular, one                                                                                              |
| can view the learned parameters of a Gaussian Θ = (µ, Σ) as deﬁning an inner product over the input space       |
| Σ                                                                                                               |
| 1(y                                                                                                             |
| given by                                                                                                        |
| x, y                                                                                                            |
| µ). This in turn induces the Mahalanobis distance, which represents                                             |
| Θ = (x                                                                                                          |
| µ)(cid:62)                                                                                                      |
| −                                                                                                               |
| −                                                                                                               |
| −                                                                                                               |
| (cid:104)                                                                                                       |
| (cid:105)                                                                                                       |
| how a change in the input affects the features learned by the classiﬁer. This metric is not necessarily aligned |
| with the metric in which the adversary is constrained,                                                          |
| the (cid:96)2-norm. Actually, we show that adversarial                                                          |
| vulnerability arises exactly as a misalignment of these two metrics.                                            |
| Theorem 1 (Adversarial vulnerability from misalignment). Consider an adversary whose perturbation is deter-     |
| mined by the “Lagrangian penalty” form of                                                                       |
| (12), i.e.                                                                                                      |
| C                                                                                                               |
| δ                                                                                                               |
| max                                                                                                             |
| (cid:96)(x + δ; y                                                                                               |
| µ, Σ)                                                                                                           |
| δ                                                                                                               |
| ·                                                                                                               |
| −                                                                                                               |
| · (cid:107)                                                                                                     |
| (cid:107)2,                                                                                                     |
| 1                                                                                                               |
| is a constant trading off NLL minimization and the adversarial constraint14. Then, the adversarial              |
| where C                                                                                                         |
| )                                                                                                               |
| σmin(Σ                                                                                                          |
| ≥                                                                                                               |
| loss                                                                                                            |
| Ladv incurred by the non-robustly learned (µ, Σ) is given by:                                                   |
| 2                                                                                                               |



## Page 10

Figure 4 illustrates this phenomenon in the two-dimensional case. With ℓ2-bounded adversarial training
the gradient direction (perpendicular to the decision boundary) becomes increasingly aligned under the ℓ2
inner product with the vector between the means (µ).
Discussion.
Our theoretical analysis suggests that rather than offering any quantitative classiﬁcation ben-
eﬁts, a natural way to view the role of robust optimization is as enforcing a prior over the features learned
by the classiﬁer. In particular, training with an ℓ2-bounded adversary prevents the classiﬁer from relying
heavily on features which induce a metric dissimilar to the ℓ2 metric. The strength of the adversary then
allows for a trade-off between the enforced prior, and the data-dependent features.
Robustness and accuracy.
Note that in the setting described so far, robustness can be at odds with ac-
curacy since robust training prevents us from learning the most accurate classiﬁer (a similar conclusion
is drawn in [Tsi+19]). However, we note that there are very similar settings where non-robust features
manifest themselves in the same way, yet a classiﬁer with perfect robustness and accuracy is still attainable.
Concretely, consider the distributions pictured in Figure 14 in Appendix D.10. It is straightforward to show
that while there are many perfectly accurate classiﬁers, any standard loss function will learn an accurate yet
non-robust classiﬁer. Only when robust training is employed does the classiﬁer learn a perfectly accurate
and perfectly robust decision boundary.
5
Related Work
Several models for explaining adversarial examples have been proposed in prior work, utilizing ideas rang-
ing from ﬁnite-sample overﬁtting to high-dimensional statistical phenomena [Gil+18; FFF18; For+19; TG16;
Sha+19a; MDM18; Sha+19b; GSS15; BPR18]. The key differentiating aspect of our model is that adversar-
ial perturbations arise as well-generalizing, yet brittle, features, rather than statistical anomalies or effects of
poor statistical concentration. In particular, adversarial vulnerability does not stem from using a speciﬁc
model class or a speciﬁc training method, since standard training on the “robustiﬁed” data distribution of
Section 3.1 leads to robust models. At the same time, as shown in Section 3.2, these non-robust features are
sufﬁcient to learn a good standard classiﬁer. We discuss the connection between our model and others in
detail in Appendix A. We discuss additional related work in Appendix B.
6
Conclusion
In this work, we cast the phenomenon of adversarial examples as a natural consequence of the presence of
highly predictive but non-robust features in standard ML datasets. We provide support for this hypothesis by
20
15
10
5
0
5
10
15
20
Feature x1
10.0
7.5
5.0
2.5
0.0
2.5
5.0
7.5
10.0
Feature x2
Maximum likelihood estimate
2 unit ball
1-induced metric unit ball
Samples from 
(0,
)
20
15
10
5
0
5
10
15
20
Feature x1
10.0
7.5
5.0
2.5
0.0
2.5
5.0
7.5
10.0
Feature x2
True Parameters ( = 0)
Samples from 
( ,
)
Samples from 
(
,
)
20
15
10
5
0
5
10
15
20
Feature x1
10.0
7.5
5.0
2.5
0.0
2.5
5.0
7.5
10.0
Feature x2
Robust parameters, = 1.0
20
15
10
5
0
5
10
15
20
Feature x1
10.0
7.5
5.0
2.5
0.0
2.5
5.0
7.5
10.0
Feature x2
Robust parameters, = 10.0
Figure 4: An empirical demonstration of the effect illustrated by Theorem 2—as the adversarial perturba-
tion budget ε is increased, the learned mean µ remains constant, but the learned covariance “blends” with
the identity matrix, effectively adding more and more uncertainty onto the non-robust feature.
10


**Table 11 from page 10**

| 0                                                                                                                |
|:-----------------------------------------------------------------------------------------------------------------|
| Figure 4 illustrates this phenomenon in the two-dimensional case. With (cid:96)2-bounded adversarial training    |
| the gradient direction (perpendicular to the decision boundary) becomes increasingly aligned under the (cid:96)2 |
| inner product with the vector between the means (µ).                                                             |
| Discussion.                                                                                                      |
| Our theoretical analysis suggests that rather than offering any quantitative classiﬁcation ben-                  |
| eﬁts, a natural way to view the role of robust optimization is as enforcing a prior over the features learned    |
| by the classiﬁer.                                                                                                |
| In particular, training with an (cid:96)2-bounded adversary prevents the classiﬁer from relying                  |
| heavily on features which induce a metric dissimilar to the (cid:96)2 metric. The strength of the adversary then |
| allows for a trade-off between the enforced prior, and the data-dependent features.                              |
| Robustness and accuracy.                                                                                         |
| Note that                                                                                                        |
| in the setting described so far, robustness can be at odds with ac-                                              |
| curacy since robust                                                                                              |
| training prevents us from learning the most accurate classiﬁer (a similar conclusion                             |
| is drawn in [Tsi+19]). However, we note that                                                                     |
| there are very similar settings where non-robust                                                                 |
| features                                                                                                         |
| manifest themselves in the same way, yet a classiﬁer with perfect robustness and accuracy is still attainable.   |
| Concretely, consider the distributions pictured in Figure 14 in Appendix D.10. It is straightforward to show     |
| that while there are many perfectly accurate classiﬁers, any standard loss function will learn an accurate yet   |
| non-robust classiﬁer. Only when robust training is employed does the classiﬁer learn a perfectly accurate        |
| and perfectly robust decision boundary.                                                                          |

**Table 12 from page 10**

| 0          | 1   | 2   | 3                           | 4   | 5          | 6            | 7                          | 8   | 9   | 10         | 11   | 12   | 13   | 14                     | 15         | 16   | 17           | 18   | 19   | 20         | 21   | 22   | 23   | 24                        | 25         | 26   | 27   | 28   | 29   | 30         |
|:-----------|:----|:----|:----------------------------|:----|:-----------|:-------------|:---------------------------|:----|:----|:-----------|:-----|:-----|:-----|:-----------------------|:-----------|:-----|:-------------|:-----|:-----|:-----------|:-----|:-----|:-----|:--------------------------|:-----------|:-----|:-----|:-----|:-----|:-----------|
| 10.0       |     |     | Maximum likelihood estimate |     |            |              |                            |     |     | 10.0       |      |      |      | True Parameters ( = 0) |            |      |              |      |      | 10.0       |      |      |      | Robust parameters,  = 1.0 |            |      |      |      |      | 10.0       |
|            |     |     |                             |     |            | 2 unit ball  |                            |     |     |            |      |      |      |                        |            |      | Samples from | (    | )    |            |      |      |      |                           |            |      |      |      |      |            |
|            |     |     |                             |     |            |              |                            |     |     |            |      |      |      |                        |            |      |              | ,    |      |            |      |      |      |                           |            |      |      |      |      |            |
|            |     |     |                             |     |            |              | 1-induced metric unit ball |     |     |            |      |      |      |                        |            |      | Samples from | (    | )    |            |      |      |      |                           |            |      |      |      |      |            |
|            |     |     |                             |     |            |              |                            |     |     |            |      |      |      |                        |            |      |              | ,    |      |            |      |      |      |                           |            |      |      |      |      |            |
| 7.5        |     |     |                             |     |            | Samples from |                            | (0, |     | 7.5        |      |      |      |                        |            |      |              |      |      | 7.5        |      |      |      |                           |            |      |      |      |      | 7.5        |
|            |     |     |                             |     |            |              |                            | )   |     |            |      |      |      |                        |            |      |              |      |      |            |      |      |      |                           |            |      |      |      |      |            |
| 5.0        |     |     |                             |     |            |              |                            |     |     | 5.0        |      |      |      |                        |            |      |              |      |      | 5.0        |      |      |      |                           |            |      |      |      |      | 5.0        |
| 2.5        |     |     |                             |     |            |              |                            |     |     | 2.5        |      |      |      |                        |            |      |              |      |      | 2.5        |      |      |      |                           |            |      |      |      |      | 2.5        |
| Feature x2 |     |     |                             |     |            |              |                            |     |     | Feature x2 |      |      |      |                        |            |      |              |      |      | Feature x2 |      |      |      |                           |            |      |      |      |      | Feature x2 |
| 0.0        |     |     |                             |     |            |              |                            |     |     | 0.0        |      |      |      |                        |            |      |              |      |      | 0.0        |      |      |      |                           |            |      |      |      |      | 0.0        |
| 2.5        |     |     |                             |     |            |              |                            |     |     | 2.5        |      |      |      |                        |            |      |              |      |      | 2.5        |      |      |      |                           |            |      |      |      |      | 2.5        |
| 5.0        |     |     |                             |     |            |              |                            |     |     | 5.0        |      |      |      |                        |            |      |              |      |      | 5.0        |      |      |      |                           |            |      |      |      |      | 5.0        |
| 7.5        |     |     |                             |     |            |              |                            |     |     | 7.5        |      |      |      |                        |            |      |              |      |      | 7.5        |      |      |      |                           |            |      |      |      |      | 7.5        |
| 10.0       |     |     |                             |     |            |              |                            |     |     | 10.0       |      |      |      |                        |            |      |              |      |      | 10.0       |      |      |      |                           |            |      |      |      |      | 10.0       |
|            | 20  | 15  | 10                          | 5   | 0          | 5            | 10                         | 15  | 20  |            | 20   | 15   | 10   | 5                      | 0          | 5    | 10           | 15   | 20   |            | 20   | 15   | 10   | 5                         | 0          | 5    | 10   | 15   | 20   | 20         |
|            |     |     |                             |     | Feature x1 |              |                            |     |     |            |      |      |      |                        | Feature x1 |      |              |      |      |            |      |      |      |                           | Feature x1 |      |      |      |      |            |



## Page 11

explicitly disentangling robust and non-robust features in standard datasets, as well as showing that non-
robust features alone are sufﬁcient for good generalization. Finally, we study these phenomena in more
detail in a theoretical setting where we can rigorously study adversarial vulnerability, robust training, and
gradient alignment.
Our ﬁndings prompt us to view adversarial examples as a fundamentally human phenomenon. In par-
ticular, we should not be surprised that classiﬁers exploit highly predictive features that happen to be
non-robust under a human-selected notion of similarity, given such features exist in real-world datasets.
In the same manner, from the perspective of interpretability, as long as models rely on these non-robust
features, we cannot expect to have model explanations that are both human-meaningful and faithful to
the models themselves. Overall, attaining models that are robust and interpretable will require explicitly
encoding human priors into the training process.
7
Acknowledgements
We thank Preetum Nakkiran for suggesting the experiment of Appendix D.9 (i.e. replicating Figure 3 but
with targeted attacks). We also are grateful to the authors of Engstrom et al. [Eng+19a] (Chris Olah, Dan
Hendrycks, Justin Gilmer, Reiichiro Nakano, Preetum Nakkiran, Gabriel Goh, Eric Wallace)—for their in-
sights and efforts replicating, extending, and discussing our experimental results.
Work supported in part by the NSF grants CCF-1553428, CCF-1563880, CNS-1413920, CNS-1815221,
IIS-1447786, IIS-1607189, the Microsoft Corporation, the Intel Corporation, the MIT-IBM Watson AI Lab
research grant, and an Analog Devices Fellowship.
References
[ACW18]
Anish Athalye, Nicholas Carlini, and David A. Wagner. “Obfuscated Gradients Give a False
Sense of Security: Circumventing Defenses to Adversarial Examples”. In: International Confer-
ence on Machine Learning (ICML). 2018.
[Ath+18]
Anish Athalye et al. “Synthesizing Robust Adversarial Examples”. In: International Conference
on Machine Learning (ICML). 2018.
[BCN06]
Cristian Buciluˇa, Rich Caruana, and Alexandru Niculescu-Mizil. “Model compression”. In: In-
ternational Conference on Knowledge Discovery and Data Mining (KDD). 2006.
[Big+13]
Battista Biggio et al. “Evasion attacks against machine learning at test time”. In: Joint European
conference on machine learning and knowledge discovery in databases (ECML-KDD). 2013.
[BPR18]
Sébastien Bubeck, Eric Price, and Ilya Razenshteyn. “Adversarial examples from computa-
tional constraints”. In: arXiv preprint arXiv:1805.10204. 2018.
[Car+19]
Nicholas Carlini et al. “On Evaluating Adversarial Robustness”. In: ArXiv preprint arXiv:1902.06705.
2019.
[CRK19]
Jeremy M Cohen, Elan Rosenfeld, and J Zico Kolter. “Certiﬁed adversarial robustness via ran-
domized smoothing”. In: arXiv preprint arXiv:1902.02918. 2019.
[CRP19]
Zachary Charles, Harrison Rosenberg, and Dimitris Papailiopoulos. “A Geometric Perspec-
tive on the Transferability of Adversarial Directions”. In: International Conference on Artiﬁcial
Intelligence and Statistics (AISTATS). 2019.
[CW17a]
Nicholas Carlini and David Wagner. “Adversarial Examples Are Not Easily Detected: Bypass-
ing Ten Detection Methods”. In: Workshop on Artiﬁcial Intelligence and Security (AISec). 2017.
[CW17b]
Nicholas Carlini and David Wagner. “Towards evaluating the robustness of neural networks”.
In: Symposium on Security and Privacy (SP). 2017.
[Dan67]
John M. Danskin. The Theory of Max-Min and its Application to Weapons Allocation Problems. 1967.
[Das+19]
Constantinos Daskalakis et al. “Efﬁcient Statistics, in High Dimensions, from Truncated Sam-
ples”. In: Foundations of Computer Science (FOCS). 2019.
11


**Table 13 from page 11**

| 0                                                                                                             |
|:--------------------------------------------------------------------------------------------------------------|
| explicitly disentangling robust and non-robust features in standard datasets, as well as showing that non-    |
| robust                                                                                                        |
| features alone are sufﬁcient                                                                                  |
| for good generalization.                                                                                      |
| Finally, we study these phenomena in more                                                                     |
| detail in a theoretical setting where we can rigorously study adversarial vulnerability, robust training, and |
| gradient alignment.                                                                                           |
| Our ﬁndings prompt us to view adversarial examples as a fundamentally human phenomenon.                       |
| In par-                                                                                                       |
| ticular, we should not be surprised that classiﬁers exploit highly predictive features that happen to be      |
| non-robust under a human-selected notion of similarity, given such features exist                             |
| in real-world datasets.                                                                                       |
| In the same manner,                                                                                           |
| from the perspective of                                                                                       |
| interpretability, as long as models rely on these non-robust                                                  |
| features, we cannot expect                                                                                    |
| to have model explanations that are both human-meaningful and faithful                                        |
| to                                                                                                            |
| the models themselves. Overall, attaining models that are robust and interpretable will require explicitly    |
| encoding human priors into the training process.                                                              |

**Table 14 from page 11**

| 0                                                 | 1                                                                                                    |
|:--------------------------------------------------|:-----------------------------------------------------------------------------------------------------|
| research grant, and an Analog Devices Fellowship. |                                                                                                      |
| References                                        |                                                                                                      |
| [ACW18]                                           | Anish Athalye, Nicholas Carlini, and David A. Wagner. “Obfuscated Gradients Give a False             |
|                                                   | Sense of Security: Circumventing Defenses to Adversarial Examples”. In: International Confer-        |
|                                                   | ence on Machine Learning (ICML). 2018.                                                               |
| [Ath+18]                                          | Anish Athalye et al. “Synthesizing Robust Adversarial Examples”. In: International Conference        |
|                                                   | on Machine Learning (ICML). 2018.                                                                    |
| [BCN06]                                           | Cristian Buciluˇa, Rich Caruana, and Alexandru Niculescu-Mizil. “Model compression”. In: In-         |
|                                                   | ternational Conference on Knowledge Discovery and Data Mining (KDD). 2006.                           |
| [Big+13]                                          | Battista Biggio et al. “Evasion attacks against machine learning at test time”. In: Joint European   |
|                                                   | conference on machine learning and knowledge discovery in databases (ECML-KDD). 2013.                |
| [BPR18]                                           | Sébastien Bubeck, Eric Price, and Ilya Razenshteyn. “Adversarial examples from computa-              |
|                                                   | tional constraints”. In: arXiv preprint arXiv:1805.10204. 2018.                                      |
| [Car+19]                                          | Nicholas Carlini et al. “On Evaluating Adversarial Robustness”. In: ArXiv preprint arXiv:1902.06705. |
|                                                   | 2019.                                                                                                |
| [CRK19]                                           | Jeremy M Cohen, Elan Rosenfeld, and J Zico Kolter. “Certiﬁed adversarial robustness via ran-         |
|                                                   | domized smoothing”. In: arXiv preprint arXiv:1902.02918. 2019.                                       |
| [CRP19]                                           | Zachary Charles, Harrison Rosenberg, and Dimitris Papailiopoulos. “A Geometric Perspec-              |
|                                                   | International Conference on Artiﬁcial                                                                |
|                                                   | tive on the Transferability of Adversarial Directions”.                                              |
|                                                   | In:                                                                                                  |
|                                                   | Intelligence and Statistics (AISTATS). 2019.                                                         |
| [CW17a]                                           | Nicholas Carlini and David Wagner. “Adversarial Examples Are Not Easily Detected: Bypass-            |
|                                                   | ing Ten Detection Methods”. In: Workshop on Artiﬁcial Intelligence and Security (AISec). 2017.       |
| [CW17b]                                           | Nicholas Carlini and David Wagner. “Towards evaluating the robustness of neural networks”.           |
|                                                   | In: Symposium on Security and Privacy (SP). 2017.                                                    |
| [Dan67]                                           | John M. Danskin. The Theory of Max-Min and its Application to Weapons Allocation Problems. 1967.     |
| [Das+19]                                          | Constantinos Daskalakis et al. “Efﬁcient Statistics, in High Dimensions, from Truncated Sam-         |
|                                                   | ples”. In: Foundations of Computer Science (FOCS). 2019.                                             |



## Page 12

[Din+19]
Gavin Weiguang Ding et al. “On the Sensitivity of Adversarial Robustness to Input Data Dis-
tributions”. In: International Conference on Learning Representations. 2019.
[Eng+19a]
Logan Engstrom et al. “A Discussion of ’Adversarial Examples Are Not Bugs, They Are Fea-
tures’”. In: Distill (2019). https://distill.pub/2019/advex-bugs-discussion. DOI: 10 . 23915 /
distill.00019.
[Eng+19b]
Logan Engstrom et al. “A Rotation and a Translation Sufﬁce: Fooling CNNs with Simple Trans-
formations”. In: International Conference on Machine Learning (ICML). 2019.
[FFF18]
Alhussein Fawzi, Hamza Fawzi, and Omar Fawzi. “Adversarial vulnerability for any classi-
ﬁer”. In: Advances in Neural Information Processing Systems (NeuRIPS). 2018.
[FMF16]
Alhussein Fawzi, Seyed-Mohsen Moosavi-Dezfooli, and Pascal Frossard. “Robustness of clas-
siﬁers: from adversarial to random noise”. In: Advances in Neural Information Processing Systems.
2016.
[For+19]
Nic Ford et al. “Adversarial Examples Are a Natural Consequence of Test Error in Noise”. In:
arXiv preprint arXiv:1901.10513. 2019.
[Fur+18]
Tommaso Furlanello et al. “Born-Again Neural Networks”. In: International Conference on Ma-
chine Learning (ICML). 2018.
[Gei+19]
Robert Geirhos et al. “ImageNet-trained CNNs are biased towards texture; increasing shape
bias improves accuracy and robustness.” In: International Conference on Learning Representations.
2019.
[Gil+18]
Justin Gilmer et al. “Adversarial spheres”. In: Workshop of International Conference on Learning
Representations (ICLR). 2018.
[Goh19a]
Gabriel Goh. “A Discussion of ’Adversarial Examples Are Not Bugs, They Are Features’: Ro-
bust Feature Leakage”. In: Distill (2019). https://distill.pub/2019/advex-bugs-discussion/response-
2. DOI: 10.23915/distill.00019.2.
[Goh19b]
Gabriel Goh. “A Discussion of ’Adversarial Examples Are Not Bugs, They Are Features’: Two
Examples of Useful, Non-Robust Features”. In: Distill (2019). https://distill.pub/2019/advex-
bugs-discussion/response-3. DOI: 10.23915/distill.00019.3.
[GSS15]
Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. “Explaining and Harnessing Ad-
versarial Examples”. In: International Conference on Learning Representations (ICLR). 2015.
[HD19]
Dan Hendrycks and Thomas G. Dietterich. “Benchmarking Neural Network Robustness to
Common Corruptions and Surface Variations”. In: International Conference on Learning Repre-
sentations (ICLR). 2019.
[He+16]
Kaiming He et al. “Deep Residual Learning for Image Recognition”. In: Conference on Computer
Vision and Pattern Recognition (CVPR). 2016.
[He+17]
Warren He et al. “Adversarial example defense: Ensembles of weak defenses are not strong”.
In: USENIX Workshop on Offensive Technologies (WOOT). 2017.
[HVD14]
Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. “Distilling the Knowledge in a Neural Net-
work”. In: Neural Information Processing Systems (NeurIPS) Deep Learning Workshop. 2014.
[JLT18]
Saumya Jetley, Nicholas Lord, and Philip Torr. “With friends like these, who needs adver-
saries?” In: Advances in Neural Information Processing Systems (NeurIPS). 2018.
[Kri09]
Alex Krizhevsky. “Learning Multiple Layers of Features from Tiny Images”. In: Technical report.
2009.
[KSJ19]
Beomsu Kim, Junghoon Seo, and Taegyun Jeon. “Bridging Adversarial Robustness and Gra-
dient Interpretability”. In: International Conference on Learning Representations Workshop on Safe
Machine Learning (ICLR SafeML). 2019.
[Lec+19]
Mathias Lecuyer et al. “Certiﬁed robustness to adversarial examples with differential privacy”.
In: Symposium on Security and Privacy (SP). 2019.
12


**Table 15 from page 12**

| 0         | 1                                                                                                   |
|:----------|:----------------------------------------------------------------------------------------------------|
| [Din+19]  | Gavin Weiguang Ding et al. “On the Sensitivity of Adversarial Robustness to Input Data Dis-         |
|           | tributions”. In: International Conference on Learning Representations. 2019.                        |
| [Eng+19a] | Logan Engstrom et al. “A Discussion of ’Adversarial Examples Are Not Bugs, They Are Fea-            |
|           | tures’”.                                                                                            |
|           | In: Distill                                                                                         |
|           | (2019). https://distill.pub/2019/advex-bugs-discussion. DOI: 10 . 23915 /                           |
|           | distill.00019.                                                                                      |
| [Eng+19b] | Logan Engstrom et al. “A Rotation and a Translation Sufﬁce: Fooling CNNs with Simple Trans-         |
|           | formations”. In: International Conference on Machine Learning (ICML). 2019.                         |
| [FFF18]   | Alhussein Fawzi, Hamza Fawzi, and Omar Fawzi. “Adversarial vulnerability for any classi-            |
|           | ﬁer”. In: Advances in Neural Information Processing Systems (NeuRIPS). 2018.                        |
| [FMF16]   | Alhussein Fawzi, Seyed-Mohsen Moosavi-Dezfooli, and Pascal Frossard. “Robustness of clas-           |
|           | siﬁers: from adversarial to random noise”. In: Advances in Neural Information Processing Systems.   |
|           | 2016.                                                                                               |
| [For+19]  | Nic Ford et al. “Adversarial Examples Are a Natural Consequence of Test Error in Noise”. In:        |
|           | arXiv preprint arXiv:1901.10513. 2019.                                                              |
| [Fur+18]  | Tommaso Furlanello et al. “Born-Again Neural Networks”. In: International Conference on Ma-         |
|           | chine Learning (ICML). 2018.                                                                        |
| [Gei+19]  | Robert Geirhos et al. “ImageNet-trained CNNs are biased towards texture;                            |
|           | increasing shape                                                                                    |
|           | bias improves accuracy and robustness.” In: International Conference on Learning Representations.   |
|           | 2019.                                                                                               |
| [Gil+18]  | Justin Gilmer et al. “Adversarial spheres”. In: Workshop of International Conference on Learning    |
|           | Representations (ICLR). 2018.                                                                       |
| [Goh19a]  | Gabriel Goh. “A Discussion of ’Adversarial Examples Are Not Bugs, They Are Features’: Ro-           |
|           | bust Feature Leakage”. In: Distill (2019). https://distill.pub/2019/advex-bugs-discussion/response- |
|           | 2. DOI: 10.23915/distill.00019.2.                                                                   |
| [Goh19b]  | Gabriel Goh. “A Discussion of ’Adversarial Examples Are Not Bugs, They Are Features’: Two           |
|           | Examples of Useful, Non-Robust Features”. In: Distill (2019). https://distill.pub/2019/advex-       |
|           | bugs-discussion/response-3. DOI: 10.23915/distill.00019.3.                                          |
| [GSS15]   | Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. “Explaining and Harnessing Ad-            |
|           | versarial Examples”. In: International Conference on Learning Representations (ICLR). 2015.         |
| [HD19]    | Dan Hendrycks and Thomas G. Dietterich. “Benchmarking Neural Network Robustness to                  |
|           | International Conference on Learning Repre-                                                         |
|           | Common Corruptions and Surface Variations”.                                                         |
|           | In:                                                                                                 |
|           | sentations (ICLR). 2019.                                                                            |
| [He+16]   | Kaiming He et al. “Deep Residual Learning for Image Recognition”. In: Conference on Computer        |
|           | Vision and Pattern Recognition (CVPR). 2016.                                                        |
| [He+17]   | Warren He et al. “Adversarial example defense: Ensembles of weak defenses are not strong”.          |
|           | In: USENIX Workshop on Offensive Technologies (WOOT). 2017.                                         |
| [HVD14]   | Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. “Distilling the Knowledge in a Neural Net-           |
|           | work”. In: Neural Information Processing Systems (NeurIPS) Deep Learning Workshop. 2014.            |
| [JLT18]   | Saumya Jetley, Nicholas Lord, and Philip Torr. “With friends like these, who needs adver-           |
|           | saries?” In: Advances in Neural Information Processing Systems (NeurIPS). 2018.                     |
| [Kri09]   | Alex Krizhevsky. “Learning Multiple Layers of Features from Tiny Images”. In: Technical report.     |
|           | 2009.                                                                                               |
| [KSJ19]   | Beomsu Kim, Junghoon Seo, and Taegyun Jeon. “Bridging Adversarial Robustness and Gra-               |
|           | dient Interpretability”. In: International Conference on Learning Representations Workshop on Safe  |
|           | Machine Learning (ICLR SafeML). 2019.                                                               |
| [Lec+19]  | Mathias Lecuyer et al. “Certiﬁed robustness to adversarial examples with differential privacy”.     |
|           | In: Symposium on Security and Privacy (SP). 2019.                                                   |



## Page 13

[Liu+17]
Yanpei Liu et al. “Delving into Transferable Adversarial Examples and Black-box Attacks”. In:
International Conference on Learning Representations (ICLR). 2017.
[LM00]
Beatrice Laurent and Pascal Massart. “Adaptive estimation of a quadratic functional by model
selection”. In: Annals of Statistics. 2000.
[Mad+18]
Aleksander Madry et al. “Towards deep learning models resistant to adversarial attacks”. In:
International Conference on Learning Representations (ICLR). 2018.
[MDM18]
Saeed Mahloujifar, Dimitrios I Diochnos, and Mohammad Mahmoody. “The curse of concen-
tration in robust learning: Evasion and poisoning attacks from concentration of measure”. In:
AAAI Conference on Artiﬁcial Intelligence (AAAI). 2018.
[Moo+17]
Seyed-Mohsen Moosavi-Dezfooli et al. “Universal adversarial perturbations”. In: conference on
computer vision and pattern recognition (CVPR). 2017.
[MV15]
Aravindh Mahendran and Andrea Vedaldi. “Understanding deep image representations by
inverting them”. In: computer vision and pattern recognition (CVPR). 2015.
[Nak19a]
Preetum Nakkiran. “A Discussion of ’Adversarial Examples Are Not Bugs, They Are Features’:
Adversarial Examples are Just Bugs, Too”. In: Distill (2019). https://distill.pub/2019/advex-
bugs-discussion/response-5. DOI: 10.23915/distill.00019.5.
[Nak19b]
Preetum Nakkiran. “Adversarial robustness may be at odds with simplicity”. In: arXiv preprint
arXiv:1901.00532. 2019.
[OMS17]
Chris Olah, Alexander Mordvintsev, and Ludwig Schubert. “Feature Visualization”. In: Distill.
2017.
[Pap+17]
Nicolas Papernot et al. “Practical black-box attacks against machine learning”. In: Asia Confer-
ence on Computer and Communications Security. 2017.
[PMG16]
Nicolas Papernot, Patrick McDaniel, and Ian Goodfellow. “Transferability in Machine Learn-
ing: from Phenomena to Black-box Attacks using Adversarial Samples”. In: ArXiv preprint
arXiv:1605.07277. 2016.
[Rec+19]
Benjamin Recht et al. “Do CIFAR-10 Classiﬁers Generalize to CIFAR-10?” In: International Con-
ference on Machine Learning (ICML). 2019.
[RSL18]
Aditi Raghunathan, Jacob Steinhardt, and Percy Liang. “Certiﬁed defenses against adversarial
examples”. In: International Conference on Learning Representations (ICLR). 2018.
[Rus+15]
Olga Russakovsky et al. “ImageNet Large Scale Visual Recognition Challenge”. In: International
Journal of Computer Vision (IJCV). 2015.
[Sch+18]
Ludwig Schmidt et al. “Adversarially Robust Generalization Requires More Data”. In: Ad-
vances in Neural Information Processing Systems (NeurIPS). 2018.
[Sha+19a]
Ali Shafahi et al. “Are adversarial examples inevitable?” In: International Conference on Learning
Representations (ICLR). 2019.
[Sha+19b]
Adi Shamir et al. “A Simple Explanation for the Existence of Adversarial Examples with Small
Hamming Distance”. In: arXiv preprint arXiv:1901.10861. 2019.
[SHS19]
David Stutz, Matthias Hein, and Bernt Schiele. “Disentangling Adversarial Robustness and
Generalization”. In: Computer Vision and Pattern Recognition (CVPR). 2019.
[Smi+17]
D. Smilkov et al. “SmoothGrad: removing noise by adding noise”. In: ICML workshop on visu-
alization for deep learning. 2017.
[Sug+19]
Arun Sai Suggala et al. “Revisiting Adversarial Risk”. In: Conference on Artiﬁcial Intelligence and
Statistics (AISTATS). 2019.
[Sze+14]
Christian Szegedy et al. “Intriguing properties of neural networks”. In: International Conference
on Learning Representations (ICLR). 2014.
[TG16]
Thomas Tanay and Lewis Grifﬁn. “A Boundary Tilting Perspective on the Phenomenon of
Adversarial Examples”. In: ArXiv preprint arXiv:1608.07690. 2016.
13


**Table 16 from page 13**

| 0         | 1                                                                                                   |
|:----------|:----------------------------------------------------------------------------------------------------|
| [Liu+17]  | Yanpei Liu et al. “Delving into Transferable Adversarial Examples and Black-box Attacks”. In:       |
|           | International Conference on Learning Representations (ICLR). 2017.                                  |
| [LM00]    | Beatrice Laurent and Pascal Massart. “Adaptive estimation of a quadratic functional by model        |
|           | selection”. In: Annals of Statistics. 2000.                                                         |
| [Mad+18]  | Aleksander Madry et al. “Towards deep learning models resistant to adversarial attacks”. In:        |
|           | International Conference on Learning Representations (ICLR). 2018.                                  |
| [MDM18]   | Saeed Mahloujifar, Dimitrios I Diochnos, and Mohammad Mahmoody. “The curse of concen-               |
|           | tration in robust learning: Evasion and poisoning attacks from concentration of measure”. In:       |
|           | AAAI Conference on Artiﬁcial Intelligence (AAAI). 2018.                                             |
| [Moo+17]  | Seyed-Mohsen Moosavi-Dezfooli et al. “Universal adversarial perturbations”. In: conference on       |
|           | computer vision and pattern recognition (CVPR). 2017.                                               |
| [MV15]    | Aravindh Mahendran and Andrea Vedaldi. “Understanding deep image representations by                 |
|           | inverting them”. In: computer vision and pattern recognition (CVPR). 2015.                          |
| [Nak19a]  | Preetum Nakkiran. “A Discussion of ’Adversarial Examples Are Not Bugs, They Are Features’:          |
|           | Adversarial Examples are Just Bugs, Too”. In: Distill (2019). https://distill.pub/2019/advex-       |
|           | bugs-discussion/response-5. DOI: 10.23915/distill.00019.5.                                          |
| [Nak19b]  | Preetum Nakkiran. “Adversarial robustness may be at odds with simplicity”. In: arXiv preprint       |
|           | arXiv:1901.00532. 2019.                                                                             |
| [OMS17]   | Chris Olah, Alexander Mordvintsev, and Ludwig Schubert. “Feature Visualization”. In: Distill.       |
|           | 2017.                                                                                               |
| [Pap+17]  | Nicolas Papernot et al. “Practical black-box attacks against machine learning”. In: Asia Confer-    |
|           | ence on Computer and Communications Security. 2017.                                                 |
| [PMG16]   | Nicolas Papernot, Patrick McDaniel, and Ian Goodfellow. “Transferability in Machine Learn-          |
|           | ing:                                                                                                |
|           | from Phenomena to Black-box Attacks using Adversarial Samples”.                                     |
|           | In: ArXiv preprint                                                                                  |
|           | arXiv:1605.07277. 2016.                                                                             |
| [Rec+19]  | Benjamin Recht et al. “Do CIFAR-10 Classiﬁers Generalize to CIFAR-10?” In: International Con-       |
|           | ference on Machine Learning (ICML). 2019.                                                           |
| [RSL18]   | Aditi Raghunathan, Jacob Steinhardt, and Percy Liang. “Certiﬁed defenses against adversarial        |
|           | examples”. In: International Conference on Learning Representations (ICLR). 2018.                   |
| [Rus+15]  | Olga Russakovsky et al. “ImageNet Large Scale Visual Recognition Challenge”. In: International      |
|           | Journal of Computer Vision (IJCV). 2015.                                                            |
| [Sch+18]  | Ludwig Schmidt et al. “Adversarially Robust Generalization Requires More Data”.                     |
|           | In: Ad-                                                                                             |
|           | vances in Neural Information Processing Systems (NeurIPS). 2018.                                    |
| [Sha+19a] | Ali Shafahi et al. “Are adversarial examples inevitable?” In: International Conference on Learning  |
|           | Representations (ICLR). 2019.                                                                       |
| [Sha+19b] | Adi Shamir et al. “A Simple Explanation for the Existence of Adversarial Examples with Small        |
|           | Hamming Distance”. In: arXiv preprint arXiv:1901.10861. 2019.                                       |
| [SHS19]   | David Stutz, Matthias Hein, and Bernt Schiele. “Disentangling Adversarial Robustness and            |
|           | Generalization”. In: Computer Vision and Pattern Recognition (CVPR). 2019.                          |
| [Smi+17]  | D. Smilkov et al. “SmoothGrad: removing noise by adding noise”. In: ICML workshop on visu-          |
|           | alization for deep learning. 2017.                                                                  |
| [Sug+19]  | Arun Sai Suggala et al. “Revisiting Adversarial Risk”. In: Conference on Artiﬁcial Intelligence and |
|           | Statistics (AISTATS). 2019.                                                                         |
| [Sze+14]  | Christian Szegedy et al. “Intriguing properties of neural networks”. In: International Conference   |
|           | on Learning Representations (ICLR). 2014.                                                           |
| [TG16]    | Thomas Tanay and Lewis Grifﬁn. “A Boundary Tilting Perspective on the Phenomenon of                 |
|           | Adversarial Examples”. In: ArXiv preprint arXiv:1608.07690. 2016.                                   |



## Page 14

[Tra+17]
Florian Tramer et al. “The Space of Transferable Adversarial Examples”. In: ArXiv preprint
arXiv:1704.03453. 2017.
[Tsi+19]
Dimitris Tsipras et al. “Robustness May Be at Odds with Accuracy”. In: International Conference
on Learning Representations (ICLR). 2019.
[Ues+18]
Jonathan Uesato et al. “Adversarial Risk and the Dangers of Evaluating Against Weak At-
tacks”. In: International Conference on Machine Learning (ICML). 2018.
[Wan+18]
Tongzhou Wang et al. “Dataset Distillation”. In: ArXiv preprint arXiv:1811.10959. 2018.
[WK18]
Eric Wong and J Zico Kolter. “Provable defenses against adversarial examples via the convex
outer adversarial polytope”. In: International Conference on Machine Learning (ICML). 2018.
[Xia+19]
Kai Y. Xiao et al. “Training for Faster Adversarial Robustness Veriﬁcation via Inducing ReLU
Stability”. In: International Conference on Learning Representations (ICLR). 2019.
[Zou+18]
Haosheng Zou et al. “Geometric Universality of Adversarial Examples in Deep Learning”. In:
Geometry in Machine Learning ICML Workshop (GIML). 2018.
14


**Table 17 from page 14**

| 0        | 1                                                                                               |
|:---------|:------------------------------------------------------------------------------------------------|
| [Tra+17] | Florian Tramer et al. “The Space of Transferable Adversarial Examples”.                         |
|          | In: ArXiv preprint                                                                              |
|          | arXiv:1704.03453. 2017.                                                                         |
| [Tsi+19] | Dimitris Tsipras et al. “Robustness May Be at Odds with Accuracy”. In: International Conference |
|          | on Learning Representations (ICLR). 2019.                                                       |
| [Ues+18] | Jonathan Uesato et al. “Adversarial Risk and the Dangers of Evaluating Against Weak At-         |
|          | tacks”. In: International Conference on Machine Learning (ICML). 2018.                          |
| [Wan+18] | Tongzhou Wang et al. “Dataset Distillation”. In: ArXiv preprint arXiv:1811.10959. 2018.         |
| [WK18]   | Eric Wong and J Zico Kolter. “Provable defenses against adversarial examples via the convex     |
|          | outer adversarial polytope”. In: International Conference on Machine Learning (ICML). 2018.     |
| [Xia+19] | Kai Y. Xiao et al. “Training for Faster Adversarial Robustness Veriﬁcation via Inducing ReLU    |
|          | Stability”. In: International Conference on Learning Representations (ICLR). 2019.              |
| [Zou+18] | Haosheng Zou et al. “Geometric Universality of Adversarial Examples in Deep Learning”. In:      |
|          | Geometry in Machine Learning ICML Workshop (GIML). 2018.                                        |



## Page 15

A
Connections to and Disambiguation from Other Models
Here, we describe other models for adversarial examples and how they relate to the model presented in
this paper.
Concentration of measure in high-dimensions.
An orthogonal line of work [Gil+18; FFF18; MDM18;
Sha+19a], argues that the high dimensionality of the input space can present fundamental barriers on clas-
siﬁer robustness. At a high level, one can show that, for certain data distributions, any decision boundary
will be close to a large fraction of inputs and hence no classiﬁer can be robust against small perturbations.
While there might exist such fundamental barriers to robustly classifying standard datasets, this model
cannot fully explain the situation observed in practice, where one can train (reasonably) robust classiﬁers
on standard datasets [Mad+18; RSL18; WK18; Xia+19; CRK19].
Insufﬁcient data.
Schmidt et al. [Sch+18] propose a theoretical model under which a single sample is suf-
ﬁcient to learn a good, yet non-robust classiﬁer, whereas learning a good robust classiﬁer requires O(
√
d)
samples. Under this model, adversarial examples arise due to insufﬁcient information about the true data
distribution. However, unless the adversary is strong enough (in which case no robust classiﬁer exists), ad-
versarial inputs cannot be utilized as inputs of the opposite class (as done in our experiments in Section 3.2).
We note that our model does not explicitly contradict the main thesis of Schmidt et al. [Sch+18]. In fact, this
thesis can be viewed as a natural consequence of our conceptual framework. In particular, since training
models robustly reduces the effective amount of information in the training data (as non-robust features
are discarded), more samples should be required to generalize robustly.
Boundary Tilting.
Tanay and Grifﬁn [TG16] introduce the “boundary tilting” model for adversarial ex-
amples, and suggest that adversarial examples are a product of over-ﬁtting. In particular, the model conjec-
tures that “adversarial examples are possible because the class boundary extends beyond the submanifold
of sample data and can be—under certain circumstances—lying close to it.” Consequently, the authors sug-
gest that mitigating adversarial examples may be a matter of regularization and preventing ﬁnite-sample
overﬁtting. In contrast, our empirical results in Section 3.2 suggest that adversarial inputs consist of features
inherent to the data distribution, since they can encode generalizing information about the target class.
Inspired by this hypothesis and concurrently to our work, Kim, Seo, and Jeon [KSJ19] present a simple
classiﬁcation task comprised of two Gaussian distributions in two dimensions. They experimentally show
that the decision boundary tends to better align with the vector between the two means for robust models.
This is a special case of our theoretical results in Section 4. (Note that this exact statement is not true beyond
two dimensions, as discussed in Section 4.)
Test Error in Noise.
Fawzi, Moosavi-Dezfooli, and Frossard [FMF16] and Ford et al. [For+19] argue that
the adversarial robustness of a classiﬁer can be directly connected to its robustness under (appropriately
scaled) random noise. While this constitutes a natural explanation of adversarial vulnerability given the
classiﬁer robustness to noise, these works do not attempt to justify the source of the latter.
At the same time, recent work [Lec+19; CRK19; For+19] utilizes random noise during training or testing
to construct adversarially robust classiﬁers. In the context of our framework, we can expect the added noise
to disproportionately affect non-robust features and thus hinder the model’s reliance on them.
Local Linearity.
Goodfellow, Shlens, and Szegedy [GSS15] suggest that the local linearity of DNNs is
largely responsible for the existence of small adversarial perturbations. While this conjecture is supported
by the effectiveness of adversarial attacks exploiting local linearity (e.g., FGSM [GSS15]), it is not sufﬁcient
to fully characterize the phenomena observed in practice. In particular, there exist adversarial examples
that violate the local linearity of the classiﬁer [Mad+18], while classiﬁers that are less linear do not exhibit
greater robustness [ACW18].
15


**Table 18 from page 15**

| 0                                                                                                                  |
|:-------------------------------------------------------------------------------------------------------------------|
| A                                                                                                                  |
| Connections to and Disambiguation from Other Models                                                                |
| Here, we describe other models for adversarial examples and how they relate to the model presented in              |
| this paper.                                                                                                        |
| Concentration of measure in high-dimensions.                                                                       |
| An orthogonal                                                                                                      |
| line of work [Gil+18; FFF18; MDM18;                                                                                |
| Sha+19a], argues that the high dimensionality of the input space can present fundamental barriers on clas-         |
| siﬁer robustness. At a high level, one can show that, for certain data distributions, any decision boundary        |
| will be close to a large fraction of inputs and hence no classiﬁer can be robust against small perturbations.      |
| While there might exist such fundamental barriers to robustly classifying standard datasets,                       |
| this model                                                                                                         |
| cannot fully explain the situation observed in practice, where one can train (reasonably) robust classiﬁers        |
| on standard datasets [Mad+18; RSL18; WK18; Xia+19; CRK19].                                                         |
| Insufﬁcient data.                                                                                                  |
| Schmidt et al. [Sch+18] propose a theoretical model under which a single sample is suf-                            |
| ﬁcient to learn a good, yet non-robust classiﬁer, whereas learning a good robust classiﬁer requires O(√d)          |
| samples. Under this model, adversarial examples arise due to insufﬁcient information about the true data           |
| distribution. However, unless the adversary is strong enough (in which case no robust classiﬁer exists), ad-       |
| versarial inputs cannot be utilized as inputs of the opposite class (as done in our experiments in Section 3.2).   |
| We note that our model does not explicitly contradict the main thesis of Schmidt et al. [Sch+18]. In fact, this    |
| thesis can be viewed as a natural consequence of our conceptual framework.                                         |
| In particular, since training                                                                                      |
| models robustly reduces the effective amount of information in the training data (as non-robust features           |
| are discarded), more samples should be required to generalize robustly.                                            |
| Boundary Tilting.                                                                                                  |
| Tanay and Grifﬁn [TG16] introduce the “boundary tilting” model for adversarial ex-                                 |
| amples, and suggest that adversarial examples are a product of over-ﬁtting. In particular, the model conjec-       |
| tures that “adversarial examples are possible because the class boundary extends beyond the submanifold            |
| of sample data and can be—under certain circumstances—lying close to it.” Consequently, the authors sug-           |
| gest that mitigating adversarial examples may be a matter of regularization and preventing ﬁnite-sample            |
| overﬁtting. In contrast, our empirical results in Section 3.2 suggest that adversarial inputs consist of features  |
| inherent to the data distribution, since they can encode generalizing information about the target class.          |
| Inspired by this hypothesis and concurrently to our work, Kim, Seo, and Jeon [KSJ19] present a simple              |
| classiﬁcation task comprised of two Gaussian distributions in two dimensions. They experimentally show             |
| that the decision boundary tends to better align with the vector between the two means for robust models.          |
| This is a special case of our theoretical results in Section 4. (Note that this exact statement is not true beyond |
| two dimensions, as discussed in Section 4.)                                                                        |
| Test Error in Noise.                                                                                               |
| Fawzi, Moosavi-Dezfooli, and Frossard [FMF16] and Ford et al. [For+19] argue that                                  |
| the adversarial robustness of a classiﬁer can be directly connected to its robustness under (appropriately         |
| scaled) random noise. While this constitutes a natural explanation of adversarial vulnerability given the          |
| classiﬁer robustness to noise, these works do not attempt to justify the source of the latter.                     |
| At the same time, recent work [Lec+19; CRK19; For+19] utilizes random noise during training or testing             |
| to construct adversarially robust classiﬁers. In the context of our framework, we can expect the added noise       |
| to disproportionately affect non-robust features and thus hinder the model’s reliance on them.                     |
| Local Linearity.                                                                                                   |
| Goodfellow, Shlens, and Szegedy [GSS15] suggest                                                                    |
| that                                                                                                               |
| the local                                                                                                          |
| linearity of DNNs is                                                                                               |
| largely responsible for the existence of small adversarial perturbations. While this conjecture is supported       |
| by the effectiveness of adversarial attacks exploiting local linearity (e.g., FGSM [GSS15]), it is not sufﬁcient   |
| to fully characterize the phenomena observed in practice.                                                          |
| In particular,                                                                                                     |
| there exist adversarial examples                                                                                   |
| that violate the local linearity of the classiﬁer [Mad+18], while classiﬁers that are less linear do not exhibit   |
| greater robustness [ACW18].                                                                                        |



## Page 16

Piecewise-linear decision boundaries.
Shamir et al. [Sha+19b] prove that the geometric structure of the
classiﬁer’s decision boundaries can lead to sparse adversarial perturbations. However, this result does not
take into account the distance to the decision boundary along these direction or feasibility constraints on
the input domain. As a result, it cannot meaningfully distinguish between classiﬁers that are brittle to small
adversarial perturbations and classiﬁers that are moderately robust.
Theoretical constructions which incidentally exploit non-robust features.
Bubeck, Price, and Razen-
shteyn [BPR18] and Nakkiran [Nak19b] propose theoretical models where the barrier to learning robust
classiﬁers is, respectively, due to computational constraints or model complexity. In order to construct dis-
tributions that admit accurate yet non-robust classiﬁers they (implicitly) utilize the concept of non-robust
features. Namely, they add a low-magnitude signal to each input that encodes the true label. This allows a
classiﬁer to achieve perfect standard accuracy, but cannot be utilized in an adversarial setting as this signal
is susceptible to small adversarial perturbations.
B
Additional Related Work
We describe previously proposed models for the existence of adversarial examples in the previous section.
Here we discuss other work that is methodologically or conceptually similar to ours.
Distillation.
The experiments performed in Section 3.1 can be seen as a form of distillation. There is a line
of work, known as model distillation [HVD14; Fur+18; BCN06], where the goal is to train a new model to
mimic another already trained model. This is typically achieved by adding some regularization terms to
the loss in order to encourage the two models to be similar, often replacing training labels with some other
target based on the already trained model. While it might be possible to successfully distill a robust model
using these methods, our goal was to achieve it by only modifying the training set (leaving the training pro-
cess unchanged), hence demonstrating that adversarial vulnerability is mainly a property of the dataset.
Closer to our work is dataset distillation [Wan+18] which considers the problem of reconstructing a clas-
siﬁer from an alternate dataset much smaller than the original training set. This method aims to produce
inputs that directly encode the weights of the already trained model by ensuring that the classiﬁer’s gra-
dient with respect to these inputs approximates the desired weights. (As a result, the inputs constructed
do not resemble natural inputs.) This approach is orthogonal to our goal since we are not interested in
encoding the particular weights into the dataset but rather in imposing a structure to its features.
Adversarial Transferabiliy.
In our work, we posit that a potentially natural consequence of the existence
of non-robust features is adversarial transferability [Pap+17; Liu+17; PMG16]. A recent line of work has
considered this phenomenon from a theoretical perspective, conﬁned to simple models, or unbounded per-
turbations [CRP19; Zou+18]. Tramer et al. [Tra+17] study transferability empirically, by ﬁnding adversarial
subspaces, (orthogonal vectors whose linear combinations are adversarial perturbations). The authors ﬁnd
that there is a signiﬁcant overlap in the adversarial subspaces between different models, and identify this
as a source of transferability. In our work, we provide a potential reason for this overlap—these directions
correspond to non-robust features utilized by models in a similar manner.
Universal Adversarial Perturbations
Moosavi-Dezfooli et al. [Moo+17] construct perturbations that can
cause misclassiﬁcation when applied to multiple different inputs. More recently, Jetley, Lord, and Torr
[JLT18] discover input patterns that are meaningless to humans and can induce misclassiﬁcation, while
at the same time being essential for standard classiﬁcation. These ﬁndings can be naturally cast into our
framework by considering these patterns as non-robust features, providing further evidence about their
pervasiveness.
Manipulating dataset features
Ding et al. [Din+19] perform synthetic transformations on the dataset (e.g.,
image saturation) and study the performance of models on the transformed dataset under standard and ro-
bust training. While this can be seen as a method of restricting the features available to the model during
16


**Table 19 from page 16**

| 0                                                                                                               |
|:----------------------------------------------------------------------------------------------------------------|
| Piecewise-linear decision boundaries.                                                                           |
| Shamir et al. [Sha+19b] prove that the geometric structure of the                                               |
| classiﬁer’s decision boundaries can lead to sparse adversarial perturbations. However, this result does not     |
| take into account the distance to the decision boundary along these direction or feasibility constraints on     |
| the input domain. As a result, it cannot meaningfully distinguish between classiﬁers that are brittle to small  |
| adversarial perturbations and classiﬁers that are moderately robust.                                            |
| Theoretical constructions which incidentally exploit non-robust features.                                       |
| Bubeck, Price, and Razen-                                                                                       |
| shteyn [BPR18] and Nakkiran [Nak19b] propose theoretical models where the barrier to learning robust            |
| classiﬁers is, respectively, due to computational constraints or model complexity. In order to construct dis-   |
| tributions that admit accurate yet non-robust classiﬁers they (implicitly) utilize the concept of non-robust    |
| features. Namely, they add a low-magnitude signal to each input that encodes the true label. This allows a      |
| classiﬁer to achieve perfect standard accuracy, but cannot be utilized in an adversarial setting as this signal |
| is susceptible to small adversarial perturbations.                                                              |



## Page 17

training, it is unclear how well these models would perform on the standard test set. Geirhos et al. [Gei+19]
aim to quantify the relative dependence of standard models on shape and texture information of the in-
put. They introduce a version of ImageNet where texture information has been removed and observe an
improvement to certain corruptions.
C
Experimental Setup
C.1
Datasets
For our experimental analysis, we use the CIFAR-10 [Kri09] and (restricted) ImageNet [Rus+15] datasets.
Attaining robust models for the complete ImageNet dataset is known to be a challenging problem, both
due to the hardness of the learning problem itself, as well as the computational complexity. We thus restrict
our focus to a subset of the dataset which we denote as restricted ImageNet. To this end, we group together
semantically similar classes from ImageNet into 9 super-classes shown in Table 2. We train and evaluate
only on examples corresponding to these classes.
Class
Corresponding ImageNet Classes
“Dog”
151 to 268
“Cat”
281 to 285
“Frog”
30 to 32
“Turtle”
33 to 37
“Bird”
80 to 100
“Primate”
365 to 382
“Fish”
389 to 397
“Crab”
118 to 121
“Insect”
300 to 319
Table 2: Classes used in the Restricted ImageNet model. The class ranges are inclusive.
C.2
Models
We use the ResNet-50 architecture for our baseline standard and adversarially trained classiﬁers on CIFAR-
10 and restricted ImageNet. For each model, we grid search over three learning rates (0.1, 0.01, 0.05), two
batch sizes (128, 256) including/not including a learning rate drop (a single order of magnitude) and data
augmentation. We use the standard training parameters for the remaining parameters. The hyperparame-
ters used for each model are given in Table 3.
Dataset
LR
Batch Size
LR Drop
Data Aug.
Momentum
Weight Decay
bDR (CIFAR)
0.1
128
Yes
Yes
0.9
5 · 10−4
bDR (Restricted ImageNet)
0.01
128
No
Yes
0.9
5 · 10−4
bDNR (CIFAR)
0.1
128
Yes
Yes
0.9
5 · 10−4
bDrand (CIFAR)
0.01
128
Yes
Yes
0.9
5 · 10−4
bDrand (Restricted ImageNet)
0.01
256
No
No
0.9
5 · 10−4
bDdet (CIFAR)
0.1
128
Yes
No
0.9
5 · 10−4
bDdet (Restricted ImageNet)
0.05
256
No
No
0.9
5 · 10−4
Table 3: Hyperparameters for the models trained in the main paper. All hyperparameters were obtained
through a grid search.
17


**Table 20 from page 17**

| 0                                                                                                           | 1                              |
|:------------------------------------------------------------------------------------------------------------|:-------------------------------|
| our focus to a subset of the dataset which we denote as restricted ImageNet. To this end, we group together |                                |
| semantically similar classes from ImageNet into 9 super-classes shown in Table 2. We train and evaluate     |                                |
| Class                                                                                                       | Corresponding ImageNet Classes |
| “Dog”                                                                                                       | 151 to 268                     |
| “Cat”                                                                                                       | 281 to 285                     |
| “Frog”                                                                                                      | 30 to 32                       |
| “Turtle”                                                                                                    | 33 to 37                       |
| “Bird”                                                                                                      | 80 to 100                      |
| “Primate”                                                                                                   | 365 to 382                     |
| “Fish”                                                                                                      | 389 to 397                     |
| “Crab”                                                                                                      | 118 to 121                     |
| “Insect”                                                                                                    | 300 to 319                     |

**Table 21 from page 17**

| 0                                                                                                          | 1                          | 2    | 3          | 4       | 5         | 6        | 7   | 8            | 9   | 10   |
|:-----------------------------------------------------------------------------------------------------------|:---------------------------|:-----|:-----------|:--------|:----------|:---------|:----|:-------------|:----|:-----|
| batch sizes (128, 256) including/not including a learning rate drop (a single order of magnitude) and data |                            |      |            |         |           |          |     |              |     |      |
| augmentation. We use the standard training parameters for the remaining parameters. The hyperparame-       |                            |      |            |         |           |          |     |              |     |      |
| ters used for each model are given in Table 3.                                                             |                            |      |            |         |           |          |     |              |     |      |
| Dataset                                                                                                    |                            | LR   | Batch Size | LR Drop | Data Aug. | Momentum |     | Weight Decay |     |      |
|                                                                                                            |                            |      |            |         |           |          |     |              |     | 4    |
|                                                                                                            | R (CIFAR)                  | 0.1  | 128        | Yes     | Yes       | 0.9      | 5   |              | 10− |      |
|                                                                                                            |                            |      |            |         |           |          |     | ·            |     | 4    |
| (cid:98)D                                                                                                  | R (Restricted ImageNet)    | 0.01 | 128        | No      | Yes       | 0.9      | 5   |              | 10− |      |
|                                                                                                            |                            |      |            |         |           |          |     | ·            |     | 4    |
| (cid:98)D                                                                                                  | NR (CIFAR)                 | 0.1  | 128        | Yes     | Yes       | 0.9      | 5   |              | 10− |      |
|                                                                                                            |                            |      |            |         |           |          |     | ·            |     | 4    |
| (cid:98)D                                                                                                  | rand (CIFAR)               | 0.01 | 128        | Yes     | Yes       | 0.9      | 5   |              | 10− |      |
|                                                                                                            |                            |      |            |         |           |          |     | ·            |     | 4    |
| (cid:98)D                                                                                                  | rand (Restricted ImageNet) | 0.01 | 256        | No      | No        | 0.9      | 5   |              | 10− |      |
|                                                                                                            |                            |      |            |         |           |          |     | ·            |     | 4    |
| (cid:98)D                                                                                                  | det (CIFAR)                | 0.1  | 128        | Yes     | No        | 0.9      | 5   |              | 10− |      |
|                                                                                                            |                            |      |            |         |           |          |     | ·            |     | 4    |
| (cid:98)D                                                                                                  | det (Restricted ImageNet)  | 0.05 | 256        | No      | No        | 0.9      | 5   |              | 10− |      |
| (cid:98)D                                                                                                  |                            |      |            |         |           |          |     | ·            |     |      |



## Page 18

C.3
Adversarial training
To obtain robust classiﬁers, we employ the adversarial training methodology proposed in [Mad+18]. Specif-
ically, we train against a projected gradient descent (PGD) adversary constrained in ℓ2-norm starting from
the original image. Following Madry et al. [Mad+18] we normalize the gradient at each step of PGD to
ensure that we move a ﬁxed distance in ℓ2-norm per step. Unless otherwise speciﬁed, we use the values of
ϵ provided in Table 4 to train/evaluate our models. We used 7 steps of PGD with a step size of ε/5.
Adversary
CIFAR-10
Restricted Imagenet
ℓ2
0.5
3
Table 4: Value of ϵ used for ℓ2 adversarial training/evaluation of each dataset.
C.4
Constructing a Robust Dataset
In Section 3.1, we describe a procedure to construct a dataset that contains features relevant only to a given
(standard/robust) model. To do so, we optimize the training objective in (6). Unless otherwise speciﬁed,
we initialize xr as a different randomly chosen sample from the training set. (For the sake of completeness,
we also try initializing with a Gaussian noise instead as shown in Table 7.) We then perform normalized
gradient descent (ℓ2-norm of gradient is ﬁxed to be constant at each step). At each step we clip the input
xr to in the [0, 1] range so as to ensure that it is a valid image. Details on the optimization procedure are
shown in Table 5. We provide the pseudocode for the construction in Figure 5.
GETROBUSTDATASET(D)
1. CR ←ADVERSARIALTRAINING(D)
gR ←mapping learned by CR from the input to the representation layer
2. DR ←{}
3. For (x, y) ∈D
x′ ∼D
xR ←arg minz∈[0,1]d ∥gR(z) −gR(x)∥2
# Solved using ℓ2-PGD starting from x′
DR ←DR
S {(xR, y)}
4. Return DR
Figure 5: Algorithm to construct a “robust” dataset, by restricting to features used by a robust model.
CIFAR-10
Restricted Imagenet
step size
0.1
1
iterations
1000
2000
Table 5: Parameters used for optimization procedure to construct dataset in Section 3.1.
18


**Table 22 from page 18**

| 0                                                                                                               | 1                                                                                                                 | 2                                                         |
|:----------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------|
| C.3                                                                                                             | Adversarial training                                                                                              |                                                           |
|                                                                                                                 | To obtain robust classiﬁers, we employ the adversarial training methodology proposed in [Mad+18]. Specif-         |                                                           |
|                                                                                                                 | ically, we train against a projected gradient descent (PGD) adversary constrained in (cid:96)2-norm starting from |                                                           |
| the original                                                                                                    | image.                                                                                                            | [Mad+18] we normalize the gradient at each step of PGD to |
|                                                                                                                 | Following Madry et al.                                                                                            |                                                           |
| ensure that we move a ﬁxed distance in (cid:96)2-norm per step. Unless otherwise speciﬁed, we use the values of |                                                                                                                   |                                                           |
|                                                                                                                 | (cid:101) provided in Table 4 to train/evaluate our models. We used 7 steps of PGD with a step size of ε/5.       |                                                           |



## Page 19

C.5
Non-robust features sufﬁce for standard classiﬁcation
To construct the dataset as described in Section 3.2, we use the standard projected gradient descent (PGD)
procedure described in [Mad+18] to construct an adversarial example for a given input from the dataset (7).
Perturbations are constrained in ℓ2-norm while each PGD step is normalized to a ﬁxed step size. The details
for our PGD setup are described in Table 6. We provide pseudocode in Figure 6.
GETNONROBUSTDATASET(D, ε)
1. DNR ←{}
2. C ←STANDARDTRAINING(D)
3. For (x, y) ∈D
t uar
∼[C]
# or t ←(y + 1) mod C
xNR ←min||x′−x||≤ε LC(x′, t)
# Solved using ℓ2 PGD
DNR ←DNR
S {(xNR, t)}
4. Return DNR
Figure 6: Algorithm to construct a dataset where input-label association is based entirely on non-robust
features.
Attack Parameters
CIFAR-10
Restricted Imagenet
ε
0.5
3
step size
0.1
0.1
iterations
100
100
Table 6: Projected gradient descent parameters used to construct constrained adversarial examples in Sec-
tion 3.2.
19


**Table 23 from page 19**

| 0                                                                                                                  | 1                                                     |
|:-------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------|
| C.5                                                                                                                | Non-robust features sufﬁce for standard classiﬁcation |
| To construct the dataset as described in Section 3.2, we use the standard projected gradient descent (PGD)         |                                                       |
| procedure described in [Mad+18] to construct an adversarial example for a given input from the dataset (7).        |                                                       |
| Perturbations are constrained in (cid:96)2-norm while each PGD step is normalized to a ﬁxed step size. The details |                                                       |
| for our PGD setup are described in Table 6. We provide pseudocode in Figure 6.                                     |                                                       |
|                                                                                                                    | GETNONROBUSTDATASET(D, ε)                             |
|                                                                                                                    | 1. DNR ← {}                                           |
|                                                                                                                    | 2. C                                                  |
|                                                                                                                    | STANDARDTRAINING(D)                                   |
|                                                                                                                    | ←                                                     |
|                                                                                                                    | D                                                     |
|                                                                                                                    | 3. For (x, y)                                         |
|                                                                                                                    | ∈                                                     |
|                                                                                                                    | uar                                                   |
|                                                                                                                    | t                                                     |
|                                                                                                                    | [C]                                                   |
|                                                                                                                    | # or t                                                |
|                                                                                                                    | (y + 1) mod C                                         |
|                                                                                                                    | ∼                                                     |
|                                                                                                                    | ←                                                     |
|                                                                                                                    | min                                                   |
|                                                                                                                    | # Solved using (cid:96)2 PGD                          |
|                                                                                                                    | x                                                     |
|                                                                                                                    | ε LC(x(cid:48), t)                                    |
|                                                                                                                    | xNR ←                                                 |
|                                                                                                                    | ||                                                    |
|                                                                                                                    | x(cid:48)−                                            |
|                                                                                                                    | ||≤                                                   |
|                                                                                                                    | DNR                                                   |
|                                                                                                                    | (xNR, t)                                              |
|                                                                                                                    | DNR ←                                                 |
|                                                                                                                    | {                                                     |
|                                                                                                                    | }                                                     |
|                                                                                                                    | (cid:83)                                              |
|                                                                                                                    | 4. Return DNR                                         |
| Figure 6: Algorithm to construct a dataset where input-label association is based entirely on non-robust           |                                                       |
| features.                                                                                                          |                                                       |
|                                                                                                                    | Attack Parameters                                     |
|                                                                                                                    | CIFAR-10                                              |
|                                                                                                                    | Restricted Imagenet                                   |
|                                                                                                                    | ε                                                     |
|                                                                                                                    | 0.5                                                   |
|                                                                                                                    | 3                                                     |
|                                                                                                                    | step size                                             |
|                                                                                                                    | 0.1                                                   |
|                                                                                                                    | 0.1                                                   |
|                                                                                                                    | iterations                                            |
|                                                                                                                    | 100                                                   |
|                                                                                                                    | 100                                                   |
| Table 6: Projected gradient descent parameters used to construct constrained adversarial examples in Sec-          |                                                       |
| tion 3.2.                                                                                                          |                                                       |
|                                                                                                                    | 19                                                    |



## Page 20

D
Omitted Experiments and Figures
D.1
Detailed evaluation of models trained on “robust” dataset
In Section 3.1, we generate a “robust” training set by restricting the dataset to only contain features relevant
to a robust model (robust dataset) or a standard model (non-robust dataset). This is performed by choos-
ing either a random input from the training set or random noise16 and then performing the optimization
procedure described in (6). The performance of these classiﬁers along with various baselines is shown in
Table 7. We observe that while the robust dataset constructed from noise resembles the original, the corre-
sponding non-robust does not (Figure 7). This also leads to suboptimal performance of classiﬁers trained
on this dataset (only 46% standard accuracy) potentially due to a distributional shift.
Robust Accuracy
Model
Accuracy
ε = 0.25
ε = 0.5
Standard Training
95.25 %
4.49%
0.0%
Robust Training
90.83%
82.48%
70.90%
Trained on non-robust dataset (constructed from images)
87.68%
0.82%
0.0%
Trained on non-robust dataset (constructed from noise)
45.60%
1.50%
0.0%
Trained on robust dataset (constructed from images)
85.40%
48.20 %
21.85%
Trained on robust dataset (constructed from noise)
84.10%
48.27 %
29.40%
Table 7: Standard and robust classiﬁcation performance on the CIFAR-10 test set of: an (i) ERM classiﬁer;
(ii) ERM classiﬁer trained on a dataset obtained by distilling features relevant to ERM classiﬁer in (i); (iii)
adversarially trained classiﬁer (ε = 0.5); (iv) ERM classiﬁer trained on dataset obtained by distilling features
used by robust classiﬁer in (iii). Simply restricting the set of available features during ERM to features used
by a standard model yields non-trivial robust accuracy.
“deer’’
“truck’’
“cat’’
“ship’’
“bird’’
D
!DNR
!DR
Figure 7: Robust and non-robust datasets for CIFAR-10 when the process starts from noise (as opposed to
random images as in Figure 2a).
16We use 10k steps to construct the dataset from noise, instead to using 1k steps done when the input is a different training set image
(cf. Table 5).
20


**Table 24 from page 20**

| 0                                                                                                                | 1                                                         |
|:-----------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------|
| D                                                                                                                | Omitted Experiments and Figures                           |
| D.1                                                                                                              | Detailed evaluation of models trained on “robust” dataset |
| In Section 3.1, we generate a “robust” training set by restricting the dataset to only contain features relevant |                                                           |
| to a robust model (robust dataset) or a standard model (non-robust dataset). This is performed by choos-         |                                                           |
| ing either a random input from the training set or random noise16 and then performing the optimization           |                                                           |
| procedure described in (6). The performance of these classiﬁers along with various baselines is shown in         |                                                           |
| Table 7. We observe that while the robust dataset constructed from noise resembles the original, the corre-      |                                                           |
| sponding non-robust does not (Figure 7). This also leads to suboptimal performance of classiﬁers trained         |                                                           |
| on this dataset (only 46% standard accuracy) potentially due to a distributional shift.                          |                                                           |

**Table 25 from page 20**

| 0                                                                                                        | 1        | 2               | 3       |
|:---------------------------------------------------------------------------------------------------------|:---------|:----------------|:--------|
| sponding non-robust does not (Figure 7). This also leads to suboptimal performance of classiﬁers trained |          |                 |         |
| on this dataset (only 46% standard accuracy) potentially due to a distributional shift.                  |          |                 |         |
|                                                                                                          |          | Robust Accuracy |         |
| Model                                                                                                    | Accuracy | ε = 0.25        | ε = 0.5 |
| Standard Training                                                                                        | 95.25 %  | 4.49%           | 0.0%    |
| Robust Training                                                                                          | 90.83%   | 82.48%          | 70.90%  |
| Trained on non-robust dataset (constructed from images)                                                  | 87.68%   | 0.82%           | 0.0%    |
| Trained on non-robust dataset (constructed from noise)                                                   | 45.60%   | 1.50%           | 0.0%    |
| Trained on robust dataset (constructed from images)                                                      | 85.40%   | 48.20 %         | 21.85%  |
| Trained on robust dataset (constructed from noise)                                                       | 84.10%   | 48.27 %         | 29.40%  |



## Page 21

D.2
Adversarial evaluation
To verify the robustness of our classiﬁers trained on the ‘robust” dataset, we evaluate them with strong
attacks [Car+19]. In particular, we try up to 2500 steps of projected gradient descent (PGD), increasing
steps until the accuracy plateaus, and also try the CW-ℓ2 loss function [CW17b] with 1000 steps. For each
attack we search over step size. We ﬁnd that over all attacks and step sizes, the accuracy of the model does
not drop by more than 2%, and plateaus at 48.27% for both PGD and CW-ℓ2 (the value given in Figure 2).
We show a plot of accuracy in terms of the number of PGD steps used in Figure 8.
0
100
101
102
103
Number of PGD steps
0
20
40
60
80
100
Adversarial accuracy to = 0.25 (%)
Figure 8: Robust accuracy as a function of the number of PGD steps used to generate the attack. The
accuracy plateaus at 48.27%.
D.3
Performance of “robust” training and test set
In Section 3.1, we observe that an ERM classiﬁer trained on a “robust” training dataset bDR (obtained by
restricting features to those relevant to a robust model) attains non-trivial robustness (cf. Figure 1 and
Table 7). In Table 8, we evaluate the adversarial accuracy of the model on the corresponding robust training
set (the samples which the classiﬁer was trained on) and test set (unseen samples from bDR, based on the test
set). We ﬁnd that the drop in robustness comes from a combination of generalization gap (the robustness
on the bDR test set is worse than it is on the robust training set) and distributional shift (the model performs
better on the robust test set consisting of unseen samples from bDR than on the standard test set containing
unseen samples from D).
Dataset
Robust Accuracy
Robust training set
77.33%
Robust test set
62.49%
Standard test set
48.27%
Table 8: Performance of model trained on the robust dataset on the robust training and test sets as well as
the standard CIFAR-10 test set. We observe that the drop in robust accuracy stems from a combination of
generalization gap and distributional shift. The adversary is constrained to ε = 0.25 in ℓ2-norm.
21


**Table 26 from page 21**

| 0                                                                                | 1                                                                                                                | 2      | 3          |
|:---------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------|:-------|:-----------|
| D.2                                                                              | Adversarial evaluation                                                                                           |        |            |
|                                                                                  | To verify the robustness of our classiﬁers trained on the ‘robust” dataset, we evaluate them with strong         |        |            |
| attacks [Car+19].                                                                | In particular, we try up to 2500 steps of projected gradient descent                                             | (PGD), | increasing |
|                                                                                  | steps until the accuracy plateaus, and also try the CW-(cid:96)2 loss function [CW17b] with 1000 steps. For each |        |            |
|                                                                                  | attack we search over step size. We ﬁnd that over all attacks and step sizes, the accuracy of the model does     |        |            |
|                                                                                  | not drop by more than 2%, and plateaus at 48.27% for both PGD and CW-(cid:96)2 (the value given in Figure 2).    |        |            |
| We show a plot of accuracy in terms of the number of PGD steps used in Figure 8. |                                                                                                                  |        |            |



## Page 22

D.4
Classiﬁcation based on non-robust features
Figure 9 shows sample images from D, bDrand and bDdet constructed using a standard (non-robust) ERM
classiﬁer, and an adversarially trained (robust) classiﬁer.
D
Using  
non-robust
Using  
robust
(a) bDrand
D
Using  
non-robust
Using  
robust
(b) bDdet
Figure 9: Random samples from datasets where the input-label correlation is entirely based on non-robust
features. Samples are generated by performing small adversarial perturbations using either random ( bDrand)
or deterministic ( bDdet) label-target mappings for every sample in the training set. Each image shows: top:
original; middle: adversarial perturbations using a standard ERM-trained classiﬁer; bottom: adversarial per-
turbations using a robust classiﬁer (adversarially trained against ε = 0.5).
In Table 9, we repeat the experiments in Table 1 based on datasets constructed using a robust model.
Note that using a robust model to generate the bDdet and bDrand datasets will not result in non-robust features
that are strongly predictive of t (since the prediction of the classiﬁer will not change). Thus, training a model
on these datasets leads to poor accuracy on the standard test set from D.
Observe from Figure 10 that models trained on datasets derived from the robust model show a decline
in test accuracy as training progresses. In Table 9, the accuracy numbers reported correspond to the last
iteration, and not the best performance. This is because we have no way to cross-validate in a meaningful
way as the validation set itself comes from bDrand or bDdet, and not from the true data distribution D. Thus,
validation accuracy will not be predictive of the true test accuracy, and thus will not help determine when
to early stop.
Model used
to construct dataset
Dataset used in training
D
bDrand
bDdet
Robust
95.3%
25.2 %
5.8%
Standard
95.3%
63.3 %
43.7%
Table 9: Repeating the experiments of Table 1 using a robust model to construct the datasets D, bDrand and
bDdet. Results in Table 1 are reiterated for comparison.
22


**Table 27 from page 22**

| 0                                                                                                   | 1                                                                                                                 | 2     |
|:----------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------|:------|
| D.4                                                                                                 |                                                                                                                   |       |
| Classiﬁcation based on non-robust features                                                          |                                                                                                                   |       |
| Figure 9 shows sample images from                                                                   | det constructed using a standard (non-robust) ERM                                                                 |       |
| ,                                                                                                   |                                                                                                                   |       |
| rand and                                                                                            |                                                                                                                   |       |
| D                                                                                                   |                                                                                                                   |       |
| (cid:98)D                                                                                           |                                                                                                                   |       |
| (cid:98)D                                                                                           |                                                                                                                   |       |
| classiﬁer, and an adversarially trained (robust) classiﬁer.                                         |                                                                                                                   |       |
| D                                                                                                   |                                                                                                                   |       |
| Using                                                                                               |                                                                                                                   |       |
| non-robust                                                                                          |                                                                                                                   |       |
| Using                                                                                               |                                                                                                                   |       |
| non-robust                                                                                          |                                                                                                                   |       |
| D                                                                                                   |                                                                                                                   |       |
| Using                                                                                               |                                                                                                                   |       |
| robust                                                                                              |                                                                                                                   |       |
| Using                                                                                               |                                                                                                                   |       |
| robust                                                                                              |                                                                                                                   |       |
| (a)                                                                                                 |                                                                                                                   |       |
| (b)                                                                                                 |                                                                                                                   |       |
| rand                                                                                                |                                                                                                                   |       |
| det                                                                                                 |                                                                                                                   |       |
| (cid:98)D                                                                                           | Figure 9: Random samples from datasets where the input-label correlation is entirely based on non-robust          |       |
| (cid:98)D                                                                                           |                                                                                                                   |       |
| features. Samples are generated by performing small adversarial perturbations using either random ( |                                                                                                                   | rand) |
| or deterministic (                                                                                  | (cid:98)D                                                                                                         | top:  |
|                                                                                                     | det) label-target mappings for every sample in the training set. Each image shows:                                |       |
| (cid:98)D                                                                                           | original; middle: adversarial perturbations using a standard ERM-trained classiﬁer; bottom: adversarial per-      |       |
| turbations using a robust classiﬁer (adversarially trained against ε = 0.5).                        |                                                                                                                   |       |
|                                                                                                     | In Table 9, we repeat the experiments in Table 1 based on datasets constructed using a robust model.              |       |
| Note that using a robust model to generate the                                                      | rand datasets will not result in non-robust features                                                              |       |
| det and                                                                                             |                                                                                                                   |       |
| (cid:98)D                                                                                           | that are strongly predictive of t (since the prediction of the classiﬁer will not change). Thus, training a model |       |
| (cid:98)D                                                                                           |                                                                                                                   |       |
| on these datasets leads to poor accuracy on the standard test set from                              |                                                                                                                   |       |
| .                                                                                                   |                                                                                                                   |       |
| D                                                                                                   |                                                                                                                   |       |
|                                                                                                     | Observe from Figure 10 that models trained on datasets derived from the robust model show a decline               |       |
| in test accuracy as training progresses.                                                            | the accuracy numbers reported correspond to the last                                                              |       |
| In Table 9,                                                                                         |                                                                                                                   |       |
|                                                                                                     | iteration, and not the best performance. This is because we have no way to cross-validate in a meaningful         |       |
| way as the validation set itself comes from                                                         | det, and not from the true data distribution D. Thus,                                                             |       |
| rand or                                                                                             |                                                                                                                   |       |
| (cid:98)D                                                                                           | validation accuracy will not be predictive of the true test accuracy, and thus will not help determine when       |       |
| (cid:98)D                                                                                           |                                                                                                                   |       |
| to early stop.                                                                                      |                                                                                                                   |       |
| Dataset used in training                                                                            |                                                                                                                   |       |
| Model used                                                                                          |                                                                                                                   |       |
| to construct dataset                                                                                |                                                                                                                   |       |
| rand                                                                                                |                                                                                                                   |       |
| det                                                                                                 |                                                                                                                   |       |
| D                                                                                                   |                                                                                                                   |       |
| (cid:98)D                                                                                           |                                                                                                                   |       |
| (cid:98)D                                                                                           |                                                                                                                   |       |
| Robust                                                                                              |                                                                                                                   |       |
| 95.3%                                                                                               |                                                                                                                   |       |
| 25.2 %                                                                                              |                                                                                                                   |       |
| 5.8%                                                                                                |                                                                                                                   |       |
| Standard                                                                                            |                                                                                                                   |       |
| 95.3%                                                                                               |                                                                                                                   |       |
| 63.3 %                                                                                              |                                                                                                                   |       |
| 43.7%                                                                                               |                                                                                                                   |       |
| Table 9: Repeating the experiments of Table 1 using a robust model to construct the datasets        | rand and                                                                                                          |       |
| ,                                                                                                   |                                                                                                                   |       |
| D                                                                                                   |                                                                                                                   |       |
| (cid:98)D                                                                                           |                                                                                                                   |       |
| (cid:98)D                                                                                           |                                                                                                                   |       |
| det. Results in Table 1 are reiterated for comparison.                                              |                                                                                                                   |       |
| 22                                                                                                  |                                                                                                                   |       |



## Page 23

D.5
Accuracy curves
25
50
75
100
125
150
175
200
Epoch
0
20
40
60
80
100
Accuracy (%)
Using ERM-trained Model
Train
Test
25
50
75
100
125
150
175
200
Epoch
0
20
40
60
80
100
Accuracy (%)
Using Robust Model
Train
Test
(a) Trained using bDrand training set
25
50
75
100
125
150
175
200
Epoch
0
20
40
60
80
100
Accuracy (%)
Using ERM-trained Model
Train
Test
20
40
60
80
100
120
140
160
180
Epoch
0
20
40
60
80
100
Accuracy (%)
Using Robust Model
Train
Test
(b) Trained using bDdet training set
Figure 10: Test accuracy on D of standard classiﬁers trained on datasets where input-label correlation is
based solely on non-robust features as in Section 3.2. The datasets are constructed using either a non-
robust/standard model (left column) or a robust model (right column). The labels used are either random
( bDrand; top row) or correspond to a deterministic permutation ( bDdet; bottom row).
23


**Table 28 from page 23**

| 0                                                                                                       |
|:--------------------------------------------------------------------------------------------------------|
| D.5                                                                                                     |
| Accuracy curves                                                                                         |
| Using ERM-trained Model                                                                                 |
| Using Robust Model                                                                                      |
| 100                                                                                                     |
| 100                                                                                                     |
| Train                                                                                                   |
| Train                                                                                                   |
| Test                                                                                                    |
| Test                                                                                                    |
| 80                                                                                                      |
| 80                                                                                                      |
| Accuracy (%)                                                                                            |
| Accuracy (%)                                                                                            |
| 60                                                                                                      |
| 60                                                                                                      |
| 40                                                                                                      |
| 40                                                                                                      |
| 20                                                                                                      |
| 20                                                                                                      |
| 0                                                                                                       |
| 0                                                                                                       |
| 25                                                                                                      |
| 50                                                                                                      |
| 75                                                                                                      |
| 100                                                                                                     |
| 125                                                                                                     |
| 150                                                                                                     |
| 175                                                                                                     |
| 200                                                                                                     |
| 25                                                                                                      |
| 50                                                                                                      |
| 75                                                                                                      |
| 100                                                                                                     |
| 125                                                                                                     |
| 150                                                                                                     |
| 175                                                                                                     |
| 200                                                                                                     |
| Epoch                                                                                                   |
| Epoch                                                                                                   |
| (a) Trained using                                                                                       |
| rand training set                                                                                       |
| (cid:98)D                                                                                               |
| Using ERM-trained Model                                                                                 |
| Using Robust Model                                                                                      |
| 100                                                                                                     |
| 100                                                                                                     |
| Train                                                                                                   |
| Test                                                                                                    |
| 80                                                                                                      |
| 80                                                                                                      |
| Accuracy (%)                                                                                            |
| Accuracy (%)                                                                                            |
| 60                                                                                                      |
| 60                                                                                                      |
| 40                                                                                                      |
| 40                                                                                                      |
| 20                                                                                                      |
| 20                                                                                                      |
| Train                                                                                                   |
| Test                                                                                                    |
| 0                                                                                                       |
| 0                                                                                                       |
| 25                                                                                                      |
| 50                                                                                                      |
| 75                                                                                                      |
| 100                                                                                                     |
| 125                                                                                                     |
| 150                                                                                                     |
| 175                                                                                                     |
| 200                                                                                                     |
| 20                                                                                                      |
| 40                                                                                                      |
| 60                                                                                                      |
| 80                                                                                                      |
| 100                                                                                                     |
| 120                                                                                                     |
| 140                                                                                                     |
| 160                                                                                                     |
| 180                                                                                                     |
| Epoch                                                                                                   |
| Epoch                                                                                                   |
| (cid:98)D                                                                                               |
| (b) Trained using                                                                                       |
| det training set                                                                                        |
| Figure 10: Test accuracy on                                                                             |
| of standard classiﬁers trained on datasets where input-label correlation is                             |
| D                                                                                                       |
| based solely on non-robust                                                                              |
| features as in Section 3.2.                                                                             |
| The datasets are constructed using either a non-                                                        |
| robust/standard model (left column) or a robust model (right column). The labels used are either random |
| (cid:98)D                                                                                               |
| (cid:98)D                                                                                               |
| (                                                                                                       |
| rand; top row) or correspond to a deterministic permutation (                                           |
| det; bottom row).                                                                                       |
| 23                                                                                                      |



## Page 24

D.6
Performance of ERM classiﬁers on relabeled test set
In Table 10), we evaluate the performance of classiﬁers trained on bDdet on both the original test set drawn
from D, and the test set relabelled using t(y) = (y + 1) mod C. Observe that the classiﬁer trained on
bDdet constructed using a robust model actually ends up learning permuted labels based on robust features
(indicated by high test accuracy on the relabelled test set).
Model used to construct
training dataset for bDdet
Dataset used in testing
D
relabelled-D
Standard
43.7%
16.2%
Robust
5.8%
65.5%
Table 10: Performance of classiﬁers trained using bDdet training set constructed using either standard or
robust models. The classiﬁers are evaluated both on the standard test set from D and the test set relabeled
using t(y) = (y + 1) mod C. We observe that using a robust model for the construction results in a model
that largely predicts the permutation of labels, indicating that the dataset does not have strongly predictive
non-robust features.
D.7
Generalization to CIFAR-10.1
Recht et al. [Rec+19] have constructed an unseen but distribution-shifted test set for CIFAR-10. They show
that for many previously proposed models, accuracy on the CIFAR-10.1 test set can be predicted as a linear
function of performance on the CIFAR-10 test set.
As a sanity check (and a safeguard against any potential adaptive overﬁtting to the test set via hyper-
parameters, historical test set reuse, etc.) we note that the classiﬁers trained on bDdet and bDrand achieve 44%
and 55% generalization on the CIFAR-10.1 test set, respectively. This demonstrates non-trivial generaliza-
tion, and actually perform better than the linear ﬁt would predict (given their accuracies on the CIFAR-10
test set).
24


**Table 29 from page 24**

| 0                                                                                                              |
|:---------------------------------------------------------------------------------------------------------------|
| D.6                                                                                                            |
| Performance of ERM classiﬁers on relabeled test set                                                            |
| In Table 10), we evaluate the performance of classiﬁers trained on                                             |
| det on both the original test set drawn                                                                        |
| from                                                                                                           |
| , and the test set relabelled using t(y) = (y + 1) mod C. Observe that                                         |
| the classiﬁer trained on                                                                                       |
| (cid:98)D                                                                                                      |
| D                                                                                                              |
| det constructed using a robust model actually ends up learning permuted labels based on robust features        |
| (cid:98)D                                                                                                      |
| (indicated by high test accuracy on the relabelled test set).                                                  |
| Model used to construct                                                                                        |
| Dataset used in testing                                                                                        |
| training dataset for                                                                                           |
| det                                                                                                            |
| relabelled-                                                                                                    |
| (cid:98)D                                                                                                      |
| D                                                                                                              |
| D                                                                                                              |
| Standard                                                                                                       |
| 43.7%                                                                                                          |
| 16.2%                                                                                                          |
| Robust                                                                                                         |
| 5.8%                                                                                                           |
| 65.5%                                                                                                          |
| Table 10: Performance of classiﬁers trained using                                                              |
| training set constructed using either standard or                                                              |
| det                                                                                                            |
| robust models. The classiﬁers are evaluated both on the standard test set from                                 |
| and the test set relabeled                                                                                     |
| (cid:98)D                                                                                                      |
| D                                                                                                              |
| using t(y) = (y + 1) mod C. We observe that using a robust model for the construction results in a model       |
| that largely predicts the permutation of labels, indicating that the dataset does not have strongly predictive |
| non-robust features.                                                                                           |
| D.7                                                                                                            |
| Generalization to CIFAR-10.1                                                                                   |
| Recht et al. [Rec+19] have constructed an unseen but distribution-shifted test set for CIFAR-10. They show     |
| that for many previously proposed models, accuracy on the CIFAR-10.1 test set can be predicted as a linear     |
| function of performance on the CIFAR-10 test set.                                                              |
| As a sanity check (and a safeguard against any potential adaptive overﬁtting to the test set via hyper-        |
| parameters, historical test set reuse, etc.) we note that the classiﬁers trained on                            |
| det and                                                                                                        |
| rand achieve 44%                                                                                               |
| (cid:98)D                                                                                                      |
| (cid:98)D                                                                                                      |
| and 55% generalization on the CIFAR-10.1 test set, respectively. This demonstrates non-trivial generaliza-     |
| tion, and actually perform better than the linear ﬁt would predict (given their accuracies on the CIFAR-10     |
| test set).                                                                                                     |
| 24                                                                                                             |



## Page 25

D.8
Omitted Results for Restricted ImageNet
“dog’’
“primate’’
“insect’’
“crab’’
“bird’’
D
!DR
Figure 11: Repeating the experiments shown in Figure 2 for the Restricted ImageNet dataset. Sample
images from the resulting dataset.
Std Training 
 using 
Adv Training 
 using 
Std Training 
 using 
R
0
20
40
60
80
100
Test Accuracy on 
 (%)
Std accuracy
Adv accuracy ( = 0.5)
Figure 12: Repeating the experiments shown in Figure 2 for the Restricted ImageNet dataset. Standard and
robust accuracy of models trained on these datasets.
25


**Table 30 from page 25**

| 0                                                                                                        | 1                                | 2      |
|:---------------------------------------------------------------------------------------------------------|:---------------------------------|:-------|
| D.8                                                                                                      |                                  |        |
| Omitted Results for Restricted ImageNet                                                                  |                                  |        |
| “dog’’                                                                                                   | “crab’’                          |        |
| “primate’’                                                                                               | “bird’’                          |        |
| “insect’’                                                                                                |                                  |        |
| D                                                                                                        |                                  |        |
| R                                                                                                        |                                  |        |
| !D                                                                                                       |                                  |        |
| Figure 11: Repeating the experiments shown in Figure 2 for                                               | the Restricted ImageNet dataset. | Sample |
| images from the resulting dataset.                                                                       |                                  |        |
| 100                                                                                                      | Adv accuracy ( = 0.5)            |        |
| Std accuracy                                                                                             |                                  |        |
| (%)                                                                                                      |                                  |        |
| 80                                                                                                       |                                  |        |
| 60                                                                                                       |                                  |        |
| Test Accuracy on                                                                                         |                                  |        |
| 40                                                                                                       |                                  |        |
| 20                                                                                                       |                                  |        |
| 0                                                                                                        |                                  |        |
| Std Training                                                                                             | Std Training                     |        |
| Adv Training                                                                                             |                                  |        |
| using                                                                                                    | R                                |        |
|  using                                                                                                   |  using                           |        |
| Figure 12: Repeating the experiments shown in Figure 2 for the Restricted ImageNet dataset. Standard and |                                  |        |
| robust accuracy of models trained on these datasets.                                                     |                                  |        |
| 25                                                                                                       |                                  |        |



## Page 26

D.9
Targeted Transferability
25
30
35
40
45
Test accuracy (%; trained on Dy + 1)
0
10
20
30
40
50
60
70
80
Transfer success rate (%)
VGG-16
Inception-v3
ResNet-18
DenseNet
ResNet-50
Figure 13: Transfer rate of targeted adversarial examples (measured in terms of attack success rate, not just
misclassiﬁcation) from a ResNet-50 to different architectures alongside test set performance of these archi-
tecture when trained on the dataset generated in Section 3.2. Architectures more susceptible to transfer
attacks also performed better on the standard test set supporting our hypothesis that adversarial transfer-
ability arises from utilizing similar non-robust features.
D.10
Robustness vs. Accuracy
1.0
0.5
0.0
0.5
1.0
1.0
0.5
0.0
0.5
1.0
>
/2
ERM Classifier
Robust + Accurate Classifier
Figure 14: An example where adversarial vulnerability can arise from ERM training on any standard loss
function due to non-robust features (the green line shows the ERM-learned decision boundary). There
exists, however, a classiﬁer that is both perfectly robust and accurate, resulting from robust training, which
forces the classiﬁer to ignore the x2 feature despite its predictiveness.
26


**Table 31 from page 26**

| 0                         | 1   | 2      | 3            |
|:--------------------------|:----|:-------|:-------------|
|                           | 80  |        |              |
|                           |     |        | ResNet-50    |
|                           | 70  |        |              |
|                           | 60  |        |              |
|                           | 50  |        |              |
| Transfer success rate (%) | 40  |        |              |
|                           |     |        | DenseNet     |
|                           |     |        | ResNet-18    |
|                           | 30  |        |              |
|                           |     |        | Inception-v3 |
|                           | 20  |        |              |
|                           |     | VGG-16 |              |
|                           | 10  |        |              |
|                           | 0   |        |              |



## Page 27

E
Gaussian MLE under Adversarial Perturbation
In this section, we develop a framework for studying non-robust features by studying the problem of max-
imum likelihood classiﬁcation between two Gaussian distributions. We ﬁrst recall the setup of the problem,
then present the main theorems from Section 4. First we build the techniques necessary for their proofs.
E.1
Setup
We consider the setup where a learner receives labeled samples from two distributions, N (µ∗, Σ∗), and
N (−µ∗, Σ∗). The learner’s goal is to be able to classify new samples as being drawn from D1 or D2 accord-
ing to a maximum likelihood (MLE) rule.
A simple coupling argument demonstrates that this problem can actually be reduced to learning the
parameters bµ, bΣ of a single Gaussian N (−µ∗, Σ∗), and then employing a linear classiﬁer with weight bΣ−1bµ.
In the standard setting, maximum likelihoods estimation learns the true parameters, µ∗and Σ∗, and thus
the learned classiﬁcation rule is C(x) = 1{x⊤Σ−1µ > 0}.
In this work, we consider the problem of adversarially robust maximum likelihood estimation. In partic-
ular, rather than simply being asked to classify samples, the learner will be asked to classify adversarially
perturbed samples x + δ, where δ ∈∆is chosen to maximize the loss of the learner. Our goal is to derive the
parameters µ, Σ corresponding to an adversarially robust maximum likelihood estimate of the parameters
of N (µ∗, Σ∗). Note that since we have access to Σ∗(indeed, the learner can just run non-robust MLE to get
access), we work in the space where Σ∗is a diagonal matrix, and we restrict the learned covariance Σ to the
set of diagonal matrices.
Notation.
We denote the parameters of the sampled Gaussian by µ∗∈Rd, and Σ∗∈{diag(u)|u ∈Rd}.
We use σmin(X) to represent the smallest eigenvalue of a square matrix X, and ℓ(·; x) to represent the
Gaussian negative log-likelihood for a single sample x. For convenience, we often use v = x −µ, and
R = ∥µ∗∥. We also deﬁne the  operator to represent the vectorization of the diagonal of a matrix. In
particular, for a matrix X ∈Rd×d, we have that X = v ∈Rd if vi = Xii.
E.2
Outline and Key Results
We focus on the case where ∆= B2(ϵ) for some ϵ > 0, i.e. the ℓ2 ball, corresponding to the following
minimax problem:
min
µ,Σ Ex∼N (µ∗,Σ∗)

max
δ:∥δ∥=ε ℓ(µ, Σ; x + δ)

(13)
We ﬁrst derive the optimal adversarial perturbation for this setting (Section E.3.1), and prove Theorem 1
(Section E.3.2). We then propose an alternate problem, in which the adversary picks a linear operator to
be applied to a ﬁxed vector, rather than picking a speciﬁc perturbation vector (Section E.3.3). We argue
via Gaussian concentration that the alternate problem is indeed reﬂective of the original model (and in
particular, the two become equivalent as d →∞). In particular, we propose studying the following in place
of (13):
min
µ,Σ max
M∈M Ex∼N (µ∗,Σ∗) [ℓ(µ, Σ; x + M(x −µ))]
(14)
where M =
n
M ∈Rd×d : Mij = 0 ∀i ̸= j, Ex∼N (µ∗,Σ∗)
h
∥Mv∥2
2
i
= ϵ2o
.
Our goal is to characterize the behavior of the robustly learned covariance Σ in terms of the true covari-
ance matrix Σ∗and the perturbation budget ε. The proof is through Danskin’s Theorem, which allows us
to use any maximizer of the inner problem M∗in computing the subgradient of the inner minimization.
After showing the applicability of Danskin’s Theorem (Section E.3.4) and then applying it (Section E.3.5) to
prove our main results (Section E.3.7). Our three main results, which we prove in the following section, are
presented below.
First, we consider a simpliﬁed version of (13), in which the adversary solves a maximization with a ﬁxed
Lagrangian penalty, rather than a hard ℓ2 constraint. In this setting, we show that the loss contributed by
27


**Table 32 from page 27**

| 0                                                                                                          |
|:-----------------------------------------------------------------------------------------------------------|
| E                                                                                                          |
| Gaussian MLE under Adversarial Perturbation                                                                |
| In this section, we develop a framework for studying non-robust features by studying the problem of max-   |
| imum likelihood classiﬁcation between two Gaussian distributions. We ﬁrst recall the setup of the problem, |
| then present the main theorems from Section 4. First we build the techniques necessary for their proofs.   |
| E.1                                                                                                        |
| Setup                                                                                                      |
| (µ                                                                                                         |
| We consider the setup where a learner receives labeled samples from two distributions,                     |
| , Σ                                                                                                        |
| ), and                                                                                                     |
| ∗                                                                                                          |
| ∗                                                                                                          |
| N                                                                                                          |
| (                                                                                                          |
| µ                                                                                                          |
| , Σ                                                                                                        |
| ). The learner’s goal is to be able to classify new samples as being drawn from                            |
| ∗                                                                                                          |
| ∗                                                                                                          |
| N                                                                                                          |
| −                                                                                                          |
| D1 or                                                                                                      |
| D2 accord-                                                                                                 |
| ing to a maximum likelihood (MLE) rule.                                                                    |
| A simple coupling argument demonstrates that                                                               |
| this problem can actually be reduced to learning the                                                       |
| 1                                                                                                          |
| Σ                                                                                                          |
| (                                                                                                          |
| µ                                                                                                          |
| parameters                                                                                                 |
| µ,                                                                                                         |
| Σ of a single Gaussian                                                                                     |
| , Σ                                                                                                        |
| ), and then employing a linear classiﬁer with weight                                                       |
| µ.                                                                                                         |
| −                                                                                                          |
| ∗                                                                                                          |
| ∗                                                                                                          |
| N                                                                                                          |
| −                                                                                                          |
| In the standard setting, maximum likelihoods estimation learns the true parameters, µ                      |
| and Σ                                                                                                      |
| , and thus                                                                                                 |
| ∗                                                                                                          |
| ∗                                                                                                          |
| Σ                                                                                                          |
| the learned classiﬁcation rule is C(x) = 1                                                                 |
| 1µ > 0                                                                                                     |
| .                                                                                                          |
| (cid:98)                                                                                                   |
| x(cid:62)                                                                                                  |
| −                                                                                                          |
| (cid:98)                                                                                                   |
| (cid:98)                                                                                                   |
| {                                                                                                          |
| }                                                                                                          |
| In this work, we consider the problem of adversarially robust maximum likelihood estimation.               |
| In partic-                                                                                                 |
| ular, rather than simply being asked to classify samples,                                                  |
| the learner will be asked to classify adversarially                                                        |
| perturbed samples x + δ, where δ                                                                           |
| ∆ is chosen to maximize the loss of the learner. Our goal is to derive the                                 |
| ∈                                                                                                          |
| parameters µ, Σ corresponding to an adversarially robust maximum likelihood estimate of the parameters     |
| (µ                                                                                                         |
| of                                                                                                         |
| , Σ                                                                                                        |
| ). Note that since we have access to Σ                                                                     |
| (indeed, the learner can just run non-robust MLE to get                                                    |
| N                                                                                                          |
| access), we work in the space where Σ                                                                      |
| ∗ is a diagonal matrix, and we restrict the learned covariance Σ to the                                    |
| set of diagonal matrices.                                                                                  |
| Rd                                                                                                         |
| u                                                                                                          |
| Notation. We denote the parameters of the sampled Gaussian by µ                                            |
| Rd, and Σ                                                                                                  |
| diag(u)                                                                                                    |
| .                                                                                                          |
| ∗ ∈                                                                                                        |
| ∗ ∈ {                                                                                                      |
| |                                                                                                          |
| ∈                                                                                                          |
| }                                                                                                          |
| to represent                                                                                               |
| the smallest eigenvalue of a square matrix X, and (cid:96)(                                                |
| ; x)                                                                                                       |
| to represent                                                                                               |
| the                                                                                                        |
| We use σmin(X)                                                                                             |
| ·                                                                                                          |
| Gaussian negative log-likelihood for a single sample x.                                                    |
| For convenience, we often use v = x                                                                        |
| µ, and                                                                                                     |
| −                                                                                                          |
| R =                                                                                                        |
| µ                                                                                                          |
| . We also deﬁne the                                                                                        |
| operator to represent                                                                                      |
| the vectorization of                                                                                       |
| the diagonal of a matrix.                                                                                  |
| In                                                                                                         |
| (cid:107)                                                                                                  |
| ∗(cid:107)                                                                                                 |
| Rd                                                                                                         |
| = v                                                                                                        |
| particular, for a matrix X                                                                                 |
| d, we have that X                                                                                          |
| Rd if vi = Xii.                                                                                            |
| ×                                                                                                          |
| ∈                                                                                                          |
| ∈                                                                                                          |
| (cid:21)                                                                                                   |
| E.2                                                                                                        |
| Outline and Key Results                                                                                    |
| We focus on the case where ∆ =                                                                             |
| i.e.                                                                                                       |
| the (cid:96)2 ball, corresponding to the following                                                         |
| B2((cid:101)) for some (cid:101) > 0,                                                                      |
| minimax problem:                                                                                           |
| E                                                                                                          |
| (cid:96)(µ, Σ; x + δ)                                                                                      |
| min                                                                                                        |
| (13)                                                                                                       |
| max                                                                                                        |
| x                                                                                                          |
| (µ∗,Σ                                                                                                      |
| ∗)                                                                                                         |
| µ,Σ                                                                                                        |
| δ                                                                                                          |
| =ε                                                                                                         |
| δ:                                                                                                         |
| ∼N                                                                                                         |
| (cid:107)                                                                                                  |
| (cid:107)                                                                                                  |
| (cid:21)                                                                                                   |
| (cid:20)                                                                                                   |
| We ﬁrst derive the optimal adversarial perturbation for this setting (Section E.3.1), and prove Theorem 1  |
| (Section E.3.2). We then propose an alternate problem,                                                     |
| in which the adversary picks a linear operator to                                                          |
| be applied to a ﬁxed vector, rather than picking a speciﬁc perturbation vector (Section E.3.3). We argue   |
| via Gaussian concentration that                                                                            |
| the alternate problem is indeed reﬂective of                                                               |
| the original model                                                                                         |
| (and in                                                                                                    |
| particular, the two become equivalent as d                                                                 |
| ∞). In particular, we propose studying the following in place                                              |
| →                                                                                                          |
| of (13):                                                                                                   |
| E                                                                                                          |



## Page 28

the adversary corresponds to a misalignment between the data metric (the Mahalanobis distance, induced
by Σ−1), and the ℓ2 metric:
Theorem 1 (Adversarial vulnerability from misalignment). Consider an adversary whose perturbation is deter-
mined by the “Lagrangian penalty” form of (12), i.e.
max
δ
ℓ(x + δ; y · µ, Σ) −C · ∥δ∥2,
where C ≥
1
σmin(Σ∗) is a constant trading off NLL minimization and the adversarial constraint17. Then, the adversarial
loss Ladv incurred by the non-robustly learned (µ, Σ) is given by:
Ladv(Θ) −L(Θ) = tr

I + (C · Σ∗−I)−12
−d,
and, for a ﬁxed tr(Σ∗) = k the above is minimized by Σ∗= k
d I.
We then return to studying (14), where we provide upper and lower bounds on the learned robust covari-
ance matrix Σ:
Theorem 2 (Robustly Learned Parameters). Just as in the non-robust case, µr = µ∗, i.e. the true mean is learned.
For the robust covariance Σr, there exists an ε0 > 0, such that for any ε ∈[0, ε0),
Σr = 1
2Σ∗+ 1
λ · I +
r
1
λ · Σ∗+ 1
4Σ2∗,
where
Ω
 
1 + ε1/2
ε1/2 + ε3/2
!
≤λ ≤O
 
1 + ε1/2
ε1/2
!
.
Finally, we show that in the worst case over mean vectors µ∗, the gradient of the adversarial robust classiﬁer
aligns more with the inter-class vector:
Theorem 3 (Gradient alignment). Let f (x) and fr(x) be monotonic classiﬁers based on the linear separator induced
by standard and ℓ2-robust maximum likelihood classiﬁcation, respectively. The maximum angle formed between the
gradient of the classiﬁer (wrt input) and the vector connecting the classes can be smaller for the robust model:
min
µ
⟨µ, ∇x fr(x)⟩
∥µ∥· ∥∇x fr(x)∥> min
µ
⟨µ, ∇x f (x)⟩
∥µ∥· ∥∇x f (x)∥.
E.3
Proofs
In the ﬁrst section, we have shown that the classiﬁcation between two Gaussian distributions with identical
covariance matrices centered at µ∗and −µ∗can in fact be reduced to learning the parameters of a single
one of these distributions.
Thus, in the standard setting, our goal is to solve the following problem:
min
µ,Σ Ex∼N (µ∗,Σ∗) [ℓ(µ, Σ; x)] := min
µ,Σ Ex∼N (µ∗,Σ∗) [−log (N (µ, Σ; x))] .
Note that in this setting, one can simply ﬁnd differentiate ℓwith respect to both µ and Σ, and obtain
closed forms for both (indeed, these closed forms are, unsurprisingly, µ∗and Σ∗). Here, we consider the
existence of a malicious adversary who is allowed to perturb each sample point x by some δ. The goal of the
adversary is to maximize the same loss that the learner is minimizing.
E.3.1
Motivating example: ℓ2-constrained adversary
We ﬁrst consider, as a motivating example, an ℓ2-constrained adversary. That is, the adversary is allowed
to perturb each sampled point by δ : ∥δ∥2 = ε. In this case, the minimax problem being solved is the
following:
min
µ,Σ Ex∼N (µ∗,Σ∗)

max
∥δ∥=ε ℓ(µ, Σ; x + δ)

.
(15)
The following Lemma captures the optimal behaviour of the adversary:
17The constraint on C is to ensure the problem is concave.
28


**Table 33 from page 28**

| 0                                                                                                           |
|:------------------------------------------------------------------------------------------------------------|
| the adversary corresponds to a misalignment between the data metric (the Mahalanobis distance, induced      |
| by Σ                                                                                                        |
| 1), and the (cid:96)2 metric:                                                                               |
| −                                                                                                           |
| Theorem 1 (Adversarial vulnerability from misalignment). Consider an adversary whose perturbation is deter- |
| mined by the “Lagrangian penalty” form of                                                                   |
| (12), i.e.                                                                                                  |
| C                                                                                                           |
| δ                                                                                                           |
| max                                                                                                         |
| (cid:96)(x + δ; y                                                                                           |
| µ, Σ)                                                                                                       |
| δ                                                                                                           |
| ·                                                                                                           |
| −                                                                                                           |
| · (cid:107)                                                                                                 |
| (cid:107)2,                                                                                                 |
| 1                                                                                                           |
| where C                                                                                                     |
| is a constant trading off NLL minimization and the adversarial constraint17. Then, the adversarial          |
| )                                                                                                           |
| σmin(Σ                                                                                                      |
| ≥                                                                                                           |
| loss                                                                                                        |
| Ladv incurred by the non-robustly learned (µ, Σ) is given by:                                               |
| 2                                                                                                           |



## Page 29

Lemma 1. In the minimax problem captured in (15) (and earlier in (13)), the optimal adversarial perturbation δ∗is
given by
δ∗=

λI −Σ−1−1
Σ−1v = (λΣ −I)−1 v,
(16)
where v = x −µ, and λ is set such that ∥δ∗∥2 = ε.
Proof. In this context, we can solve the inner maximization problem with Lagrange multipliers. In the
following we write ∆= B2(ε) for brevity, and discard terms not containing δ as well as constant factors
freely:
arg max
δ∈∆ℓ(µ, Σ; x + δ)−= arg max
δ∈∆(x + δ −µ)⊤Σ−1 (x + δ −µ)
= arg max
δ∈∆(x −µ)⊤Σ−1(x −µ) + 2δ⊤Σ−1(x −µ) + δ⊤Σ−1δ
= arg max
δ∈∆δ⊤Σ−1(x −µ) + 1
2δ⊤Σ−1δ.
(17)
Now we can solve (17) using the aforementioned Lagrange multipliers. In particular, note that the maxi-
mum of (17) is attained at the boundary of the ℓ2 ball ∆. Thus, we can solve the following system of two
equations to ﬁnd δ, rewriting the norm constraint as 1
2∥δ∥2
2 = 1
2ε2:
(
∇δ

δ⊤Σ−1(x −µ) + 1
2δ⊤Σ−1δ

= λ∇δ
 ∥δ∥2
2 −ε2 =⇒Σ−1(x −µ) + Σ−1δ = λδ
∥δ∥2
2 = ε2.
(18)
For clarity, we write v = x −µ: then, combining the above, we have that
δ∗=

λI −Σ−1−1
Σ−1v = (λΣ −I)−1 v,
(19)
our ﬁnal result for the maximizer of the inner problem, where λ is set according to the norm constraint.
E.3.2
Variant with Fixed Lagrangian (Theorem 1)
To simplify the analysis of Theorem 1, we consider a version of (15) with a ﬁxed Lagrangian penalty, rather
than a norm constraint:
max ℓ(x + δ; y · µ, Σ) −C · ∥δ∥2.
Note then, that by Lemma 1, the optimal perturbation δ∗is given by
δ∗= (CΣ −I)−1 .
We now proceed to the proof of Theorem 1.
Theorem 1 (Adversarial vulnerability from misalignment). Consider an adversary whose perturbation is deter-
mined by the “Lagrangian penalty” form of (12), i.e.
max
δ
ℓ(x + δ; y · µ, Σ) −C · ∥δ∥2,
where C ≥
1
σmin(Σ∗) is a constant trading off NLL minimization and the adversarial constraint18. Then, the adversarial
loss Ladv incurred by the non-robustly learned (µ, Σ) is given by:
Ladv(Θ) −L(Θ) = tr

I + (C · Σ∗−I)−12
−d,
and, for a ﬁxed tr(Σ∗) = k the above is minimized by Σ∗= k
d I.
18The constraint on C is to ensure the problem is concave.
29


**Table 34 from page 29**

| 0                                                                                                         |
|:----------------------------------------------------------------------------------------------------------|
| Lemma 1.                                                                                                  |
| In the minimax problem captured in (15) (and earlier in (13)), the optimal adversarial perturbation δ∗ is |
| given by                                                                                                  |
| 1                                                                                                         |
| 1                                                                                                         |
| −                                                                                                         |
| λI                                                                                                        |
| 1v = (λΣ                                                                                                  |
| 1 v,                                                                                                      |
| (16)                                                                                                      |
| I)−                                                                                                       |
| δ∗ =                                                                                                      |
| Σ−                                                                                                        |
| Σ−                                                                                                        |
| −                                                                                                         |
| −                                                                                                         |
| (cid:16)                                                                                                  |
| (cid:17)                                                                                                  |
| where v = x                                                                                               |
| µ, and λ is set such that                                                                                 |
| −                                                                                                         |
| (cid:107)                                                                                                 |
| δ∗(cid:107)2 = ε.                                                                                         |
| Proof.                                                                                                    |
| In this context, we can solve the inner maximization problem with Lagrange multipliers.                   |
| In the                                                                                                    |
| following we write ∆ =                                                                                    |
| B2(ε) for brevity, and discard terms not containing δ as well as constant factors                         |
| freely:                                                                                                   |



## Page 30

Proof. We begin by expanding the Gaussian negative log-likelihood for the relaxed problem:
Ladv(Θ) −L(Θ) = Ex∼N (µ∗,Σ∗)
h
2 · v⊤(C · Σ −I)−⊤Σ−1v + v⊤(C · Σ −I)−⊤Σ−1 (C · Σ −I)−1 v
i
= Ex∼N (µ∗,Σ∗)
h
2 · v⊤(C · ΣΣ −Σ)−1 v + v⊤(C · Σ −I)−⊤Σ−1 (C · Σ −I)−1 v
i
Recall that we are considering the vulnerability at the MLE parameters µ∗and Σ∗:
Ladv(Θ) −L(Θ) = Ev∼N (0,I)

2 · v⊤Σ1/2
∗

C · Σ2
∗−Σ∗
−1
Σ1/2
∗
v
+ v⊤Σ1/2
∗
(C · Σ∗−I)−⊤Σ−1
∗
(C · Σ∗−I)−1 Σ1/2
∗
v
i
= Ev∼N (0,I)

2 · v⊤(C · Σ∗−I)−1 v + v⊤Σ1/2
∗

C2Σ3
∗−2C · Σ2
∗+ Σ∗
−1
Σ1/2
∗
v

= Ev∼N (0,I)
h
2 · v⊤(C · Σ∗−I)−1 v + v⊤(C · Σ∗−I)−2 v
i
= Ev∼N (0,I)
h
−∥v∥2
2 + v⊤Iv + 2 · v⊤(C · Σ∗−I)−1 v + v⊤(C · Σ∗−I)−2 v
i
= Ev∼N (0,I)

−∥v∥2
2 + v⊤
I + (C · Σ∗−I)−12
v

= tr

I + (C · Σ∗−I)−12
−d
This shows the ﬁrst part of the theorem. It remains to show that for a ﬁxed k = tr(Σ∗), the adversarial risk
is minimized by Σ∗= k
d I:
min
Σ∗
Ladv(Θ) −L(Θ) = min
Σ∗
tr

I + (C · Σ∗−I)−12
= min
{σi}
d
∑
i=1

1 +
1
C · σi −1
2
,
where {σi} are the eigenvalues of Σ∗. Now, we have that ∑σi = k by assumption, so by optimality condi-
tions, we have that Σ∗minimizes the above if ∇{σi} ∝⃗1, i.e. if ∇σi = ∇σj for all i, j. Now,
∇σi = −2 ·

1 +
1
C · σi −1

·
C
(C · σi −1)2
= −2 ·
C2 · σi
(C · σi −1)3 .
Then, by solving analytically, we ﬁnd that
−2 ·
C2 · σi
(C · σi −1)3 = −2 ·
C2 · σj
(C · σj −1)3
admits only one real solution, σi = σj. Thus, Σ∗∝I. Scaling to satisfy the trace constraint yields Σ∗= k
d I,
which concludes the proof.
E.3.3
Real objective
Our motivating example (Section E.3.1) demonstrates that the optimal perturbation for the adversary in the
ℓ2-constrained case is actually a linear function of v, and in particular, that the optimal perturbation can
be expressed as Dv for a diagonal matrix D. Note, however, that the problem posed in (15) is not actually
30


**Table 35 from page 30**

| 0                                                                               | 1                  | 2    | 3   | 4   | 5        | 6   | 7   | 8         | 9         | 10       | 11        |
|:--------------------------------------------------------------------------------|:-------------------|:-----|:----|:----|:---------|:----|:----|:----------|:----------|:---------|:----------|
| (cid:104)                                                                       |                    |      |     |     |          |     |     |           |           |          | (cid:105) |
| ΣΣ                                                                              |                    |      |     | Σ   |          |     |     |           |           |          |           |
| Σ                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| = E                                                                             | I)−(cid:62) Σ−     | 1 (C |     |     |          |     | I)− | 1 v       |           |          |           |
| 2                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| Σ)−                                                                             |                    |      |     |     |          |     |     |           |           |          |           |
| v(cid:62) (C                                                                    |                    |      |     |     |          |     |     |           |           |          |           |
| 1 v + v(cid:62) (C                                                              |                    |      |     |     |          |     |     |           |           |          |           |
| x                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| (µ∗,Σ                                                                           |                    |      |     |     |          |     |     |           |           |          |           |
| ∗)                                                                              |                    |      |     |     |          |     |     |           |           |          |           |
| ·                                                                               |                    |      | ·   |     | −        |     |     |           |           |          |           |
| ·                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| −                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| ·                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| −                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| ∼N                                                                              |                    |      |     |     |          |     |     |           |           |          |           |
| (cid:104)                                                                       |                    |      |     |     |          |     |     |           | (cid:105) |          |           |
| Recall that we are considering the vulnerability at the MLE parameters µ∗ and Σ |                    |      |     |     |          |     |     |           |           |          |           |
| ∗:                                                                              |                    |      |     |     |          |     |     |           |           |          |           |
| 1                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| −                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| Σ2                                                                              |                    |      |     |     |          |     |     |           |           |          |           |
| Σ                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| Σ1/2                                                                            |                    |      |     |     |          |     |     |           |           |          |           |
| (Θ) = E                                                                         |                    |      |     |     |          |     |     |           |           |          |           |
| v                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| C                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| 2                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| v(cid:62)Σ1/2                                                                   |                    |      |     |     |          |     |     |           |           |          |           |
| v                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| (0,I)                                                                           |                    |      |     |     |          |     |     |           |           |          |           |
| Ladv(Θ)                                                                         |                    |      |     |     |          |     |     |           |           |          |           |
| ∗                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| − L                                                                             |                    |      |     |     |          |     |     |           |           |          |           |
| ·                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| ·                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| ∗                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| ∗ −                                                                             |                    |      |     |     |          |     |     |           |           |          |           |
| ∗                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| ∼N                                                                              |                    |      |     |     |          |     |     |           |           |          |           |
| (cid:20)                                                                        |                    |      |     |     |          |     |     |           |           |          |           |
| (cid:16)                                                                        |                    |      |     |     |          |     |     |           |           |          |           |
| (cid:17)                                                                        |                    |      |     |     |          |     |     |           |           |          |           |
| 1                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| Σ                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| Σ                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| 1 Σ1/2                                                                          |                    |      |     |     |          |     |     |           |           |          |           |
| v                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| (C                                                                              |                    |      |     |     |          |     |     |           |           |          |           |
| (C                                                                              |                    |      |     |     |          |     |     |           |           |          |           |
| I)−                                                                             |                    |      |     |     |          |     |     |           |           |          |           |
| + v(cid:62)Σ1/2                                                                 |                    |      |     |     |          |     |     |           |           |          |           |
| I)−(cid:62) Σ−                                                                  |                    |      |     |     |          |     |     |           |           |          |           |
| ·                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| ∗ −                                                                             |                    |      |     |     |          |     |     |           |           |          |           |
| ·                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| ∗ −                                                                             |                    |      |     |     |          |     |     |           |           |          |           |
| ∗                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| ∗                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| ∗                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| (cid:105)                                                                       |                    |      |     |     |          |     | 1   |           |           |          |           |
| Σ                                                                               | Σ2                 |      |     |     |          | −   |     | Σ1/2      |           |          |           |
| = E                                                                             |                    |      | + Σ |     |          |     |     |           | v         |          |           |
| C2Σ3                                                                            |                    |      |     |     |          |     |     |           |           |          |           |
| 2                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| 2C                                                                              |                    |      |     |     |          |     |     |           |           |          |           |
| I)−                                                                             |                    |      |     |     |          |     |     |           |           |          |           |
| v(cid:62) (C                                                                    |                    |      |     |     |          |     |     |           |           |          |           |
| 1 v + v(cid:62)Σ1/2                                                             |                    |      |     |     |          |     |     |           |           |          |           |
| v                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| (0,I)                                                                           |                    |      |     |     |          |     |     |           |           |          |           |
| ·                                                                               | ∗                  |      |     | ∗   |          |     |     | ∗         |           |          |           |
| ·                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| ∗ −                                                                             |                    |      |     |     |          |     |     |           |           |          |           |
| ·                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| ∗                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| ∗ −                                                                             |                    |      |     |     |          |     |     |           |           |          |           |
| ∼N                                                                              |                    |      |     |     |          |     |     |           |           |          |           |
| (cid:20)                                                                        |                    |      |     |     |          |     |     |           |           | (cid:21) |           |
| (cid:16)                                                                        |                    |      |     |     | (cid:17) |     |     |           |           |          |           |
| Σ                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| Σ                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| = E                                                                             |                    |      |     |     |          |     |     |           |           |          |           |
| 2 v                                                                             |                    |      |     |     |          |     |     |           |           |          |           |
| 2                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| I)−                                                                             |                    |      |     |     |          |     |     |           |           |          |           |
| I)−                                                                             |                    |      |     |     |          |     |     |           |           |          |           |
| v(cid:62) (C                                                                    |                    |      |     |     |          |     |     |           |           |          |           |
| 1 v + v(cid:62) (C                                                              |                    |      |     |     |          |     |     |           |           |          |           |
| v                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| (0,I)                                                                           |                    |      |     |     |          |     |     |           |           |          |           |
| ·                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| ·                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| ∗ −                                                                             |                    |      |     |     |          |     |     |           |           |          |           |
| ·                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| ∗ −                                                                             |                    |      |     |     |          |     |     |           |           |          |           |
| ∼N                                                                              |                    |      |     |     |          |     |     |           |           |          |           |
| (cid:104)                                                                       |                    |      |     |     |          |     |     |           |           |          |           |
| (cid:105)                                                                       |                    |      |     |     |          |     |     |           |           |          |           |
| Σ                                                                               |                    | Σ    |     |     |          |     |     |           |           |          |           |
| 22                                                                              | 1 v + v(cid:62) (C |      |     |     |          | I)− |     | 2 v       |           |          |           |
| = E                                                                             |                    |      |     |     |          |     |     |           |           |          |           |
| v                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| I)−                                                                             |                    |      |     |     |          |     |     |           |           |          |           |
| + v(cid:62) Iv + 2                                                              |                    |      |     |     |          |     |     |           |           |          |           |
| v(cid:62) (C                                                                    |                    |      |     |     |          |     |     |           |           |          |           |
| v                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| (0,I)                                                                           |                    |      |     |     |          |     |     |           |           |          |           |
| −(cid:107)                                                                      | ·                  |      | ∗ − |     |          |     |     |           |           |          |           |
| (cid:107)                                                                       |                    |      |     |     |          |     |     |           |           |          |           |
| ·                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| ·                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| ∗ −                                                                             |                    |      |     |     |          |     |     |           |           |          |           |
| ∼N                                                                              |                    |      |     |     |          |     |     |           |           |          |           |
| (cid:104)                                                                       |                    |      |     |     |          |     |     | (cid:105) |           |          |           |
| 2                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| 1                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| Σ                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| = E                                                                             |                    |      |     |     |          |     |     |           |           |          |           |
| 22                                                                              |                    |      |     |     |          |     |     |           |           |          |           |
| v                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| I + (C                                                                          |                    |      |     |     |          |     |     |           |           |          |           |
| v                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| I)−                                                                             |                    |      |     |     |          |     |     |           |           |          |           |
| + v(cid:62)                                                                     |                    |      |     |     |          |     |     |           |           |          |           |
| v                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| (0,I)                                                                           |                    |      |     |     |          |     |     |           |           |          |           |
| ·                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| ∗ −                                                                             |                    |      |     |     |          |     |     |           |           |          |           |
| −(cid:107)                                                                      |                    |      |     |     |          |     |     |           |           |          |           |
| (cid:107)                                                                       |                    |      |     |     |          |     |     |           |           |          |           |
| ∼N                                                                              |                    |      |     |     |          |     |     |           |           |          |           |
| (cid:20)                                                                        |                    |      |     |     |          |     |     |           |           |          |           |
| (cid:21)                                                                        |                    |      |     |     |          |     |     |           |           |          |           |
| (cid:16)                                                                        |                    |      |     |     |          |     |     |           |           |          |           |
| (cid:17)                                                                        |                    |      |     |     |          |     |     |           |           |          |           |
| 2                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| 1                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| Σ                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| I + (C                                                                          |                    |      |     |     |          |     |     |           |           |          |           |
| d                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| = tr                                                                            |                    |      |     |     |          |     |     |           |           |          |           |
| I)−                                                                             |                    |      |     |     |          |     |     |           |           |          |           |
| ·                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| ∗ −                                                                             |                    |      |     |     |          |     |     |           |           |          |           |
| −                                                                               |                    |      |     |     |          |     |     |           |           |          |           |
| (cid:21)                                                                        |                    |      |     |     |          |     |     |           |           |          |           |
| (cid:20)(cid:16)                                                                |                    |      |     |     |          |     |     |           |           |          |           |
| (cid:17)                                                                        |                    |      |     |     |          |     |     |           |           |          |           |



## Page 31

a minimax problem, due to the presence of the expectation between the outer minimization and the inner
maximization. Motivated by this and (19), we deﬁne the following robust problem:
min
µ,Σ max
M∈M Ex∼N (µ∗,Σ∗) [ℓ(µ, Σ; x + Mv)] ,
(20)
where M =
n
M ∈Rd×d : Mij = 0 ∀i ̸= j, Ex∼N (µ∗,Σ∗)
h
∥Mv∥2
2
i
= ϵ2o
.
First, note that this objective is slightly different from that of (15). In the motivating example, δ is con-
strained to always have ε-norm, and thus is normalizer on a per-sample basis inside of the expectation. In
contrast, here the classiﬁer is concerned with being robust to perturbations that are linear in v, and of ε2
squared norm in expectation.
Note, however, that via the result of Laurent and Massart [LM00] showing strong concentration for the
norms of Gaussian random variables, in high dimensions this bound on expectation has a corresponding
high-probability bound on the norm. In particular, this implies that as d →∞, ∥Mv∥2 = ε almost surely,
and thus the problem becomes identical to that of (15). We now derive the optimal M for a given (µ, Σ):
Lemma 2. Consider the minimax problem described by (20), i.e.
min
µ,Σ max
M∈M Ex∼N (µ∗,Σ∗) [ℓ(µ, Σ; x + Mv)] .
Then, the optimal action M∗of the inner maximization problem is given by
M = (λΣ −I)−1 ,
(21)
where again λ is set so that M ∈M.
Proof. We accomplish this in a similar fashion to what was done for δ∗, using Lagrange multipliers:
∇MEx∼N (µ∗,Σ∗)

v⊤MΣ−1v + 1
2v⊤MΣ−1Mv

= λ∇MEx∼N (µ∗,Σ∗)
h
∥Mv∥2
2 −ε2i
Ex∼N (µ∗,Σ∗)
h
Σ−1vv⊤+ Σ−1Mvv⊤i
= Ex∼N (µ∗,Σ∗)
h
λMvv⊤i
Σ−1Σ∗+ Σ−1MΣ∗= λMΣ∗
M = (λΣ −I)−1 ,
where λ is a constant depending on Σ and µ enforcing the expected squared-norm constraint.
Indeed, note that the optimal M for the adversary takes a near-identical form to the optimal δ (19), with the
exception that λ is not sample-dependent but rather varies only with the parameters.
E.3.4
Danskin’s Theorem
The main tool in proving our key results is Danskin’s Theorem [Dan67], a powerful theorem from minimax
optimization which contains the following key result:
Theorem 4 (Danskin’s Theorem). Suppose φ(x, z) : R × Z →R is a continuous function of two arguments,
where Z ⊂Rm is compact. Deﬁne f (x) = maxz∈Z φ(x, z). Then, if for every z ∈Z, φ(x, z) is convex and
differentiable in x, and ∂φ
∂x is continuous:
The subdifferential of f (x) is given by
∂f (x) = conv
∂φ(x, z)
∂x
: z ∈Z0(x)

,
where conv(·) represents the convex hull operation, and Z0 is the set of maximizers deﬁned as
Z0(x) =

z : φ(x, z) = max
z∈Z φ(x, z)

.
31


**Table 36 from page 31**

| 0                                                                                                      | 1                                                                                                            | 2                                                                |
|:-------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------|
| a minimax problem, due to the presence of the expectation between the outer minimization and the inner |                                                                                                              |                                                                  |
| maximization. Motivated by this and (19), we deﬁne the following robust problem:                       |                                                                                                              |                                                                  |
|                                                                                                        | E                                                                                                            |                                                                  |
|                                                                                                        | min                                                                                                          | (20)                                                             |
|                                                                                                        | max                                                                                                          |                                                                  |
|                                                                                                        | ∗) [(cid:96)(µ, Σ; x + Mv)] ,                                                                                |                                                                  |
|                                                                                                        | x                                                                                                            |                                                                  |
|                                                                                                        | (µ∗,Σ                                                                                                        |                                                                  |
|                                                                                                        | µ,Σ                                                                                                          |                                                                  |
|                                                                                                        | M                                                                                                            |                                                                  |
|                                                                                                        | ∼N                                                                                                           |                                                                  |
|                                                                                                        | ∈M                                                                                                           |                                                                  |
|                                                                                                        | Rd                                                                                                           |                                                                  |
|                                                                                                        | =                                                                                                            | 22                                                               |
|                                                                                                        | M                                                                                                            | Mv                                                               |
|                                                                                                        | i                                                                                                            | = (cid:101)2                                                     |
|                                                                                                        | where                                                                                                        | .                                                                |
|                                                                                                        | = j, E                                                                                                       |                                                                  |
|                                                                                                        | d : Mij = 0                                                                                                  |                                                                  |
|                                                                                                        | ×                                                                                                            | x                                                                |
|                                                                                                        |                                                                                                              | (µ∗,Σ                                                            |
|                                                                                                        |                                                                                                              | ∗)                                                               |
|                                                                                                        | M                                                                                                            | (cid:107)                                                        |
|                                                                                                        | ∈                                                                                                            | (cid:107)                                                        |
|                                                                                                        | ∀                                                                                                            | ∼N                                                               |
|                                                                                                        | (cid:54)                                                                                                     |                                                                  |
|                                                                                                        | (cid:110)                                                                                                    | (cid:104)                                                        |
|                                                                                                        |                                                                                                              | (cid:105)                                                        |
|                                                                                                        |                                                                                                              | (cid:111)                                                        |
| First, note that                                                                                       | this objective is slightly different                                                                         | (15).                                                            |
|                                                                                                        | from that of                                                                                                 | In the motivating example, δ is con-                             |
|                                                                                                        | strained to always have ε-norm, and thus is normalizer on a per-sample basis inside of the expectation.      | In                                                               |
|                                                                                                        | contrast, here the classiﬁer is concerned with being robust to perturbations that are linear in v, and of ε2 |                                                                  |
| squared norm in expectation.                                                                           |                                                                                                              |                                                                  |
| Note, however,                                                                                         | that via the result of Laurent and Massart                                                                   | [LM00] showing strong concentration for the                      |
| norms of Gaussian random variables,                                                                    |                                                                                                              | in high dimensions this bound on expectation has a corresponding |
| high-probability bound on the norm.                                                                    | In particular, this implies that as d                                                                        | Mv                                                               |
|                                                                                                        |                                                                                                              | ∞,                                                               |
|                                                                                                        |                                                                                                              | →                                                                |
|                                                                                                        |                                                                                                              | (cid:107)                                                        |
|                                                                                                        |                                                                                                              | (cid:107)2 = ε almost surely,                                    |
|                                                                                                        | and thus the problem becomes identical to that of (15). We now derive the optimal M for a given (µ, Σ):      |                                                                  |



## Page 32

In short, given a minimax problem of the form minx maxy∈C f (x, y) where C is a compact set, if f (·, y) is
convex for all values of y, then rather than compute the gradient of g(x) := maxy∈C f (x, y), we can simply
ﬁnd a maximizer y∗for the current parameter x; Theorem 4 ensures that ∇x f (x, y∗) ∈∂xg(x). Note that M
is trivially compact (by the Heine-Borel theorem), and differentiability/continuity follow rather straight-
forwardly from our reparameterization (c.f. (22)), and so it remains to show that the outer minimization is
convex for any ﬁxed M.
Convexity of the outer minimization.
Note that even in the standard case (i.e. non-adversarial), the
Gaussian negative log-likelihood is not convex with respect to (µ, Σ). Thus, rather than proving convexity
of this function directly, we employ the parameterization used by [Das+19]: in particular, we write the
problem in terms of T = Σ−1 and m = Σ−1µ. Under this parameterization, we show that the robust
problem is convex for any ﬁxed M.
Lemma 3. Under the aforementioned parameterization of T = Σ−1 and m = Σ−1µ, the following “Gaussian robust
negative log-likelihood” is convex:
Ex∼N (µ∗,Σ∗) [ℓ(m, T; x + Mv)] .
Proof. To prove this, we show that the likelihood is convex even with respect to a single sample x; the
result follows, since a convex combination of convex functions remains convex. We begin by looking at the
likelihood of a single sample x ∼N (µ∗, Σ∗):
L(µ, Σ; x + M(x −µ)) =
1
p
(2π)k|Σ|
exp

−1
2(x −µ)⊤(I + M)2Σ−1(x −µ)

=
1
√
(2π)k|Σ| exp

−1
2(x −µ)⊤(I + M)2Σ−1(x −µ)

Z
1
√
(2π)k|(I+M)−2Σ| exp

−1
2(x −µ)⊤(I + M)2Σ−1(x −µ)

=
|I + M|−1 exp

−1
2x⊤(I + M)2Σ−1x + µ⊤(I + M)2Σ−1x

Z
exp

−1
2x⊤(I + M)2Σ−1x + µ⊤(I + M)2Σ−1x

In terms of the aforementioned T and m, and for convenience deﬁning A = (I + M)2:
ℓ(x) = |A|−1/2 +
1
2x⊤ATx −m⊤Ax

−log
Z
exp
1
2x⊤ATx −m⊤Ax

∇ℓ(x) =
 1
2(Axx⊤)
−Ax

−
Z  1
2(Axx⊤)
−Ax

exp

1
2x⊤ATx −m⊤Ax

Z
exp

1
2x⊤ATx −m⊤Ax

=
 1
2(Axx⊤)
−Ax

−Ez∼N (T−1m,(AT)−1)
 1
2(Azz⊤)
−Az

.
(22)
From here, following an identical argument to [Das+19] Equation (3.7), we ﬁnd that
Hℓ= Covz∼N(T−1m,(AT)−1)
" 
−1
2 AzzT

z
!
,
 
−1
2 AzzT

z
!#
≽0,
i.e. that the log-likelihood is indeed convex with respect to

T
m

, as desired.
32


**Table 37 from page 32**

| 0                                                                                                           |
|:------------------------------------------------------------------------------------------------------------|
| f (                                                                                                         |
| , y) is                                                                                                     |
| In short, given a minimax problem of the form minx maxy                                                     |
| C f (x, y) where C is a compact set, if                                                                     |
| ∈                                                                                                           |
| ·                                                                                                           |
| convex for all values of y, then rather than compute the gradient of g(x) := maxy                           |
| C f (x, y), we can simply                                                                                   |
| ∈                                                                                                           |
| ∂x g(x). Note that                                                                                          |
| ﬁnd a maximizer y∗ for the current parameter x; Theorem 4 ensures that                                      |
| ∇x f (x, y∗)                                                                                                |
| ∈                                                                                                           |
| M                                                                                                           |
| is trivially compact (by the Heine-Borel                                                                    |
| theorem), and differentiability/continuity follow rather straight-                                          |
| forwardly from our reparameterization (c.f. (22)), and so it remains to show that the outer minimization is |
| convex for any ﬁxed M.                                                                                      |
| Convexity of                                                                                                |
| the outer minimization.                                                                                     |
| Note that even in the standard case (i.e.                                                                   |
| non-adversarial),                                                                                           |
| the                                                                                                         |
| Gaussian negative log-likelihood is not convex with respect to (µ, Σ). Thus, rather than proving convexity  |
| of                                                                                                          |
| this function directly, we employ the parameterization used by [Das+19]:                                    |
| in particular, we write the                                                                                 |
| problem in terms of T = Σ                                                                                   |
| 1 and m = Σ                                                                                                 |
| 1µ. Under this parameterization, we show that                                                               |
| the robust                                                                                                  |
| −                                                                                                           |
| −                                                                                                           |
| problem is convex for any ﬁxed M.                                                                           |
| 1 and m = Σ                                                                                                 |
| Lemma 3. Under the aforementioned parameterization of T = Σ                                                 |
| 1µ, the following “Gaussian robust                                                                          |
| −                                                                                                           |
| −                                                                                                           |
| negative log-likelihood” is convex:                                                                         |
| E                                                                                                           |
| x                                                                                                           |
| (µ∗,Σ                                                                                                       |
| ∗) [(cid:96)(m, T; x + Mv)] .                                                                               |
| ∼N                                                                                                          |
| Proof. To prove this, we show that                                                                          |
| the likelihood is convex even with respect                                                                  |
| to a single sample x;                                                                                       |
| the                                                                                                         |
| result follows, since a convex combination of convex functions remains convex. We begin by looking at the   |
| (µ                                                                                                          |
| likelihood of a single sample x                                                                             |
| , Σ                                                                                                         |
| ):                                                                                                          |
| ∗                                                                                                           |
| ∗                                                                                                           |
| ∼ N                                                                                                         |



## Page 33

E.3.5
Applying Danskin’s Theorem
The previous two parts show that we can indeed apply Danskin’s theorem to the outer minimization, and
in particular that the gradient of f at M = M∗is in the subdifferential of the outer minimization problem.
We proceed by writing out this gradient explicitly, and then setting it to zero (note that since we have shown
f is convex for all choices of perturbation, we can use the fact that a convex function is globally minimized
⇐⇒its subgradient contains zero). We continue from above, plugging in (21) for M and using (22) to write
the gradients of ℓwith respect to T and m.
0 = ∇"T
m
#ℓ= Ex∼N (µ∗,Σ∗)
 1
2(Axx⊤)
−Ax

−Ez∼N (T−1m,(AT)−1)
 1
2(Azz⊤)
−Az

= Ex∼N (µ∗,Σ∗)
 1
2(Axx⊤)
−Ax

−Ez∼N (T−1m,(AT)−1)
 1
2(Azz⊤)
−Az

=
 1
2(AΣ∗)
−Aµ∗

−Ez∼N (T−1m,(AT)−1)
 1
2(A(AT)−1)
−AT−1m

=
 1
2 AΣ∗
−Aµ∗

−
 1
2 A(AT)−1
−AT−1m

=
 1
2 AΣ∗−1
2T−1
AT−1m −Aµ∗

(23)
Using this fact, we derive an implicit expression for the robust covariance matrix Σ. Note that for the
sake of brevity, we now use M to denote the optimal adversarial perturbation (previously deﬁned as M∗
in (21)). This implicit formulation forms the foundation of the bounds given by our main results.
Lemma 4. The minimax problem discussed throughout this work admits the following (implicit) form of solution:
Σ = 1
λ I + 1
2Σ∗+
r
1
λΣ∗+ 1
4Σ2∗,
where λ is such that M ∈M, and is thus dependent on Σ.
Proof. Rewriting (23) in the standard parameterization (with respect to µ, Σ) and re-expanding A = (I +
M)2 yields:
0 = ∇"T
m
#ℓ=

1
2(I + M)2Σ∗−1
2Σ
(I + M)2µ −(I + M)2µ∗

Now, note that the equations involving µ and Σ are completely independent, and thus can be solved
separately. In terms of µ, the relevant system of equations is Aµ −Aµ∗= 0, where multiplying by the
inverse A gives that
µ = µ∗.
(24)
This tells us that the mean learned via ℓ2-robust maximum likelihood estimation is precisely the true mean
of the distribution.
Now, in the same way, we set out to ﬁnd Σ by solving the relevant system of equations:
Σ−1
∗
= Σ−1(M + I)2.
(25)
Now, we make use of the Woodbury Matrix Identity in order to write (I + M) as
I + (λΣ −I)−1 = I +
 
−I −
 1
λΣ−1 −I
−1!
= −
 1
λΣ−1 −I
−1
.
33


**Table 38 from page 33**

| 0                                                                                                              |
|:---------------------------------------------------------------------------------------------------------------|
| E.3.5                                                                                                          |
| Applying Danskin’s Theorem                                                                                     |
| The previous two parts show that we can indeed apply Danskin’s theorem to the outer minimization, and          |
| in particular that the gradient of                                                                             |
| f at M = M∗ is in the subdifferential of the outer minimization problem.                                       |
| We proceed by writing out this gradient explicitly, and then setting it to zero (note that since we have shown |
| f                                                                                                              |
| is convex for all choices of perturbation, we can use the fact that a convex function is globally minimized    |
| its subgradient contains zero). We continue from above, plugging in (21) for M and using (22) to write         |
| ⇐⇒                                                                                                             |
| the gradients of (cid:96) with respect to T and m.                                                             |



## Page 34

Thus, we can revisit (25) as follows:
Σ−1
∗
= Σ−1
 1
λΣ−1 −I
−2
1
λ2 Σ−1
∗Σ−2 −
 2
λΣ−1
∗
+ I

Σ−1 + Σ−1
∗
= 0
1
λ2 Σ−1
∗
−
 2
λΣ−1
∗
+ I

Σ + Σ−1
∗Σ2 = 0
We now apply the quadratic formula to get an implicit expression for Σ (implicit since technically λ
depends on Σ):
Σ =
 
2
λΣ−1
∗
+ I ±
r
4
λΣ−1
∗
+ I
!
1
2Σ∗
= 1
λ I + 1
2Σ∗+
r
1
λΣ∗+ 1
4Σ2∗.
(26)
This concludes the proof.
E.3.6
Bounding λ
We now attempt to characterize the shape of λ as a function of ε. First, we use the fact that E[∥Xv∥2] =
tr(X2) for standard normally-drawn v. Thus, λ is set such that tr(Σ∗M2) = ε, i.e:
∑
i=0
Σ∗
ii
(λΣii −1)2 = ε
(27)
Now, consider ε2 as a function of λ. Observe that for λ ≥
1
σmin(Σ), we have that M must be positive semi-
deﬁnite, and thus ε2 decays smoothly from ∞(at λ =
1
σmin ) to zero (at λ = ∞). Similarly, for λ ≤
1
σmax(Σ),
ε decays smoothly as λ decreases. Note, however, that such values of λ would necessarily make M negative
semi-deﬁnite, which would actually help the log-likelihood. Thus, we can exclude this case; in particular, for
the remainder of the proofs, we can assume λ ≥
1
σmax(Σ).
Also observe that the zeros of ε in terms of λ are only at λ = ±∞. Using this, we can show that there
exists some ε0 for which, for all ε < ε0, the only corresponding possible valid value of λ is where λ ≥
1
σmin .
This idea is formalized in the following Lemma.
Lemma 5. For every Σ∗, there exists some ε0 > 0 for which, for all ε ∈[0, ε0) the only admissible value of λ is such
that λ ≥
1
σmin(Σ), and thus such that M is positive semi-deﬁnite.
Proof. We prove the existence of such an ε0 by lower bounding ε (in terms of λ) for any ﬁnite λ > 0 that
does not make M PSD. Providing such a lower bound shows that for small enough ε (in particular, less than
this lower bound), the only corresponding values of λ are as desired in the statement19.
In particular, if M is not PSD, then there must exist at least one index k such that λΣkk < 1, and thus
(λΣkk −1)2 ≤1 for all λ > 0. We can thus lower bound (27) as:
ε = ∑
i=0
Σ∗
ii
(λΣii −1)2 ≥
Σ∗
kk
(λΣkk −1)2 ≥Σ∗
kk ≥σmin(Σ∗) > 0
(28)
By contradiction, it follows that for any ε < σmin(Σ∗)2, the only admissible λ is such that M is PSD, i.e.
according to the statement of the Lemma.
19Since our only goal is existence, we lose many factors from the analysis that would give a tighter bound on ε0.
34


**Table 39 from page 34**

| 0                                                                                                                 |
|:------------------------------------------------------------------------------------------------------------------|
| Thus, we can revisit (25) as follows:                                                                             |
| 2                                                                                                                 |
| −                                                                                                                 |
| 1                                                                                                                 |
| 1                                                                                                                 |
| 1                                                                                                                 |
| 1 λ                                                                                                               |
| I                                                                                                                 |
| Σ−                                                                                                                |
| = Σ−                                                                                                              |
| Σ−                                                                                                                |
| −                                                                                                                 |
| ∗                                                                                                                 |
| (cid:18)                                                                                                          |
| (cid:19)                                                                                                          |
| 1                                                                                                                 |
| 2                                                                                                                 |
| 1                                                                                                                 |
| 1                                                                                                                 |
| 1 λ                                                                                                               |
| 2 λ                                                                                                               |
| + I                                                                                                               |
| = 0                                                                                                               |
| Σ−                                                                                                                |
| Σ−                                                                                                                |
| Σ−                                                                                                                |
| Σ−                                                                                                                |
| 1 + Σ−                                                                                                            |
| 2                                                                                                                 |
| −                                                                                                                 |
| ∗                                                                                                                 |
| ∗                                                                                                                 |
| ∗                                                                                                                 |
| (cid:18)                                                                                                          |
| (cid:19)                                                                                                          |
| 1                                                                                                                 |
| 1                                                                                                                 |
| 1                                                                                                                 |
| 1 λ                                                                                                               |
| 2 λ                                                                                                               |
| + I                                                                                                               |
| Σ2 = 0                                                                                                            |
| Σ−                                                                                                                |
| Σ−                                                                                                                |
| Σ + Σ−                                                                                                            |
| 2                                                                                                                 |
| −                                                                                                                 |
| ∗                                                                                                                 |
| ∗                                                                                                                 |
| ∗                                                                                                                 |
| (cid:18)                                                                                                          |
| (cid:19)                                                                                                          |
| We now apply the quadratic formula to get an implicit expression for Σ (implicit since technically λ              |
| depends on Σ):                                                                                                    |
| 1                                                                                                                 |
| 1                                                                                                                 |
| Σ                                                                                                                 |
| 2 λ                                                                                                               |
| 4 λ                                                                                                               |
| 1 2                                                                                                               |
| Σ =                                                                                                               |
| + I                                                                                                               |
| + I                                                                                                               |
| Σ−                                                                                                                |
| Σ−                                                                                                                |
| ∗                                                                                                                 |
| ∗                                                                                                                 |
| ∗                                                                                                                 |
| (cid:33)                                                                                                          |
| (cid:32)                                                                                                          |
| ± (cid:114)                                                                                                       |
| Σ                                                                                                                 |
| Σ                                                                                                                 |
| Σ2                                                                                                                |
| 1 λ                                                                                                               |
| 1 2                                                                                                               |
| 1 λ                                                                                                               |
| 1 4                                                                                                               |
| =                                                                                                                 |
| +                                                                                                                 |
| +                                                                                                                 |
| I +                                                                                                               |
| .                                                                                                                 |
| (26)                                                                                                              |
| ∗                                                                                                                 |
| ∗                                                                                                                 |
| ∗                                                                                                                 |
| (cid:114)                                                                                                         |
| This concludes the proof.                                                                                         |
| E.3.6                                                                                                             |
| Bounding λ                                                                                                        |
| 2] =                                                                                                              |
| Xv                                                                                                                |
| We now attempt to characterize the shape of λ as a function of ε. First, we use the fact that E[                  |
| (cid:107)                                                                                                         |
| (cid:107)                                                                                                         |
| tr(X2) for standard normally-drawn v. Thus, λ is set such that tr(Σ                                               |
| M2) = ε, i.e:                                                                                                     |
| ∗                                                                                                                 |
| Σ                                                                                                                 |
| ∗                                                                                                                 |
| ii                                                                                                                |
| ∑ i                                                                                                               |
| = ε                                                                                                               |
| (27)                                                                                                              |
| (λΣ                                                                                                               |
| 1)2                                                                                                               |
| =0                                                                                                                |
| ii −                                                                                                              |
| 1                                                                                                                 |
| Now, consider ε2 as a function of λ. Observe that for λ                                                           |
| σmin(Σ) , we have that M must be positive semi-                                                                   |
| ≥                                                                                                                 |
| 1                                                                                                                 |
| 1                                                                                                                 |
| deﬁnite, and thus ε2 decays smoothly from ∞ (at λ =                                                               |
| ) to zero (at λ = ∞). Similarly, for λ                                                                            |
| σmin                                                                                                              |
| σmax(Σ) ,                                                                                                         |
| ≤                                                                                                                 |
| ε decays smoothly as λ decreases. Note, however, that such values of λ would necessarily make M negative          |
| semi-deﬁnite, which would actually help the log-likelihood. Thus, we can exclude this case; in particular, for    |
| 1                                                                                                                 |
| the remainder of the proofs, we can assume λ                                                                      |
| σmax(Σ) .                                                                                                         |
| ≥                                                                                                                 |
| Also observe that the zeros of ε in terms of λ are only at λ =                                                    |
| ∞. Using this, we can show that there                                                                             |
| ±                                                                                                                 |
| 1                                                                                                                 |
| .                                                                                                                 |
| exists some ε0 for which, for all ε < ε0, the only corresponding possible valid value of λ is where λ             |
| σmin                                                                                                              |
| ≥                                                                                                                 |
| This idea is formalized in the following Lemma.                                                                   |
| Lemma 5. For every Σ                                                                                              |
| , there exists some ε0 > 0 for which, for all ε                                                                   |
| [0, ε0) the only admissible value of λ is such                                                                    |
| ∗                                                                                                                 |
| ∈                                                                                                                 |
| 1                                                                                                                 |
| that λ                                                                                                            |
| σmin(Σ) , and thus such that M is positive semi-deﬁnite.                                                          |
| ≥                                                                                                                 |
| Proof. We prove the existence of such an ε0 by lower bounding ε (in terms of λ) for any ﬁnite λ > 0 that          |
| does not make M PSD. Providing such a lower bound shows that for small enough ε (in particular, less than         |
| this lower bound), the only corresponding values of λ are as desired in the statement19.                          |
| In particular,                                                                                                    |
| if M is not PSD, then there must exist at least one index k such that λΣ                                          |
| kk < 1, and thus                                                                                                  |
| (λΣ                                                                                                               |
| 1)2                                                                                                               |
| 1 for all λ > 0. We can thus lower bound (27) as:                                                                 |
| kk −                                                                                                              |
| ≤                                                                                                                 |
| Σ                                                                                                                 |
| Σ                                                                                                                 |
| ∗                                                                                                                 |
| ∗                                                                                                                 |
| kk                                                                                                                |
| ii                                                                                                                |
| ε = ∑                                                                                                             |
| (28)                                                                                                              |
| σmin(Σ∗) > 0                                                                                                      |
| (λΣ                                                                                                               |
| (λΣ                                                                                                               |
| Σ∗kk ≥                                                                                                            |
| 1)2 ≥                                                                                                             |
| 1)2 ≥                                                                                                             |
| i=0                                                                                                               |
| ii −                                                                                                              |
| kk −                                                                                                              |
| By contradiction,                                                                                                 |
| it                                                                                                                |
| follows that                                                                                                      |
| )2,                                                                                                               |
| the only admissible λ is such that M is PSD,                                                                      |
| i.e.                                                                                                              |
| for any ε < σmin(Σ                                                                                                |
| ∗                                                                                                                 |
| according to the statement of the Lemma.                                                                          |
| 19Since our only goal is existence, we lose many factors from the analysis that would give a tighter bound on ε0. |
| 34                                                                                                                |



## Page 35

In the regime ε ∈[0, ε0), note that λ is inversely proportional to ε (i.e. as ε grows, λ decreases). This
allows us to get a qualitative view of (26): as the allowed perturbation value increases, the robust covariance
Σ resembles the identity matrix more and more, and thus assigns more and more variance on initially low-
variance features. The √Σ∗term indicates that the robust model also adds uncertainty proportional to the
square root of the initial variance—thus, low-variance features will have (relatively) more uncertainty in the
robust case. Indeed, our main result actually follows as a (somewhat loose) formalization of this intuition.
E.3.7
Proof of main theorems
First, we give a proof of Theorem 2, providing lower and upper bounds on the learned robust covariance Σ
in the regime ε ∈[0, ε0).
Theorem 2 (Robustly Learned Parameters). Just as in the non-robust case, µr = µ∗, i.e. the true mean is learned.
For the robust covariance Σr, there exists an ε0 > 0, such that for any ε ∈[0, ε0),
Σr = 1
2Σ∗+ 1
λ · I +
r
1
λ · Σ∗+ 1
4Σ2∗,
where
Ω
 
1 + ε1/2
ε1/2 + ε3/2
!
≤λ ≤O
 
1 + ε1/2
ε1/2
!
.
Proof. We have already shown that µ = µ∗in the robust case (c.f. (24)). We choose ε0 to be as described,
i.e. the largest ε for which the set {λ : tr(Σ2∗M) = ε, λ ≥1/σmax(Σ)} has only one element λ (which, as we
argued, must not be less than 1/σmin(Σ)). We have argued that such an ε0 must exist.
We prove the result by combining our early derivation (in particular, (25) and (26)) with upper and lower
bound on λ, which we can compute based on properties of the trace operator. We begin by deriving a lower
bound on λ. By linear algebraic manipulation (given in Appendix E.3.8), we get the following bound:
λ ≥
d
tr(Σ)
 
1 +
r
d · σmin(Σ∗)
ε
!
(29)
Now, we can use (25) in order to remove the dependency of λ on Σ:
Σ = Σ∗(M + I)2
tr(Σ) = tr
h
(Σ1/2
∗
M + Σ1/2
∗
)2i
≤2 · tr
h
(Σ1/2
∗
M)2 + (Σ1/2
∗
)2i
≤2 · (ε + tr(Σ∗)) .
Applying this to (29) yields:
λ ≥
d/2
ε + tr(Σ∗)
 
1 +
r
d · σmin(Σ∗)
ε
!
.
Note that we can simplify this bound signiﬁcantly by writing ε = d · σmin(Σ∗)ε′ ≤tr(Σ∗)ε′, which does not
affect the result (beyond rescaling the valid regime (0, ε0)), and gives:
λ ≥
d/2
(1 + ε′)tr(Σ∗)

1 +
1
√
ε′

≥
d · (1 +
√
ε′)
2
√
ε′(1 + ε′)tr(Σ∗)
Next, we follow a similar methodology (Appendix E.3.8) in order to upper bound λ:
λ ≤
1
σmin(Σ)
 r
∥Σ∗∥F · d
ε
+ 1
!
.
Note that by (25) and positive semi-deﬁniteness of M, it must be that σmin(Σ) ≥σmin(Σ∗). Thus, we can
simplify the previous expression, also substituting ε = d · σmin(Σ∗)ε′:
λ ≤
1
σmin(Σ∗)
 s
∥Σ∗∥F
σmin(Σ∗)ε′ + 1
!
= ∥Σ∗∥F +
p
ε · σmin(Σ∗)
σmin(Σ∗)3/2√ε
35


**Table 40 from page 35**

| 0                                                                                                               |
|:----------------------------------------------------------------------------------------------------------------|
| In the regime ε                                                                                                 |
| as ε grows, λ decreases). This                                                                                  |
| [0, ε0), note that λ is inversely proportional to ε (i.e.                                                       |
| ∈                                                                                                               |
| allows us to get a qualitative view of (26): as the allowed perturbation value increases, the robust covariance |
| Σ resembles the identity matrix more and more, and thus assigns more and more variance on initially low-        |
| term indicates that the robust model also adds uncertainty proportional to the                                  |
| variance features. The √Σ                                                                                       |
| ∗                                                                                                               |
| square root of the initial variance—thus, low-variance features will have (relatively) more uncertainty in the  |
| robust case. Indeed, our main result actually follows as a (somewhat loose) formalization of this intuition.    |
| E.3.7                                                                                                           |
| Proof of main theorems                                                                                          |
| First, we give a proof of Theorem 2, providing lower and upper bounds on the learned robust covariance Σ        |
| in the regime ε                                                                                                 |
| [0, ε0).                                                                                                        |
| ∈                                                                                                               |
| Theorem 2 (Robustly Learned Parameters).                                                                        |
| Just as in the non-robust case, µr = µ∗, i.e. the true mean is learned.                                         |
| For the robust covariance Σr, there exists an ε0 > 0, such that for any ε                                       |
| [0, ε0),                                                                                                        |
| ∈                                                                                                               |



## Page 36

These bounds can be straightforwardly combined with Lemma 4, which concludes the proof.
Using this theorem, we can now show Theorem 3:
Theorem 3 (Gradient alignment). Let f (x) and fr(x) be monotonic classiﬁers based on the linear separator induced
by standard and ℓ2-robust maximum likelihood classiﬁcation, respectively. The maximum angle formed between the
gradient of the classiﬁer (wrt input) and the vector connecting the classes can be smaller for the robust model:
min
µ
⟨µ, ∇x fr(x)⟩
∥µ∥· ∥∇x fr(x)∥> min
µ
⟨µ, ∇x f (x)⟩
∥µ∥· ∥∇x f (x)∥.
Proof. To prove this, we make use of the following Lemmas:
Lemma 6. For two positive deﬁnite matrices A and B with κ(A) > κ(B), we have that κ(A + B) ≤max{κ(A), κ(B)}.
Proof. We proceed by contradiction:
κ(A + B) = λmax(A) + λmax(B)
λmin(A) + λmin(B)
κ(A) = λmax(A)
λmin(A)
κ(A) ≥κ(A + B)
⇐⇒λmax(A) (λmin(A) + λmin(B)) ≥λmin(A) (λmax(A) + λmax(B))
⇐⇒λmax(A)λmin(B) ≥λmin(A)λmax(B)
⇐⇒λmax(A)
λmin(A) ≥λmin(A)
λmax(B),
which is false by assumption. This concludes the proof.
Lemma 7 (Straightforward). For a positive deﬁnite matrix A and k > 0, we have that
κ(A + k · I) < κ(A)
κ(A + k ·
√
A) ≤κ(A).
Lemma 8 (Angle induced by positive deﬁnite matrix; folklore). 20 For a positive deﬁnite matrix A ≻0 with
condition number κ, we have that
min
x
x⊤Ax
∥Ax∥2 · ∥x∥2
= 2√κ
1 + κ .
(30)
These two results can be combined to prove the theorem. First, we show that κ(Σ) ≤κ(Σ∗):
κ(Σ) = κ
 
1
λ I + 1
2Σ∗+
r
1
λΣ∗+ 1
4Σ2∗
!
< max
(
κ
 1
λ I + 1
2Σ∗

, κ
 r
1
λΣ∗+ 1
4Σ2∗
!)
< max
(
κ (Σ∗) ,
s
κ
 1
λΣ∗+ 1
4Σ2∗
)
= max


κ (Σ∗) ,
v
u
u
tκ
 
2
λ
r
1
4Σ2∗+ 1
4Σ2∗
!


≤κ (Σ∗) .
Finally, note that (30) is a strictly decreasing function in κ, and as such, we have shown the theorem.
20A proof can be found in https://bit.ly/2L6jdAT
36


**Table 41 from page 36**

| 0                                                                                                                     | 1          | 2   |
|:----------------------------------------------------------------------------------------------------------------------|:-----------|:----|
| These bounds can be straightforwardly combined with Lemma 4, which concludes the proof.                               |            |     |
| Using this theorem, we can now show Theorem 3:                                                                        |            |     |
| Theorem 3 (Gradient alignment). Let                                                                                   |            |     |
| f (x) and fr(x) be monotonic classiﬁers based on the linear separator induced                                         |            |     |
| by standard and (cid:96)2-robust maximum likelihood classiﬁcation, respectively. The maximum angle formed between the |            |     |
| gradient of the classiﬁer (wrt input) and the vector connecting the classes can be smaller for the robust model:      |            |     |
| µ,                                                                                                                    |            |     |
| µ,                                                                                                                    |            |     |
| > min                                                                                                                 |            |     |
| .                                                                                                                     |            |     |
| min                                                                                                                   |            |     |
| ∇x f (x)                                                                                                              |            |     |
| ∇x fr(x)                                                                                                              |            |     |
| (cid:104)                                                                                                             |            |     |
| (cid:105)                                                                                                             |            |     |
| (cid:104)                                                                                                             |            |     |
| (cid:105)                                                                                                             |            |     |
| µ                                                                                                                     |            |     |
| µ                                                                                                                     |            |     |
| µ                                                                                                                     |            |     |
| µ                                                                                                                     |            |     |
| (cid:107)                                                                                                             |            |     |
| (cid:107) · (cid:107)∇x fr(x)                                                                                         |            |     |
| (cid:107)                                                                                                             |            |     |
| (cid:107)                                                                                                             |            |     |
| (cid:107) · (cid:107)∇x f (x)                                                                                         |            |     |
| (cid:107)                                                                                                             |            |     |
| Proof. To prove this, we make use of the following Lemmas:                                                            |            |     |
| Lemma 6. For two positive deﬁnite matrices A and B with κ(A) > κ(B), we have that κ(A + B)                            | κ(A), κ(B) | .   |
| max                                                                                                                   |            |     |
| ≤                                                                                                                     | }          |     |
| {                                                                                                                     |            |     |
| Proof. We proceed by contradiction:                                                                                   |            |     |
| λmax(A) + λmax(B)                                                                                                     |            |     |
| κ(A + B) =                                                                                                            |            |     |
| λmin(A) + λmin(B)                                                                                                     |            |     |
| λmax(A)                                                                                                               |            |     |
| κ(A) =                                                                                                                |            |     |
| λmin(A)                                                                                                               |            |     |
| κ(A)                                                                                                                  |            |     |
| κ(A + B)                                                                                                              |            |     |
| ≥                                                                                                                     |            |     |
| λmax(A) (λmin(A) + λmin(B))                                                                                           |            |     |
| λmin(A) (λmax(A) + λmax(B))                                                                                           |            |     |
| ⇐⇒                                                                                                                    |            |     |
| ≥                                                                                                                     |            |     |
| λmax(A)λmin(B)                                                                                                        |            |     |
| λmin(A)λmax(B)                                                                                                        |            |     |
| ⇐⇒                                                                                                                    |            |     |
| ≥                                                                                                                     |            |     |
| λmax(A)                                                                                                               |            |     |
| λmin(A)                                                                                                               |            |     |
| ,                                                                                                                     |            |     |
| ⇐⇒                                                                                                                    |            |     |
| λmax(B)                                                                                                               |            |     |
| λmin(A) ≥                                                                                                             |            |     |
| which is false by assumption. This concludes the proof.                                                               |            |     |
| Lemma 7 (Straightforward). For a positive deﬁnite matrix A and k > 0, we have that                                    |            |     |
| √A)                                                                                                                   |            |     |
| κ(A + k                                                                                                               |            |     |
| I) < κ(A)                                                                                                             |            |     |
| κ(A + k                                                                                                               |            |     |
| κ(A).                                                                                                                 |            |     |
| ·                                                                                                                     |            |     |
| ·                                                                                                                     |            |     |
| ≤                                                                                                                     |            |     |
| 20 For a positive deﬁnite matrix A                                                                                    |            |     |
| Lemma 8 (Angle induced by positive deﬁnite matrix; folklore).                                                         |            |     |
| 0 with                                                                                                                |            |     |
| (cid:31)                                                                                                              |            |     |
| condition number κ, we have that                                                                                      |            |     |
| 2√κ                                                                                                                   |            |     |
| x(cid:62) Ax                                                                                                          |            |     |
| =                                                                                                                     |            |     |
| min                                                                                                                   |            |     |
| .                                                                                                                     |            |     |
| (30)                                                                                                                  |            |     |
| x                                                                                                                     |            |     |
| Ax                                                                                                                    |            |     |
| x                                                                                                                     |            |     |
| 1 + κ                                                                                                                 |            |     |
| (cid:107)                                                                                                             |            |     |
| (cid:107)2 · (cid:107)                                                                                                |            |     |
| (cid:107)2                                                                                                            |            |     |
| κ(Σ                                                                                                                   |            |     |
| These two results can be combined to prove the theorem. First, we show that κ(Σ)                                      |            |     |
| ):                                                                                                                    |            |     |
| ∗                                                                                                                     |            |     |
| ≤                                                                                                                     |            |     |
| Σ                                                                                                                     |            |     |
| Σ                                                                                                                     |            |     |
| Σ2                                                                                                                    |            |     |
| 1 λ                                                                                                                   |            |     |
| 1 2                                                                                                                   |            |     |
| 1 λ                                                                                                                   |            |     |
| 1 4                                                                                                                   |            |     |
| +                                                                                                                     |            |     |
| +                                                                                                                     |            |     |
| I +                                                                                                                   |            |     |
| κ(Σ) = κ                                                                                                              |            |     |
| ∗                                                                                                                     |            |     |
| ∗                                                                                                                     |            |     |
| ∗(cid:33)                                                                                                             |            |     |
| (cid:32)                                                                                                              |            |     |
| (cid:114)                                                                                                             |            |     |
| Σ                                                                                                                     |            |     |
| Σ                                                                                                                     |            |     |
| Σ2                                                                                                                    |            |     |
| 1 λ                                                                                                                   |            |     |
| 1 2                                                                                                                   |            |     |
| 1 λ                                                                                                                   |            |     |
| 1 4                                                                                                                   |            |     |
| +                                                                                                                     |            |     |
| I +                                                                                                                   |            |     |
| κ                                                                                                                     |            |     |
| < max                                                                                                                 |            |     |
| , κ                                                                                                                   |            |     |
| ∗                                                                                                                     |            |     |
| ∗                                                                                                                     |            |     |
| (cid:40)                                                                                                              |            |     |
| ∗(cid:33)(cid:41)                                                                                                     |            |     |
| (cid:32)(cid:114)                                                                                                     |            |     |
| (cid:18)                                                                                                              |            |     |
| (cid:19)                                                                                                              |            |     |
| Σ                                                                                                                     |            |     |
| Σ2                                                                                                                    |            |     |
| 1 λ                                                                                                                   |            |     |
| 1 4                                                                                                                   |            |     |
| +                                                                                                                     |            |     |
| κ (Σ                                                                                                                  |            |     |
| κ                                                                                                                     |            |     |
| < max                                                                                                                 |            |     |
| ) ,                                                                                                                   |            |     |
| ∗                                                                                                                     |            |     |
| ∗                                                                                                                     |            |     |
| (cid:115)                                                                                                             |            |     |
| (cid:40)                                                                                                              |            |     |
| (cid:18)                                                                                                              |            |     |
| ∗(cid:19)(cid:41)                                                                                                     |            |     |
| Σ2                                                                                                                    |            |     |
| Σ2                                                                                                                    |            |     |
| 2 λ                                                                                                                   |            |     |
| 1 4                                                                                                                   |            |     |
| 1 4                                                                                                                   |            |     |
| +                                                                                                                     |            |     |
| κ (Σ                                                                                                                  |            |     |
| κ                                                                                                                     |            |     |
| = max                                                                                                                 |            |     |
| ) ,                                                                                                                   |            |     |
| ∗                                                                                                                     |            |     |
| ∗                                                                                                                     |            |     |
| ∗(cid:33)                                                                                                            |            |     |
| (cid:118)(cid:117)(cid:117)(cid:116)                                                                                  |            |     |
| (cid:32)                                                                                                              |            |     |
| (cid:114)                                                                                                             |            |     |
|                                                                                                                    |            |     |
|                                                                                                                     |            |     |
| κ (Σ                                                                                                                  |            |     |
| ) .                                                                                                                   |            |     |
| ∗                                                                                                                     |            |     |
| ≤                                                                                                                     |            |     |
| Finally, note that (30) is a strictly decreasing function in κ, and as such, we have shown the theorem.               |            |     |
| 20A proof can be found in https://bit.ly/2L6jdAT                                                                      |            |     |
| 36                                                                                                                    |            |     |



## Page 37

E.3.8
Bounds for λ
Lower bound.
ε = tr(Σ∗M2)
≥σmin(Σ∗) · tr(M2)
by the deﬁnition of tr(·)
≥σmin(Σ∗)
d
· tr(M)2
by Cauchy-Schwarz
≥σmin(Σ∗)
d
·
h
tr

(λΣ −I)−1i2
Expanding M (21)
≥σmin(Σ∗)
d
·
h
tr (λΣ −I)−1 · d2i2
AM-HM inequality
≥d3 · σmin(Σ∗) · [λ · tr(Σ) −d]−2
[λ · tr(Σ) −d]2 ≥d3 · σmin(Σ∗)
ε
λ · tr(Σ) −d ≥d3/2 ·
p
σmin(Σ∗)
√ε
since M is PSD
λ ≥
d
tr(Σ)
 
1 +
r
d · σmin(Σ∗)
ε
!
Upper bound
ε = tr(Σ∗M2)
≤∥Σ∗∥F · d · σmax(M)2
≤∥Σ∗∥F · d · σmin(M)−2
λ · σmin(Σ) −1 ≤
r
∥Σ∗∥F · d
ε
λ ≤
1
σmin(Σ)
 r
∥Σ∗∥F · d
ε
+ 1
!
.
37


**Table 42 from page 37**

| 0   | 1     | 2              | 3         | 4      | 5                 | 6         | 7                       |
|:----|:------|:---------------|:----------|:-------|:------------------|:----------|:------------------------|
|     |       | M2)            |           |        |                   |           |                         |
|     |       | ε = tr(Σ       |           |        |                   |           |                         |
|     |       | ∗              |           |        |                   |           |                         |
|     |       | )              | tr(M2)    |        |                   |           | )                       |
|     |       | σmin(Σ         |           |        |                   |           | by the deﬁnition of tr( |
|     |       | ∗              |           |        |                   |           | ·                       |
|     |       | ≥              |           |        |                   |           |                         |
|     |       | ·              |           |        |                   |           |                         |
|     |       | )              |           |        |                   |           |                         |
|     |       | σmin(Σ         |           |        |                   |           |                         |
|     |       | ∗              | tr(M)2    |        |                   |           | by Cauchy-Schwarz       |
|     |       | d              | ·         |        |                   |           |                         |
|     |       | ≥              |           |        |                   |           |                         |
|     |       | )              |           |        |                   | 2         |                         |
|     |       | σmin(Σ         |           |        |                   |           |                         |
|     |       |                |           |        | 1                 |           |                         |
|     |       | ∗              | tr        | (λΣ    | I)−               |           | Expanding M (21)        |
|     |       | d              | ·         | −      |                   |           |                         |
|     |       | ≥              |           |        |                   |           |                         |
|     |       |                | (cid:104) |        | (cid:17)(cid:105) |           |                         |
|     |       |                | (cid:16)  |        |                   |           |                         |
|     |       | )              |           |        |                   | 2         |                         |
|     |       | σmin(Σ         |           |        |                   |           |                         |
|     |       |                |           |        | 1                 |           |                         |
|     |       | ∗              | tr (λΣ    | I)−    | d2                |           | AM-HM inequality        |
|     |       | d              | ·         | −      | ·                 |           |                         |
|     |       | ≥              |           |        |                   |           |                         |
|     |       |                | (cid:104) |        |                   | (cid:105) |                         |
|     |       | d3             | )         | tr(Σ)  | d]−               |           |                         |
|     |       | σmin(Σ         | [λ        |        |                   |           |                         |
|     |       | ≥              | ∗         | ·      | −                 |           |                         |
|     |       | ·              | ·         |        |                   |           |                         |
|     |       | d3             | )         |        |                   |           |                         |
|     |       | σmin(Σ         |           |        |                   |           |                         |
| [λ  | tr(Σ) | d]2            | ∗         |        |                   |           |                         |
|     |       | ·              |           |        |                   |           |                         |
| ·   | −     | ε              |           |        |                   |           |                         |
|     |       | ≥              |           |        |                   |           |                         |
|     |       | d3/2           | σmin(Σ    | )      |                   |           |                         |
| λ   | tr(Σ) | d              | ∗         |        |                   |           | since M is PSD          |
|     |       | ·              |           |        |                   |           |                         |
| ·   | −     | ≥              |           |        |                   |           |                         |
|     |       | √ε             |           |        |                   |           |                         |
|     |       | (cid:112)      |           |        |                   |           |                         |
|     |       | d              | d         | σmin(Σ | )                 |           |                         |
|     |       | λ              | 1 +       | ·      | ∗                 |           |                         |
|     |       | ≥              |           | ε      |                   |           |                         |
|     |       | tr(Σ) (cid:32) |           |        |                   |           |                         |
|     |       |                | (cid:114) |        | (cid:33)          |           |                         |

