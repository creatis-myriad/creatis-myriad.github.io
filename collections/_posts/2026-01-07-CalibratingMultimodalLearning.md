---
layout: review
title: "Calibrating Multimodal Learning"
tags: multimodality self-supervised calibration
author: "Robin Trombetta"
cite:
    authors: "Huan Ma,Qingyang Zhang, Changqing Zhang, Bingzhe Wu, Huazhu Fu, Joey Tianyi Zhou, Qinghua Hu"
    title: "Calibrating Multimodal Learning"
    venue: "International Conference on Machine Learning (ICML) 2023"
pdf: "https://proceedings.mlr.press/v202/ma23i/ma23i.pdf"
---

&nbsp;

# Introduction

Multimodal (MM) deep learning have been studied a lot recently and achieved remarkable results in a wide variety of tasks, but their reliability is still under explored. The authors study the predictive confidence of multimodal models and propose a regularization technique to calibrate such models.

For multimodal models, a basic assumption is that all modalities should be predictive for the target. Although one could think of a particular setup where this assumption falls short, it is intuitively suited for most MM tasks. Under this assumption, the confidence of an ideal multimodal model should not increase when one modality is removed. An illustration of this principle is given in Figure 1. Models that don't satisfy this property may not be more influenced when some modalities are missing or corrupted. 

<div style="text-align:center">
<img src="/collections/images/CalibratingMultimodalLearning/overview_confidence.jpg" width=800></div>
<p style="text-align: center;font-style:italic">Figure 1. Illustration of the multimodal confidence hypothesis.</p>

&nbsp;

# Problem formulation and motivation

Let's take a training dataset $$\mathcal{D} = \{ \{ x_i^m \}_{m=1}^M , y_i \}_{i=1}^N$$ where $$x_i^m$$ is the $$m$$-th modality of the $$i$$-th sample and $$y_i$$ the corresponding label. We use the notation $$x^m$$ to represent the $$m$$-th modality and $$x^{\mathbb{S}}$$ to denote multiple modalities, where $$\mathbb{S}$$ is a set of modalities' indexes. $$x^{\mathbb{M}}$$ indicates the case where all the M modalities are present.

Given a $$\theta$$-parametrized neural network $$f_{\theta}$$, we define the probability distribution of a sample $$x$$ as $$P(y \mid \theta, x^{\mathbb{M}}) = \{ \hat{p_k} \}_{1}^K$$. The predicted class label label is $$\hat{y} = \arg \max_y P(y \mid \theta, x^{\mathbb{M}})$$ and the confidence is defined as $$\textrm{Conf}(x^{\mathbb{M}}) = \max _y P(y \mid \theta, x^{\mathbb{M}})$$.

The principle stated above can be formulated as follows:

*Proposition.* Given two versions of a sample $$x^{\mathbb{M}}$$, *i.e.* $$x^{\mathbb{T}}$$ and $$x^{\mathbb{S}}$$, if we can assure $$\mathbb{T} \subset \mathbb{S} \subseteq \mathbb{M}$$ then a trustworthy multimodal classifier $$f$$ should verify $$\textrm{Conf}(f(x^{\mathbb{T}})) \leq \textrm{Conf}(f(x^{\mathbb{S}}))$$.

&nbsp;

We define the **Confidence Increment (CI)** for a sample as:

$$
\textrm{CI}(x^{\mathbb{T}}, x^{\mathbb{S}}) = \textrm{Conf}(f(x^{\mathbb{T}})) - \textrm{Conf}(f(x^{\mathbb{S}}))
$$

To quantify to what extent a MM model violates the proposition above, we define a novel metric, the **Violation Ranking Rate (VRR)**:

$$
\textrm{VRR} = \mathbb{E}_{(\mathbb{T}, \mathbb{S})} [ \mathbb{1}(\textrm{CI}(x^{\mathbb{T}}, x^{\mathbb{S}})) ]
$$
with $$\mathbb{T} \subset \mathbb{S} \subseteq \mathbb{M}$$.

The authors have quantify this metric of three MM model models and show that the model often violated the main proposition (Figure 2). A naive strategy to correct this problem could be to re-balance the contribution of every modality, but the Figure 2 also shows that different samples are over-confident on different modalities.


<div style="text-align:center">
<img src="/collections/images/CalibratingMultimodalLearning/proposition_violation.jpg" width=800></div>
<p style="text-align: center;font-style:italic">Figure 2. Current MM methods violate the main proposition. The pie charts show that different samples over-rely of different modalities (e.g. 53% Mod1 indicates that among the samples that violate the proposition, there is 53 precent of samples whose confidence will increase when Mod2 is removed and the other will increase confidence when Mod1 is removed).</p>


&nbsp;

# Calibrating Multimodal Learning

Thei proposal is to add a regularization term during the training to better comply with the proposition above. One idea could be to minimize the following term:

$$
\mathcal{L}^{(\mathbb{T}, \mathbb{S})} = \textrm{Conf}(f(x^{\mathbb{T}})) - \textrm{Conf}(f(x^{\mathbb{S}}))
$$

However, they empirically show that this leads the model to have very low prediction confidence when not all the modalities are present, as shown by Figure 3.

<div style="text-align:center">
<img src="/collections/images/CalibratingMultimodalLearning/experiment_lowconf.jpg" width=800></div>
<p style="text-align: center;font-style:italic">Figure 3. Confidence estimation when penalizing the confidence difference.</p>


&nbsp;

Instead, they opted for a looser constraint by only penalizing the situation where the confidence increases when one modality is removed. Hence, the regulization term is:

$$
\mathcal{L}^{(\mathbb{T}, \mathbb{S})} = \max (0, \textrm{Conf}(f(x^{\mathbb{T}})) - \textrm{Conf}(f(x^{\mathbb{S}})))
$$

The overall training algorithm is:

<div style="text-align:center">
<img src="/collections/images/CalibratingMultimodalLearning/algorithm.jpg" width=400></div>


&nbsp;

# Experiments & Results

The authors conduct several experiments to assess whether or not their method improves the calibration, robustness and performance of MM classification models.


&nbsp;

They show that their regularization improves the VRR, *i.e.* the main proposition is less often transgressed

<div style="text-align:center">
<img src="/collections/images/CalibratingMultimodalLearning/exp_vrr.jpg" width=800></div>
<p style="text-align: center;font-style:italic">Figure 4. VRR (%) of test samples with (tick) or without (cross) the proposed regularization.</p>

<div style="text-align:center">
<img src="/collections/images/CalibratingMultimodalLearning/exp_confidence.jpg" width=400></div>
<p style="text-align: center;font-style:italic">Figure 5. Confidence estimation when one modality is removed.</p>

&nbsp;

Their regularization also improves the accuracy performance of the models.

<div style="text-align:center">
<img src="/collections/images/CalibratingMultimodalLearning/exp_performance.jpg" width=800></div>
<p style="text-align: center;font-style:italic">Figure 6. Accuracy performance comparison on several datasets of models with (tick) or without (cross) their regularization.</p>

&nbsp;

Finally, their better calibrated models improves the accuracy when part of the data is corrupted.

<div style="text-align:center">
<img src="/collections/images/CalibratingMultimodalLearning/exp_noise.jpg" width=800></div>
<p style="text-align: center;font-style:italic">Figure 7. Accuracy performance comparison when some modalities are corrupted with Gaussian noise.</p>

&nbsp;

# Conclusion

In this work, the authors study the problem of miscalibrated multimodal models. They show that the principle that *the essence of information is to eliminate uncertainty (Shannon)* is not always respected and propose a regularization term which helps better respecting it. Their additional loss also improves the accuracy performance of the models and their robustness to corrupted data, showing that the calibration of multimodal models should be studied with more deeply.



