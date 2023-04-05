---


layout: review
title: "Auditing Privacy Defenses in Federated Learning
via Generative Gradient Leakage"
tags: federated-learning privacy gradient-leakage computer-vision
author: "Matthis Manthe"
cite:
    authors: "Zhuohang Li, Jiaxin Zhang, Luyang Liu, Jian Liu,"
    title:   "Auditing Privacy Defenses in Federated Learning
via Generative Gradient Leakage"
    venue: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition 2022
pdf: "https://arxiv.org/pdf/2203.15696.pdf"
---

# Notes

* Code is available on [github](https://github.com/zhuohangli/GGL)

# Introduction

* Federated learning was initially presented as a fully **privacy-preserving** decentralized learning paradigm, never sharing raw data between participants and the server,

<div style="text-align:center">
<img src="/collections/images/generative_gradient_leakage_fl/Federated_setup.jpg" width="60%" height="30%"></div>

* Soon after its proposal, attacks designed _to extract information from the transmitted local updates_ were developed, the most notable being **Deep Leakage from Gradients[^1]**, only on small-sized inputs,

* The authors propose a new type of attack, **Generative Gradient Leakage**, _leveraging external data server-side to train a generative model used to regularize an inversion attack._

<div style="text-align:center">
<img src="/collections/images/generative_gradient_leakage_fl/generative_gradient_leakage.jpg" width="60%" height="30%"></div>

* They show promising results attacking federated frameworks training on ImageNet and CelebA, reconstructing a large image from a gradient.



<div style="text-align:center">
<img src="/collections/images/generative_gradient_leakage_fl/generative_gradient_leakage_threat_model.jpg" width="80%" height="50%"></div>

# Threat model

* **_Honest-but-curious server_**. It behaves normally, but will try its best to infer private information from what is accessible to it, **models and degraded local updates**.
* Participants apply a variety of local defense mechanisms, 
* **_Access to external data server-side from the same distribution as participants._**

# Problem definition
## Notations
* $$x\in\mathbb{R}^{h\times w}$$ an image with associated class label $$c$$,
* $$F(x)=\nabla_{\theta}\mathcal{L}(f_{\theta}(x),c)$$ the gradient of $$x$$ with respect to parameters $$\theta\in\mathbb{R}^p$$ of a network $$f_{\theta}$$,
* $$y=\mathcal{T}(F(x))$$ a gradient degraded by the defence transformation $$\mathcal{T}$$,
* $$w: \mathbb{R}^{h\times w}$$ an image prior defining the likelihood of an image,
* $$\mathcal{D}$$ a distance metric between gradients,
* $$\mathcal{G}: \mathbb{R}^k\rightarrow \mathbb{R}^{h\times w}$$ a generative network trained on external data server-side, 
* $$\mathcal{R}(G;z)$$ a regularization term, penalizing latent vector $$z$$ when deviating from the prior distribution.

## Previous formulation[^1]

$$ x^* = \underset{x\in\mathbb{R}^{h\times d}}{argmin\ } \underbrace{\mathcal{D}(y, F(x))}_\text{gradient matching loss} + \underbrace{\lambda \omega(x)}_\text{regularization} $$ 

## Novel formulation

$$ z^* = \underset{z\in\mathbb{R}^{k}}{argmin\ } \underbrace{\mathcal{D}(y, \mathcal{T}(F(G(z))))}_\text{gradient matching loss} + \underbrace{\lambda \mathcal{R}(G; z)}_\text{regularization}$$

# Method

## Label inference
For networks with a fully connected classification head using sigmoid or ReLu activation functions trained with cross-entropy loss on one-hot labels, the label can be inferred through the index of the negative entry of the gradient with respect to this layer.
 
## Gradient Transformation Estimations
_Gradient Clipping_: Given perturbation $$\mathcal{T}(w, S) = w/\text{max}(1,\frac{||w||_2}{S})$$, estimate the clipping bound $$S$$ as the l2 norm at each layer,

_Gradient Sparsification_: Given perturbation $$\mathcal{T}(w,p) = y \odot \mathcal{M}$$ with $$\mathcal{M}$$ a mask with pruning rate $$p$$, simply estimate $$\mathcal{M}$$ using the zero entries of the gradient.

## Gradient Matching Losses
_Squared l2 norm_ 
$$\mathcal{D}(y, \tilde{y}) = ||y - \tilde{y}||^2_2$$,

_Cosine similarity_ 
$$\mathcal{D}(y, \tilde{y}) = 1 - \frac{<y,\tilde{y}>}{||y||_2||\tilde{y}||_2}$$.

## Regularization Terms
_KL-based regularization_
$$\mathcal{R}(G;z)=-\frac{1}{2}\sum_{i=1}^k(1+\text{log}(\sigma_i^2)-\mu_i^2-\sigma_i^2)$$ with $$\mu_i$$ and $$\sigma_i$$ the element-wise mean and standard deviation,

_Norm-based regularization_
$$\mathcal{R}(G;z) = (||z||^2_2-k)^2$$

## Optimization Strategies
_Gradient-based_
* Adam

_Gradient-free_
* Bayesian optimization (trust region BO (TuRBO))
* Covariance matrix Adaptation evolution strategy (CMA-ES)


# Experiments
## Defence schemes
* _Additive Noise_: inject a Gaussian noise $$\epsilon \sim \mathcal{N}(0,\sigma^2I)$$ to the gradients with $$\sigma = 0.1$$,
* _Gradient Clipping_: clip the values of the gradients with a bound of $$S = 4$$,
* _Gradient Spasification_: perform magnitude-based pruning on the gradients to achieve 90% sparsity,
* _Soteria_: gradients are generated on the perturbed representation with a pruning rate of 80%.

# Results

## Comparison to SOTA and effect of defences 
<div style="text-align:center">
<img src="/collections/images/generative_gradient_leakage_fl/generative_gradient_leakage_defenses_results.jpg"></div>

<div style="text-align:center">
<img src="/collections/images/generative_gradient_leakage_fl/generative_gradient_leakage_attack_examples.jpg"></div>

<div style="text-align:center">
<img src="/collections/images/generative_gradient_leakage_fl/generative_gradient_leakage_attack_example_3.jpg"></div>
&nbsp;
## Losses and regularizations
<div style="text-align:center">
<img src="/collections/images/generative_gradient_leakage_fl/generative_gradient_leakage_loss_results.jpg" width="80%" height="50%"></div>
&nbsp;
## Optimizers
<div style="text-align:center">
<img src="/collections/images/generative_gradient_leakage_fl/generative_gradient_leakage_examples.jpg" width="80%" height="50%"></div>
<div style="text-align:center">
<img src="/collections/images/generative_gradient_leakage_fl/generative_gradient_leakage_opti_results.jpg" width="80%" height="50%"></div>
&nbsp;



# Conclusion

* The authors show promising results attacking federated frameworks training on ImageNet and CelebA, reconstructing a large image from a gradient.

* _This work must be seen as a valuable proof of concept_: the assumptions are extremely strong limiting its applicability in real contexts, but the idea is quite appealing, opening a whole new spectrum of gradient inversion techniques.

# References
[^1]: [Ligeng Zhu et al, Deep Leakage from Gradients](https://arxiv.org/abs/1906.08935)  
