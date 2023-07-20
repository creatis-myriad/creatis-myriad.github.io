---
layout: review
title: "Towards Robust Interpretability with Self-Explaining Neural Networks"
tags: interpretability deep-learning intrinsic
author: "Pierre-Elliott Thiboud"
cite:
    authors: "David Alvarez-Mellis, Tommi S. Jaakkola"
    title:   "Towards Robust Interpretability with Self-Explaining Neural Networks"
    venue:   "NeurIPS 2018"
pdf: https://arxiv.org/pdf/1806.07538.pdf
---

Github available [here](https://github.com/dmelis/SENN) (PyTorch)

# Highlights

- Post-hoc interpretability methods are not always faithful in regard to model's predictions and their explanations aren't robust to small perturbations of the input
- The authors propose a class of intrinsically interpretable neural networks models which are as interpretable as linear regression

# Introduction

- Most interpretability methods are post-hoc, explaining the model after its training
- But their explanations lack stability, small changes to the input can lead to different (or even contradicting) explanations for similar predictions
- Linear models are considered to be interpretable, so the proposed design build from this ground

# Self-Explaining Neural Networks models

## Intuition

![](/collections/images/SENN-model/SENN-model-architecture.jpg)

Linear models are considered interpretable for 3 reasons:
- Input features are clearly grounded (arising from empirical measurements)
- Each parameter $$\theta_i$$ provides direct positive/negative contribution of each feature for the prediction
- Features impact can be easily differentiated because of the sum

## Formalization

Linear regression:

$$f(x) = \sum_i^n \theta_i x_i + \theta_0 \tag{1}$$

To improve the modeling power, we can allow the coefficients themselves to depend on the input $$x$$:

$$f(x) = \theta(x)^T x \tag{2}$$

The function $$\theta(\cdot)$$ can be anything from a simple model to deep neural networks. But to maintain a decent level of interpretability, we need to ensure that $$\theta(x)$$ and $$\theta(x')$$ do not differ significantly for close inputs $$x$$ and $$x'$$. More precisely:

$$\nabla_xf(x) \approx \theta(x_0) \tag{3}$$

So the model acts locally, around each $$x_0$$, as a linear model with a vector of stable coefficients $$\theta(x_0)$$

Then, explanations can also be considered in terms of higher order feature, or concepts, derived from the input, like a function $$h(x) : \mathcal{X} \rightarrow \mathcal{Z}$$ where $$\mathcal{Z}$$ is some space of interpretable concepts. The model thus becomes:

$$f(x) = \theta(x)^T h(x) = \sum_{i=1}^K \theta(x)_i h(x)_i \tag{4} $$

And finally, the aggregation function, e.g. the sum, can also be replaced by a more general form. This general aggregation function would need specific properties like permutation invariance or the preservation of relative magnitude of the impact of the relevance values $$\theta(x)_i$$. This would result in a model of the following form:

$$ f(x) = g( \theta_1(x)h_1(x) , ... , \theta_k(x)h_k(x) ) \tag{5}$$

## Definition

The generalized model can thus be formalized as:



with the following desirable properties:

1. $$g$$ is monotone and completely additively separable
2. For every $$z_i = \theta_i(x) h_i(x)$$, $$g$$ satisfies $$\frac{\partial g}{\partial z_i} ≥ 0$$
3. $$\theta(\cdot)$$ is locally stable with respect to $$h(\cdot)$$
4. $$h_i(x)$$ is an interpretable representation of $$x$$
5. $$k$$ is small

<br/>


Properties 1 & 2 entirely depend on $$g$$ which, besides trivial addition, include affine functions $$g(z_1, ..., z_k) = \sum_i A_i z_i$$ with positive $$A_i$$.

Properties 4 & 5 are application-dependent, concepts can be:
- input features directly, $$h(x) = x$$
- subset aggregates of the input, $$h(x) = A x$$ with boolean mask matrix $$A$$
- prototype-based concepts, $$h_i(x) = \| x - z_i \|$$

The property 3 is a little bit trickier to enforce. With the goal to locally mimic linear regression, $$\theta_i$$ can be seen as constant, resulting in the following definition of the model $$f(x) = g(h(x))$$ or $$f(x) = g(z)$$ with $$z = h(x)$$. Then, using the chain rule the gradient of $$f$$ can be reformulated as $$\nabla_x f = \nabla_z f \cdot J_x^h$$. **[As they seek $$\theta(x_0) \approx \nabla_z f$$]**, the local stability enforcer/robustness loss $$\mathcal{L}_\theta$$ can be formulated as:

$$\| \nabla_x f(x) - \theta(x)^T J_x^h(x) \| \approx 0 \tag{2} $$

This results in the following loss for the model to optimize:

$$\mathcal{L}_y(f(x), y) + \lambda \mathcal{L}_\theta(f) + \xi \mathcal{L}_h(x, \hat{x}) \tag{3} $$

# Experiments

Interpretable models need to be evaluated in term of accuracy on the prediction task and how the produced explanations are perceived by explainability recipient, ie, users. So the proposed model is compared with non-interpretable counterparts and 3 criteria are defined to evaluate provided explanations:
1. **Explicitness/Intelligibility**: _are the explanations immediately understandable?_
2. **Faithfulness**: _are relevance scores indicative of "true" importance?_
3. **Stability**: _how consistent are explanations for similar/neighboring examples?_

## Datasets used

[Introduce/show examples of datasets]

- MNIST & CIFAR10  
  32x32 & 32x32x3 images  
  Original datasets with standard mean and variance normalization, using 10% of the training split for validation
- Several benchmark datasets from UCI (Diabetes, Wine, Heart, ...)  
  Using (80%, 10%, 10%) train, validation and test splits, with standard mean and variance scaling on all datasets
- Propublica's COMPAS Recidivism Risk Score  
  Filtered out inconsistent examples and rescaled ordinal variable `Number_of_priors` to range [0, 1]

## Architectures details and accuracy results

The authors do not detail their predictive performances on previously mentioned datasets, only that the proposed models are
> on-par with their non-modular, non interpretable counterparts.

They just precise achieving less than 1.3% error rate on MNIST test set or an accuracy of 78.56% on CIFAR10 test set, which is "on par for models of that size trained with some regularization method".

For each task, they used different architectures for the sub-branches $$h (\cdot)$$ and $$\theta (\cdot)$$ of the SENN model. FC stands for _Fully connected_ layer, and CL for _Convolutional_ layer. The multiplicative constant 10 in the final layer of the $$\theta$$ function for both MNIST and CIFAR10 corresponds to the number of classes. In all cases, the training occurred using Adam optimizer with a learning rate $$l = 2 \times 10^{-4}$$, and the sparsity of learned $$h(\cdot)$$ was enforced through $$\xi = 2 \times 10^{-5}$$.

![](/collections/images/SENN-model/architectures-used-per-dataset.jpg)

## Intelligibility

![](/collections/images/SENN-model/figure2.jpg)

![](/collections/images/SENN-model/results-on-cifar10.jpg)

## Faithfulness

To assess if an explanation correctly estimate the relevance of a feature without ground truth, one can rely on a proxy notion of importance: observing the effect of removing features on the model's prediction. Removing a feature in this model corresponds to setting its coefficient $$\theta_i$$ to zero.

Here, they compute the correlations between prediction probability drops and relevance scores on various points and aggregate the results (Fig.3, Left). Note that $$h(x) = x$$ for UCI datasets used here.

![](/collections/images/SENN-model/figure3.jpg)

## Stability

- Empiric tests show low robustness of interpretability methods to local perturbations of the input.

![](/collections/images/SENN-model/figure4.jpg)

- To formally define a measure, they reuse their "locally difference bounded" property as stability estimation
- Search for adversarial example maximizing this quantity
  $$ \hat{L}(x_i) = \underset{x_j \in B_\epsilon (x_i)}{\arg \max} \frac{\| f_{expl}(x_i) - f_{expl}(x_j) \|_2}{\| h(x_i) - h(x_j) \|_2} \tag{2}$$
  [Question: what's the $$...\|_2$$ for?]
- For raw-input methods, where $$h(x) = x$$, this corresponds to an estimation of the Lipschitz constant

![](/collections/images/SENN-model/figure5.jpg)

# Limitations

- No details on model performances, even in supplementary materials
- Definition of interpretable concepts is a problem in itself
- The final prediction is more explainable, but the intermediate steps are not: the concept encoder and relevance parametrizer are still complete black-boxes
