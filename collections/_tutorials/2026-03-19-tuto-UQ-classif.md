---
layout: post
title:  "Introduction to Uncertainty Quantification for Deep Learning Models"
author: 'Mathilde Dupouy'
date:   2026-03-19
categories: classification, uncertainty quantification, deep learning
---
<style>
  div.post-content p {
    text-align: justify; /* helps the reading flow */
  }
</style>

# Summary

- [**Introduction**](#introduction)
    - [Why quantifying uncertainty?](#why-quantifying-uncertainty)
    - [Sources and types of uncertainty](#sources-and-types-of-uncertainty)
- [**Modeling uncertainty**](#modeling-uncertainty)
    - [Bayesian framework](#bayesian-framework)
    - [And beyond...](#and-beyond)
- [**Quantifying uncertainty**](#quantifying-uncertainty)
    - [Different goals for different tasks](#different-goals-for-different-tasks)
    - [Bayesian Neural Networks](#bayesian-neural-networks)
    - [Monte Carlo dropout](#monte-carlo-dropout)
    - [Ensemble methods](#ensemble-methods)
    - [Evidential learning](#evidential-learning)
    - [Conformal Prediction](#conformal-prediction)
- [**Evaluating uncertainty**](#evaluating-uncertainty)
    - [Calibration](#calibration)
    - [Out-Of-Domain detection](#out-of-domain-detection)
    - [Adding evaluation of reliability](#adding-evaluation-of-reliability)
- [**References**](#references)

## **Introduction**
### Why quantifying uncertainty?
Deep learning models have achieved remarkable performance in a wide range of tasks. However, these models typically produce **point predictions**. In many real-world applications, such as medical diagnosis, autonomous driving or climate forecasting, knowing **how reliable a prediction is** can be just as important as the prediction itself. In this tutorial, we will only consider **predictive uncertainty**, in the sense of the uncertainty given an input sample.

Uncertainty quantification aims to provide a measure of confidence or reliability associated with model outputs. This is useful at two levels:
- **when the model is deployed**, it supports decision-making. For example, it can help to catch model failure, to redirect the predictions to human experts, or for the expert to discard the prediction. It can also guide active learning strategies.
- **when the model is "under construction"**, it helps evaluating a model and its limitations. It can also directly be used in some training strategies [ref ?].


### Sources and types of uncertainty
Uncertainty in machine learning predictions arises from multiple sources, which are often categorized into two main types.

**Aleatoric uncertainty (AU)** reflects the inherent randomness or noise in **the data generation process**. It is also referred to as "conflict" in classification, as it comes from an ambiguity between different classes. For example, it can arise from:
- the fact that measurement is a partial view of the true phenomenon;
- randomness of the true phenomenon itself;
- measurement noise in sensors;
- ambiguity in labeling...

This uncertainty is also sometimes called "irreducible uncertainty" because it cannot be reduced simply by collecting more data, since it is tied to the stochastic nature of the observations.


**Epistemic uncertainty (EU)**, on the other hand, arises from **a lack of knowledge about the true model**, also referred to as "ignorance". It reflects uncertainty in the model parameters or structure. It is related to all the sources of approximation needed because the perfect predictor is intractable (Figure 1):
- the choice of a model family $$\mathcal{H}$$ in which the optimal predictor $$h^*$$ is not the theoretical optimal predictor $$f_1^*$$;
- learned solution $$\tilde{h}$$ from an optimization process that does not align with the optimal $$h^*$$;
- a possible difference between training distribution and real-world distribution that shifts the optimal predictor (sometimes called "distributional uncertainty").

Unlike aleatoric uncertainty, epistemic uncertainty can often be reduced by gathering more informative data or improving the model.

<div style="text-align:center">
<img src="/collections/images/UQ_classif/EU_Heetal.jpg" width=500></div>
<p style="text-align: center;font-style:italic">Figure 1. Visualization of various model uncertainty sources [He et al., ACM Comput. Surv., 2026].</p>

This distinction has an interest both theoretically and in practice. Theoretically, as illustrated later in [**Quantifying uncertainty**](#quantifying-uncertainty), it exists mathematical decomposition of predictive uncertainty into two terms that can be linked to these two concepts. This distinction also leads to the design of different modeling techniques to capture different aspects of uncertainty.


In practice, when a prediction has high uncertainties, it triggers different actions [ref Towards].
**If AU is high**, the clinician has the confirmation that it is a difficult sample, that the input data is ambiguous. He can run other tests to clear up the ambiguity. For the data scientist, it means it can be interesting to add other sources of information (other modalities, metadata, physics etc.) to the model.
**If EU is high**, the clinician knows the prediction is not reliable, the model may fail on this input. It also means this data is of interest to further train the model. For the data scientist, it means the model needs to be more general, for example by adding data (new relevant samples, augmentations, ...).

<div style="text-align:center">
<img src="/collections/images/UQ_classif/2D_AUvsEU.svg" width=300></div>
<p style="text-align: center;font-style:italic">Figure 2. Illustration of the distinction between aleatoric and epistemic uncertainties [inspired from Hüllermeier et al., Mach. Learning, 2021].</p>

<div style="text-align:center">
<img src="/collections/images/UQ_classif/PETCTexamples_Lohretal.jpg" width=600></div>
<p style="text-align: center;font-style:italic">Figure 3. Examples of probabilistic predictions and uncertainty quantification results on a whole-body PET/CT dataset, with orange boxes surrounding the tumors identified by the physicians [Löhr et al., AIME, 2024].</p>

These two types of uncertainty are general concepts that help highlighting the different sources of uncertainty, but their distinction can be blurry. The ambiguity needs to be clear up in each specific context [Hullermeier]. These uncertainty are also strongly linked, as adding a dimension can decrease AU but increase EU, or having more samples can increase AU but decrease EU.

<div style="text-align:center">
<img src="/collections/images/UQ_classif/lessData_hullermeieretal.jpg" width=500></div>
<p style="text-align: center;font-style:italic">Figure 4.  Adding data may reduce EU but can also increase AU. [Hüllermeier et al., Mach. Learning, 2021].</p>

<div style="text-align:center">
<img src="/collections/images/UQ_classif/moreDimensions_Hullermeieretal.jpg" width=500></div>
<p style="text-align: center;font-style:italic">Figure 5. Adding a dimension may reduce AU but can also increase EU. [Hüllermeier et al., Mach. Learning, 2021].</p>


## **Modeling uncertainty**
### Bayesian inference
Uncertainty are often characterized thanks to probabilities, and in particular in the Bayesian framework. Modeling uncertainty as distributions is in fact pretty natural in the field. This is why I will develop here this framework.
We assume that we have a training set $$\mathcal{D}$$ which contains N samples drawn independently from a distribution $$p$$ on $$\mathcal{X}\times\mathcal{y}$$.
* **Uncertainty in the data (aleatoric)** Given an instance $$x_q$$, the associated outcome $$y$$ is not necessarily certain i.e. the best that we can obtain is a distribution:

$$
p(y|x_q)=\frac{p(x_q, y)}{p(x_q)}
$$

We want to find a predictor $$f : \mathcal{X} \longrightarrow \mathcal{Y}$$ (inthat would best **fit the data regarding an objective function $$l$$**:

$$
f^*(x):=\text{argmin}_{ŷ\in\mathcal{Y}}\int_\mathcal{Y}l(y,ŷ)dP(y|x)
$$

* **Uncertainty in the model (epistemic)** To do so, we only have access to an **hypothesis space $$\mathcal{H}$$,** (induction) with predictors $$h: \mathcal{X} \longrightarrow \mathcal{Y}$$) of models and we optimize a model by leveraging true labels to minimize the **risk $$R$$ under the objective function**:

$$
R(h):=\int_{\mathcal{X}\times\mathcal{Y}}l(h(x), y) dP(x, y)
$$

and we would like to obtain the best model in this space:

$$
h^* = \text{argmin}_{h\in\mathcal{H}}R(h)
$$

However, we only have access to a **finite number of elements**, thus we often choose to minimize the empirical risk:

$$
ĥ = \text{argmin}_{h\in{\mathcal{H}}} \frac{1}{N} \sum_{i=1}^N l(h(x_i), y_i)
$$

The total model uncertainty refers to the discrepancy between $$f^*$$ and $$ĥ$$. If we assume that the hypothesis space covers the optimal predictor ($$f^*=h^*$$), we can model the uncertainty due to approximation by the optimizer by a distribution $$\mathcal{Q}$$ over the space $$\mathcal{H}$$.
 
### And beyond...
The Bayesian framework is not the only way to model uncertainty... As we can think about it as modelling ignorance, other frameworks are compatible, such as **version space learning**, where we consider a **set of the predictors** that are plausible on the train set rather than a distribution. A vizualization of this diversity of methods is shown Figure 6.

<div style="text-align:center">
<img src="/collections/images/UQ_classif/frameworks_Hullermeieretal.jpg" width=500></div>
<p style="text-align: center;font-style:italic">Figure 6. Set-based versus distributional knowledge representation on the level of predictions, hypotheses, and models. Representation as sets are blue forms, whereas representation in terms of distribution are in gray shading [Hüllermeier et al., Mach. Learning, 2021].</p>

## **Quantifying uncertainty**
Nice, we have modelled uncertainty... but in practice, we want a quantity that will translate this uncertainty, to be able to manipulate it and interpret it. This part illustrates first the kind of properties we want depending on the task, then lists several methods of quantification, focusing on deep learning. 

### Different goals for different tasks
Because machine learning is used for different tasks, uncertainty is described in different ways.

<div style="text-align:center">
<img src="/collections/images/UQ_classif/task_vs_uq.svg" width=900></div>
<p style="text-align: center;font-style:italic">Figure 7. Different quantities for different tasks.</p>

### Bayesian Neural Networks (BNNs)
### Monte Carlo dropout (MCDO)
### Ensemble methods
### Test-time augmentation (TTA)
### Evidential learning
### Conformal Prediction (CP)
## **Evaluating uncertainty**
### Calibration
### Out-Of-Domain detection (OOD)
### Adding evaluation of reliability


## **References**