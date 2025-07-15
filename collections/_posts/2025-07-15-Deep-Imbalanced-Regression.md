---
layout: review
title: "Delving into Deep Imbalanced Regression"
tags: imbalanced-learning regression
author: "Clara Cousteix"
cite:
    authors: "Yuzhe Yang, Kaiwen Zha, Ying-Cong Chen, Hao Wang, Dina Katabi"
    title: "Delving into Deep Imbalanced Regression"
    venue: "38 th International Conference on Machine Learning"
pdf: "https://proceedings.mlr.press/v139/yang21m/yang21m.pdf"     
---

# Highlights

* Defines **Deep Imbalanced Regression (DIR)** as learning from imbalanced data with continuous targets.
* Introduces two distribution smoothing methods: **Label Distribution Smoothing (LDS)** and **Feature Distribution Smoothing (FDS)**.
* Benchmarks performance on several real-world datasets.
* [Code available on GitHub](https://github.com/YyzHarry/imbalanced-regression).

&nbsp;

# Related Work
## Imbalanced Classification

Many prior works have focused on imbalanced **classification** problems. Two main approaches emerged:

* **Data-based methods**: under-sampling majority classes or over-sampling minority classes (e.g., SMOTE).
* **Model-based methods**: modifying loss functions via re-weighting to compensate for imbalance.

Recent studies also show that semi-supervised and self-supervised learning can improve performance under imbalance. While these methods can partially transfer to regression, they have **intrinsic limitations** due to the continuous nature of regression targets.
## Imbalanced Regression

Imbalanced **regression** has been less explored. Most existing studies adapt SMOTE to regression by generating synthetic samples via input-target interpolation or Gaussian noise augmentation. However, these methods, inspired by classification strategies, fail to leverage the **continuity** in the label space. Moreover, linear interpolation may not produce meaningful synthetic samples.

The proposed methods in this paper differ fundamentally and can **complement** these prior approaches.

&nbsp;

# Methods: Label and Features Distribution Smoothing (LDS & FDS)

## Problem Setting

Let $$\{ (x_i, y_i) \}_{i=1}^N$$ be the training set, where $$x_i \in \mathbb{R}^d$$ isthe input and $$y_i \in \mathbb{R}$$ is the continous target. We can divide the label space $$\mathcal{Y}$$ into $$B$$ bins with equal intervals: $$[y_0, y_1[, [y_1, y_2[, .., [y_{n-1}, y_n[$$. The defined bins reflect a minimum resolution we care for grouping data in a regression task (ex: in age estimation, $$\delta y = 1$$ year). Finally, we denote $$z = f(x; \theta)$$ s the feature for x, where $$f(x; \theta)$$, is parameterized by a deep neural network model with parameter $$\theta$$. The final prediction $$\hat{y}$$ is given by a regression function $$g(z)$$.
&nbsp;

## Label Distribution Smoothing (LDS)

#### Motivation Example

To motivate LDS, the authors compare **categorical** labels (CIFAR-100) with **continuous** labels (IMDB-WIKI). Both datasets are subsampled to simulate imbalance, as illustrated on top of Firgure (1).

<div style="text-align:center">
<img src="/collections/images/Deep_Imbalanced_Regression/01_motivation_example.jpg" width=600></div>
<p style="text-align: center;font-style:italic">Figure 1.Comparison on the test error distribution (bottom) using same training label distribution (top) on two different datasets: (a) CIFAR-100, a classification task with categorical label space. (b) IMDB-WIKI, a regression task with continuous label space.</p>


ResNet-50 trained on CIFAR-100 yields a test error distribution strongly correlated with label density (high negative Pearson correlation). For IMDB-WIKI, the test error distribution is smoother and less correlated with label density (Pearson = -0.47). 

This example shows that, given the continuous caracteristic of the label, the network can learn from the neighborhood and give good performance on interpolation. So the imbalance seen by the network is different from the empirical class imbalance. Hence, compensating for data imbalance based on empirical label density is inaccurate for the continuous label space. 

#### LDS Formulation

Label Distribution Smoothing (LDS) applies **kernel density estimation** to smooth the empirical label distribution:

<div style="text-align:center">
<img src="/collections/images/Deep_Imbalanced_Regression/02_LDS.jpg" width=800></div>
<p style="text-align: center;font-style:italic">Figure 2.Label distribution smoothing (LDS) convolves a symmetric kernel with the empirical label density to estimate the effective label density distribution that accounts for the continuity of labels</p>

A symmetric kernel is any kernel that satisfies: $$k(y, y') = k(y', y) \quad$$ and $$\quad \nabla_y k(y, y') + \nabla_{y'} k(y', y) = 0, \quad \forall y, y' \in \mathcal{Y}.$$ (e.g., Gaussian or Laplace). The symmetric kernel characterizes the similarity between target values $$y'$$ and any $$y$$ w.r.t. their distance in the target space. Thus, LDS computes the effective label density distribution as: 

$$
\tilde{p}(y') = \int_{\mathcal{Y}} k(y, y') p(y) \, dy
$$

where $$p(y)$$ is the number of appearances of label of $$y$$ in the training data, and $$\tilde{p}(y')$$ is the effective density of label $$y'$$. 

The smoothed distribution correlates better with error distributions (Pearson = -0.83). Standard imbalance mitigation methods (e.g., re-weighting) can then be applied using $$\tilde{p}(y')$$.
&nbsp;

## Feature Distribution Smoothing (FDS)

We are motivated by the intuition that continuity in the target space should create a corresponding continuity in the feature space. That is, if the model works properly and the data is balanced, one expects the feature statistics corresponding to nearby targets to be close to each other.

#### Motivation example

The authors trained a plain model on the images in the IMDB-WIKI dataset to infer a person’s age from visual appearance. They focused on the feature space, ie **z**, grouped them within the same target age value and computed feature statistics (ie mean and variance) with respect of each bin, denoted $$\{\mu_b, \sigma_b\}$$. To visualize the similarity between feature statistics, we select an anchor bin $$b_0$$, and calculate the **cosine similarity** of the feature statistics between $$b_0$$ and all other bins.

<div style="text-align:center">
<img src="/collections/images/Deep_Imbalanced_Regression/03_FDS_motivation.jpg" width=800></div>
<p style="text-align: center;font-style:italic">Figure 3. Feature statistics similarity for age 30. Top: Cosine similarity of the feature mean at a particular age w.r.t. its value at the anchor age. Bottom: Cosine similarity of the feature variance at a particular age w.r.t. its value at the anchor age. The color of the
background refers to the data density in a particular target range.</p>

The figure above shows that the feature statistics around the anchor b0 = 30 are highly similar to their values at b0 = 30.  So, the figure confirms the intuition that when there is enough data, and for continuous targets, the feature statistics are similar to nearby bins. Interestingly, the figure also shows a high similarity  between the age 0 to 6, that have a very few samples, with b0=30. This unjustified similarity is due to data imbalance. Specifically, since there are not enough images for ages 0 to 6, this range thus inherits its priors from the range with the maximum amount of data, which is the range around age 30.

#### FDS Formulation

FDS aims at transfering the feature statistics between nearby target bin, so that it calibrates the potentially biased estimates of feature distribution, especially for underrepresented target values, with the following procedure:

* Compute feature mean and covariance per bin:

$$\mu_b = \frac{1}{N_b} \sum_{i=1}^{N_b} z_i$$

$$\Sigma_b = \frac{1}{N_b - 1} \sum_{i=1}^{N_b} (z_i - \mu_b)(z_i - \mu_b)^\top$$

* Smooth using kernel $$k(y_b, y_{b'})$$:

$$\tilde{\mu}_b = \sum_{b' \in \mathcal{B}} k(y_b, y_{b'}) \, \mu_{b'}$$

$$\tilde{\Sigma}_b = \sum_{b' \in \mathcal{B}} k(y_b, y_{b'}) \, \Sigma_{b'}$$

* Re-calibrate:

$$z = \tilde{\Sigma}_b^{\frac{1}{2}} \, \Sigma_b^{-\frac{1}{2}} (z - \mu_b) + \tilde{\mu}_b$$

The FDS Algorithm can be integrated to the network with a feature calibration layer after the final feature map. During the training, its employs a *momentum update* (exponential moving average) of the running statistics $$\{\mu_b, \Sigma_b\}$$ across each epoch. Correspondingly, the smoothed statistics $$\{\tilde{\mu}_b, \tilde{\Sigma}_b\}$$ are updated across different epochs but fixed within each training epoch. 

<div style="text-align:center">
<img src="/collections/images/Deep_Imbalanced_Regression/04_FDS_principle.jpg" width=800></div>
<p style="text-align: center;font-style:italic">Figure 3. Feature distribution smoothing (FDS) introduces a feature calibration layer that uses kernel smoothing to smooth the distributions of feature mean and covariance over the target space.</p>

We note that FDS can be integrated with any neural network model, as well as any past work on improving label imbalance.*

&nbsp;

# Results

## Benchmark DIR

#### Datasets
**Age Prediction**. IMDB-WIKI-DIR & AgeDB-DIR: Imbalanced training set, manually constructed balanced validation & test set 

**Health Condition Score**. SHHS-DIR, based on the SHHS dataset, containing night EEG, ECG as inputs and a general heal th score is the output. 

#### Baselines

* *Vanilla Model*: a model that does not include any technique for dealing with imbalanced data. . To combine the vanilla model with LDS, the authors re-weighted the loss function by multiplying it by the inverse of the LDS estimated density for each target bin

* *Synthetic Samples*: Using SMOTER and SMOGN as baselines. SMOTER first defines frequent and rare regions using the original label density, and creates synthetic samples for rare regions by linearly interpolating both inputs and targets. SMOGN further adds Gaussian noise to SMOTER. We note that LDS can be directly used for a better estimation of label density when dividing the target space.

* *Error Aware Loss*: Inspired from the Focal Loss, the authors introduce **Focal-R** Loss = $$\frac{1}{n} \sum_{i=1}^{n} \sigma(\lvert \beta e_i \rvert) ^{\gamma} e_i$$, where $$e_i$$ is the L1 error for i-th sample, $$σ(·)$$ is the Sigmoid function, and $$\beta, \gamma$$ are hyper-parameters. . To combine Focal-R with LDS, we multiply the loss with the inverse frequency of the estimated label density.

* *Two-stage training*: we propose a regression version called regressor re-training (RRT), where in the first stage we train the encoder normally, and in the second stage freeze the encoder and re-train the regressor g(·) with inverse re-weighting. When adding LDS, the re-weighting in the second stage is based on the label density estimated through LDS.

* *Cost-sensitive re-weighting*: Since the authors divided the target space into finite bins, classic re-weighting methods can be directly plugged in. We adopt two re-weighting schemes based on the label distribution: inverse-frequency weighting (**INV**) and its square-root weighting variant (**SQINV**). When combining with LDS, instead of using the original label density, we use the LDS estimated target density.

#### Benchmarks

<div style="text-align:center">
<img src="/collections/images/Deep_Imbalanced_Regression/05_results_DB_age.jpg" width=800></div>

SMOTER and SMOGN can actually degrade the performance in comparison to the vanilla model. Moreover, within each group, adding either LDS, FDS, or both leads to performance gains, while LDS + FDS often achieves the best results. Finally, when compared to the vanilla model, using our LDS and FDS maintains or slightly improves the performance overall and on the many-shot regions, while substantially boosting the performance for the medium-shot and few-shot regions.

<div style="text-align:center">
<img src="/collections/images/Deep_Imbalanced_Regression/06_results_DB_health_score.jpg" width=800></div>


## Further Analysis

#### Interpolation & Extrapolation

To simulate a scenario of certain target values with no samples, the authors curated the age dataset on the training set, as shown on the figure below. 

<div style="text-align:center">
<img src="/collections/images/Deep_Imbalanced_Regression/07_extrapolation_interpolation.jpg" width=800></div>
<p style="text-align: center;font-style:italic">Figure 4. The absolute MAE gains of LDS + FDS over the vanilla model, on a curated subset of IMDB-WIKI-DIR with certain target values having no training data. We establish notable performance gains w.r.t. all regions, especially for extrapolation & interpolation</p>

#### Understanding FDS

<div style="text-align:center">
<img src="/collections/images/Deep_Imbalanced_Regression/08_understanding_FDS.jpg" width=800></div>
<p style="text-align: center;font-style:italic"> Figure 5. <b>(a)</b> Feature statistics similarity for age 0, without FDS <b>(b)</b> Feature statistics similarity for age 0, with FDS <b>(c)</b> L1 distance between the running statistics {μb, Σb} and the smoothed statistics { ˜μb, ˜Σb} during training.</p>

As the figure indicates, since age 0 lies in the few-shot region, the feature statistics can have a large bias, i.e., age 0 shares large similarity with region 40 ∼ 80 as in Fig. 8(a). In contrast, when FDS is added, the statistics are better calibrated, resulting in a high similarity only in its neighborhood, and a gradually decreasing similarity score as target value becomes larger.
The L1 distance between the running statistics $$\{\mu_b, \Sigma_b\}$$ and the smoothed statistics $$\{\tilde{\mu}_b, \tilde{\Sigma}_b\}$$ during training is plotted in Fig. 8(c). Interestingly, the average L1 distance becomes smaller and gradually diminishes as the training evolves, indicating that the model learns to generate features that are more accurate even without smoothing, and finally the smoothing module can be removed during inference.

&nbsp;

# Conclusion

We introduce the DIR task that learns from natural imbalanced data with continuous targets, and generalizes to the entire target range. We propose two simple and effective algorithms for DIR that exploit the similarity between nearby targets in both label and feature spaces.
