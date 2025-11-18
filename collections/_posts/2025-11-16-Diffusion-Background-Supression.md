---
layout: review
title: "Nuclear Diffusion Models for Low-rank Background Supression in Videos"
tags: rpca, diffusion-models, ultrasound-imaging, denoising
author: "Julia Puig"
cite:
    authors: "Tristan S.W. Stevens, Oisin Nolan, Jean-Luc Robert and Ruud J.G. van Sloun"
    title:   "Nuclear Diffusion Models for Low-rank Background Supression in Videos"
    venue:   "preprint"
pdf: "https://arxiv.org/abs/2509.20886"
---

<br/>

# Highlights
- The authors propose to perform background supression by combining the power of **Robust PCA** and **Diffusion Models**, which can model highly complex data distributions without having to rely on basic assumptions. 
- They successfully apply this framework to dehazing of cardiac ultrasound images.

<br/>

# Background

## Ultrasound image quality
During ultrasound acquisitions, phenomena such as aberration and reverberation produce unwanted echoes that degrade the image quality. In particular, **haze** is an artifact that occurs due to multipath reflections and produces a white haze on the echo image.

{:refdef: style="text-align: center;"}
![](/collections/images/DehazingDiffusion/echo_images.jpg){: width="500" }
{:refdef}

This phenomenon can **make diagnosis much more difficult**, both qualitative and quantitative.

Existing dehazing techniques:
- **Harmonic imaging** receives echoes at frequencies that are multiples of the original frequency. It produces higher quality images, as multipath scatterers have less energy and therefore generate fewer harmonics. However, it results in reduced penetration depth.
- **Clutter filtering methods.**
	- Block-matching and 3D filtering algorithm (BM3D) works by grouping similar patches of the image and then stacking and filtering them. It needs assumptions on the noise distribution.
	- Temporal decompositions (PCA, SVD) allow to separate data correspoding to rapidly moving events (tissue) from data corresponding to stationary events (clutter). This assumption is not always true, leading to mistakes.
	- etc.
- **Deep learning methods.** Supervised approaches have been implemented to supress reverberation haze. They require of a supervised dataset and may have difficulty to generalize across datasets.

<br/>

## Robust PCA for background supression
An ultrasound signal *Y* is acquired. where
$$Y \in \mathbb{C}^{N \times M \times T}$$
can be decomposed in three signals:

$$ Y = L + X + N, $$

where *L* is the background signal, *X* is the tissue signal and *N* is the noise signal.

In the context of contrast-enhanced ultrasound imaging, and if the matrices are reshaped as
$$N \cdot M \times T$$
such that the columns carry the temporal information, some assumptions can be made about *L* and *S*:
- *L* is low-rank because of the spatial coherence of background signal
- *X* can be considered sparse because tissue is not everywhere in the sector

The decomposition of a matrix in low-rank and sparse components is called **Robust Principal Component Analysis** (RPCA) [[1]](https://arxiv.org/abs/0912.3599) and *L* and *S* can be found solving a convex minimization problem.

To recover *L* and *X* the following minimization problem can be written, which promotes low-rank solutions for *L* and sparse solutions for *X*:

$$ \min_{L,X} ||Y-(L+X)||^2_F + \lambda_1 ||L||_* + \lambda_2 ||X||_1, $$

where
$$||\cdot||_F$$
is the Frobenius norm (square root of the sum of all squared elements), and
$$||\cdot||_*$$
is the nuclear norm (sum of the singular values). This is then solved with methods such as ISTA or ADMM.

Alternatively, the Bayesian version of this problem can be written. The joint distribution is given by:

$$ p(Y, L, X) = p(Y | L, X) p(L) p(X). $$

Considering Gaussian noise *N* of variance
$$\sigma^2$$
, the likelihood writes as:

$$ p(Y | L, X) = N(Y; L + X, \sigma^2 I). $$

Enforcing *L* and *X* to be respectively low-rank and sparse the priors are:

$$ p(L) \propto \text{exp}(-\gamma ||L||_*), $$

and

$$ p(X) \propto \text{exp}(-\lambda ||X||_1). $$

Then, taking the negative log-likelihood and finding the maximum a posteriori (MAP) estimate is equivalent to solving the original RPCA optimization problem up to some constants:

$$ (L^*, X^*) = \underset{L,X}{\operatorname{argmax}} \hspace{0.1cm} p(L, X | Y) = \underset{L,X}{\operatorname{argmin}} (- \log p(L, X | Y)), $$

where

$$ p(L, X | Y) \propto p(Y, L, X) = p(Y | L, X) p(L) p(X). $$

<br/>

## Diffusion posterior sampling
Given a measurement model of the form:

$$ y = f(x) + n, $$

where
$$ x $$
is the signal of interest,
$$ y $$
is a measurement,
$$ f $$
is a forward operator, and
$$ n $$
is noise, we usually want to find
$$ x $$
given measurements
$$ y. $$
One way to do this is to sample from the posterior
$$ p(x|y). $$

**Diffusion posterior sampling (DPS)** [[2]](https://arxiv.org/pdf/2209.14687) allows to sample from
$$ p(x|y). $$
The posterior can be written in terms of the prior and the likelihood using Bayes rule:

$$ p(x|y) \propto p(x) p(y|x). $$

Then, the idea of DPS is to represent the prior
$$ p(x) $$
with a diffusion model and to interleave denoising updats of the prior with likelihood-guided steps that move samples towards the masurements
$$ y. $$

1. Denoising step. The forward diffusion process is defined as:

    $$ x_{\tau} = \alpha_{\tau} x_0 + \sigma_{\tau} z, $$

    where
    $$ z \sim N(0, I), x_0 \sim p(x), \tau \in [0, T], $$
    and
    $$ \alpha_{\tau}, \sigma_{\tau} $$
    are predefined noise schedules.

    Then, given a noisy sample
    $$ x_{\tau}, $$
    the denoised estimate is given by:
    
    $$ x_{0|\tau} = \frac{1}{\alpha_\tau} (x_{\tau} - \sigma_{\tau} \epsilon_{\theta}(x_{\tau}, \tau) ), $$

    where
    $$ \epsilon_{\theta}(x_{\tau}, \tau) $$
    is a neural network trained to predict the Gaussian noise
    $$z.$$

2. Guidance step. The denoised estimate is moved towards the measurement using the likelihood:

    $$ x_{0|\tau} = x_{0|\tau} + \eta \nabla_{x_0} \log p(y | x_0) \Big|_{x_0 = x_{0|\tau}}, $$

    where 
    $$ \eta $$
    is a step size and
    $$ x_{0|\tau} $$
    is used as a proxy for
    $$ x_0 $$
    in the likelihood.

<br/>

# Proposed method: Nuclear Diffusion Posterior Sampling

## Motivation
- The sparsity assumption of the RPCA model is too simple to properly model tissue.

## Idea
- Hybrid RPCA and diffusion framework that integrates the low-rank component for background with diffusion posterior sampling.

{:refdef: style="text-align: center;"}
![](/collections/images/DiffusionBackgroundSupression/comparison.jpg){: width="500" }
{:refdef}

<br/>

## Steps of the algorithm
Obs:
- Each sample is a video of shape
$$ N \times M \times T $$
- The denoising diffusion model is applied independently on channels of shape
$$ N \times M $$
- Other steps are applied on reshaped matrices
$$ N \cdot M \times T $$

{:refdef: style="text-align: center;"}
![](/collections/images/DiffusionBackgroundSupression/algorithm.jpg){: width="500" }
{:refdef} 

<br/>

# Experiments

## Configuration
- A pretrained 2D diffusion model is used for the prior on *X*
- 500 diffusion steps using SeqDiff [[3]](https://arxiv.org/html/2409.05399v1) (T=5000) with *Y* as the initialization

## Data
- 4,376 clean samples from 75 easy-to-image subjects
- 2,324 noisy samples from difficult-to-image subjects
- Size of one sample: 60 x 256 x 256

{:refdef: style="text-align: center;"}
![](/collections/images/DiffusionBackgroundSupression/example.jpg){: width="500" }
{:refdef}

## Metrics
- Generalized contrast-to-noise ratio (gCNR) between ventricle and myocardium regions.
- Kolmogorov-Smirnov statistic (KS) to quantify agreement between the original *Y* and denoised *X* tissue distributions in the myocardial region.

$$ \text{KS} = \underset{z}{\operatorname{sup}} | F_{\Omega_S(x)}(z) - F_{\Omega_S(y)}(z) |, $$

where *F()* is the empirical CFD of the respective regions of interest.

<br/>

# Results
{:refdef: style="text-align: center;"}
![](/collections/images/DiffusionBackgroundSupression/ks_statistic.jpg){: width="500" }
{:refdef}

{:refdef: style="text-align: center;"}
![](/collections/images/DiffusionBackgroundSupression/results.jpg){: width="700" }
{:refdef}

{:refdef: style="text-align: center;"}
![](/collections/images/DiffusionBackgroundSupression/boxplot.jpg){: width="500" }
{:refdef}

<br/>

# Conclusions
- The proposed method better separates dynamic tissue from low-rank background compared to RPCA.
- The work shows the potential of combining classical low-rank priors with diffusion models for background supression in videos.


