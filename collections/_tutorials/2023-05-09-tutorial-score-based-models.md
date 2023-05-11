---
layout: post
title:  "Introduction to Score-based models"
author: 'Robin Trombetta'
date:   2023-05-09
categories: score-based models
---

# Notes

* This tutorial was inspired by this [video tutorial](https://www.youtube.com/watch?v=wMmqCMwuM2Q) and this [post](https://yang-song.net/blog/2021/score/) (both by the author itself !)

&nbsp;

- [**Introduction**](#introduction)
- [**Score-based models**](#score-based-models)
  - [Score matching](#score-matching)
  - [Sampling with Langevin Dynamics](#sampling-with-langevin-dynamics)
  - [Examples of results](#examples-of-results)
- [**Conditionnal image generation**](#conditionnal-image-generation)
  - [Principle](#principle)
  - [Examples](#examples)
- [**Stochastic differential equation (SDE)**](#Stochastic-differential-equation(SDE))
  - [Link with denoising diffusion probabilistic models (DDPM)](#link-with-denoising-diffusion-probabilistic-models-(DDPM))
  - [Probability flow ODE and density estimation](#probability-flow-ode-and-density-estimation)
- [**References**](#references)

&nbsp;

## **Introduction**

The main existing generative models can be divided in two categories :
* **likelihood-based models**, which goal is to learn directly the probability density function. Examples of these models are autoregressive models, [normalizing flow](http://127.0.0.1:4000/tutorials/2023-01-05-tutorial_normalizing_flow.html) or [variational auto-encoders](http://127.0.0.1:4000/tutorials/2022-09-12-tutorial-vae.html). 
* **implicit generative models**, for which the density distribution is implicitly learnt by the model during sampling process. This is typically GANs, which have dominated the field of image generation during several years.

Such models each have their specific limitations. Likelihood-based models either have strong restrictions on the model architecture to make sure the normalizing constant of the distribution is tractable and VAEs rely on a substitutes of the likelihood the training. GANs have been historically the state-of-the-art of deep learning generative models in terms of visual quality but they do not allow density estimation and rely on adversarial learning, which is known to be particularly unstable.

In 2019, Yang Song and its collegues proposed a paradigm for image generation based on the **score function**. Instead of trying to explicitly learn a density function, it aims to represent a data distribution by learning the *gradient of the log probability function* .

>Autoencoders can thus be seen as a generalization of the dimensionality reduction PCA formalism by evolving more complex operations defined through $$\mathbf{e}$$ and $$\mathbf{d}$$ networks.

&nbsp;


### **Score-based models**

#### Score matching

Suppose that we have training data $$\{x_1, x_2, ..., x_N\}$$ that are supposed to be drawn from a distribution $$p(x)$$. The goal is to find a $$\theta$$-parametrized density function $$p_{\theta}(x)$$ that accurately estimate the real data density function. Without loss of generalization, we can write : 

$$p_{\theta}(x) = \frac{e^{f_{\theta}(x)}}{Z_{\theta}}$$

where $$Z_{\theta}$$ is a such that $$\int p_{\theta}(x)dx = 1$$. 
We would like to maximize the log-likelihood  

$$\max_{\theta} \sum_{i=1}^{N} \log{p_{\theta}(x_i)}$$


However, it require $$p_{\theta}$$ to be a normalized probability density function, which is in practice impossible for any general f_theta as it means we have to evaluate $$Z_{\theta}$$. The maximiziation of the log-likelihood is thus usually done by restricting the form of $$f_{\theta}$$ (normalizing flow) or via a lower bound of it (VAEs).

The idea behind the score-based models is to learn the *gradients of the density function* instead of the density function instead. This means that we are interested in approximating the quantity $$\Delta_x \log{p_{\theta}(x)}$$, also called the **score function**. Indeed, we have :

$$\Delta_x \log{p_{\theta}(x)} = \Delta_x (\log{(\frac{e^{f_{\theta}(x)}}{Z_{\theta}})}) = \Delta_x f_{\theta}(x) - \underbrace{\Delta_x \log{Z_{\theta}}}_{=0} = \Delta_x f_{\theta}(x)$$

so we don't need to know the normalizing constant anymore. The goal of these models is hence to approximate this quantity by a network-parameterized function $$s_{\theta}(x) \approx \Delta_x f_{\theta}(x)$$.

&nbsp;

<div style="text-align:center">
<img src="/collections/images/score_based/score_matching.jpg" width=600></div>
<p style="text-align: center;font-style:italic">Figure 1. Illustration of score function estimation.</p>

&nbsp;

The optimization can be done via Fisher divergence :

$$ \mathbb{E}_{p(x)} [\lVert \Delta_x \log{p(x)} - s_{\theta}(x) \rVert^2_2]$$

CITER , introduced a technique called score matching and showed that the previous equation can be rewritten :

$$ \mathbb{E}_{p(x)} [\frac{1}{2} \lVert \Delta_x \log{p(x)} \rVert^2_2 + \textrm{trace}(\Delta_x s_{\theta}(x))]$$

Computing trace of the output of a network requires multiple backprop so it is not efficient, but it can be avoided using sliced score matching (projection over random orientations, not detailed here) or denoised score matching (see later).

&nbsp;

#### Sampling with Langevin Dynamics

Suppose we have a score-based model $$s_{\theta}(x) \approx \Delta_x \log{p(x)}$$, we can use **Langevin dynamics** to draw samples from it. The principle is to follow a Monte Carlo Markov Chain, initialized at an arbitrary prior distribution $$x_0 \approx \pi(x)$$ (usually a standard Gaussian distribution) and iterates as follows : 

$$x_{i+1} = x_i + \epsilon \Delta_x \log{p(x)} + \sqrt{2 \epsilon} z_i , \qquad i=0,1, ..., K$$

where $$z \sim \mathcal{N}(0,1)$$. When $$\epsilon \to 0$$ and $$K \to \infty$$, the final $$x_K$$ converges to a sample drawn of $$p(x)$$. Hence having an approximation $$s_{\theta}(x)$$ to plug is the previous equation above allow sampling from the score function.

However, in practice, since sinte optimal $$s_{\theta^*}$$ is found with data samples , the estimated score functions are inaccurate in low density regions, where few data points are available to compute and optimize the score function.

$$ \mathbb{E}_{p(x)} [\lVert \Delta_x \log{p(x)} - s_{\theta}(x) \rVert^2_2] = \int \textbf{p(x)} \lVert \Delta_x \log{p(x)} - s_\theta(x) \rVert_2^2dx$$

&nbsp;

<div style="text-align:center">
<img src="/collections/images/score_based/low_density_regions.jpg" width=700></div>
<p style="text-align: center;font-style:italic">Figure 2. Estimated scores are not accurate in low density regions.</p>

&nbsp;

To tackle this issue, the data distribution is corrupted by noise (Gaussian noise typically) so that a much larger part of the space is explored during score function estimation. High noise provides useful information for Langevin dynamics, but perturbed density no longer approximates the true data density. The solution is to consider a family of increasing noise levels $$\sigma_1 < \sigma_2 < ... < \sigma_L $$ and train a noise conditional score-based model $$s_{\theta}(x, i)$$ with score matching such that $$s_{\theta}(x,i) \approx \Delta_x p_{\sigma_i}(x)$$ for all $$i=1,2,...,L$$.

The new training objective is a weighted sum of Fisher divergences for all noise scales :

$$ \sum_{i=1}^{L} \lambda (i) \mathbb{E}_{p_{\sigma_i}} [\lVert \Delta_x \log{p_{\sigma_i}} (x) - s_{\theta}(x,i) \rVert _2^2]$$

With this new way of doing, sampling is performed by a sequence of Langevin dynamics algorithms with decreasing noise scaling (*i.e.* from $$i=L$$ to $$i=1$$), a method also called *annealed Langevin dynamics*.

&nbsp;

<div style="text-align:center">
<img src="/collections/images/score_based/multiple_noise.jpg" width=700></div>
<p style="text-align: center;font-style:italic">Figure 3. Multiple noise models. Sampling procedure is done from right to left.</p>

&nbsp;

#### Examples of results

In 2021, a score-based model established new state-of-the-art performances for image generation on CIFAR-10 with an Inception score of 9.89 and FID of 2.20, beating previous best model StyleGAN2-ADA.


<div style="text-align:center">
<img src="/collections/images/score_based/example_cifar10.jpg" width=700></div>
<p style="text-align: center;font-style:italic">Figure 4. Example of image generated with a score-based model trained on CIFAR-10.</p>

&nbsp;

<div style="text-align:center">
<img src="/collections/images/score_based/example_faces.jpg" width=700></div>
<p style="text-align: center;font-style:italic">Figure 5. Examples of samples generated from a score-based model trained on CelebA-HQ (1024x1024).</p>

&nbsp;

### **Conditionnal image generation**

#### Principle


Score-based generative models are suitable for solving inverse problems. Let $$\text{x}$$ and $$\text{y}$$ be two random variables and suppose we know the forward process of generating $$y$$ from $$x$$ *i.e.* $$p(\text{y} \vert \text{x})$$. Bayes' rule gives :

$$ p(\text{x} \vert \text{y}) = \frac{p(\text{x})p(\text{y} \vert \text{x})}{p(\text{y})}$$

$$p(\text{y})$$ is unknown but this expression simplify when using the gradients with respect to $$x$$ :

$$ \Delta_x \log{p(\text{x} \vert \text{y})} = \Delta_x \log{p(\text{x})} + \Delta_x \log{p(\text{y} \vert \text{x})} - \underbrace{\Delta_x \log{p(\text{y}})}_{=0}$$

The first term is the score function of the unconditional data distribution $$s_{\theta}(x)$$, hence with the known forward process $$p(\text{y} \vert \text{x})$$ it is possible to compute the posterior score function $$\Delta_x \log{p(\text{x} \vert \text{y})}$$ and then sample from it with Langevin dynamics.

>Note that for a given dataset, we can use the same estimated score function of the unconditional distribution for different conditional tasks

&nbsp;

#### Examples

In this section we show examples of results that can be obtained with conditional score-based model, including image inpainting, image colorization and solving inverse problem for CT-scan reconstruction.
&nbsp;

<div style="text-align:center">
<img src="/collections/images/score_based/example_conditional_inpainting.jpg" D></div>
<p style="text-align: center;font-style:italic">Figure 6. Conditional score-based model for image inpainting.</p>

&nbsp;

<div style="text-align:center">
<img src="/collections/images/score_based/example_conditional_colorization.jpg" width=750></div>
<p style="text-align: center;font-style:italic">Figure 7. Conditional score-based model for image colorization.</p>

&nbsp;

<div style="text-align:center">
<img src="/collections/images/score_based/example_conditional_ct.jpg" width=750></div>
<p style="text-align: center;font-style:italic">Figure 8. Conditional score-based model for CT-scan reconstruction with 23 projections.</p>

&nbsp;

### **Stochastic differential equation (SDE)**

In the previous sections, we introduced a discrete stochastic process where an image is corrupred by a gradually increasing noise. When the number of noise scales approaches infinity, the process becomes a continuous-time stochastic process.

&nbsp;

<div style="text-align:center">
<img src="/collections/images/score_based/continuous_corruption.jpg" width=750></div>
<p style="text-align: center;font-style:italic">Figure 8. Conditional score-based model for CT-scan reconstruction with 23 projections.</p>

&nbsp;

A stochastic process $$\{ \text{x} \}_{t \in [0,T]}$$ is an ensemble infinite number of random variable associated with their probability densities $$\{ p_t(\text{x}) \} _{t \in [0,T]}$$. Many stochastic processes, incuding diffusion can be described by a **stochastic differential equation (SDE)** :

$$ d\text{x}_t = f(\text{x}_t, t)dt + \sigma(t)dw_t$$

where $$f(\cdot, t) : \mathbb{R}^d \to \mathbb{R}^d$$ is a *deterministic drift*, $$\sigma(t) \in \mathbb{R}$$ is the diffusion coefficient and $$w_t$$ the standard Brownian motion, whose infinitesimal variation $$d_w$$ can be seen as white noise and discretized $$ dw_{\epsilon} = \sqrt{\epsilon} \mathcal{N}(0,1).$$

A SDE can be reversed using the score function thanks to the following formula :

$$ dx = [f(\text{x},t) - g(t)^2 \Delta_x \log{p_t(\text{x})} ]dt + g(t) d\overline{w}$$

where $$d\overline{w}$$ is the Brownian process when time flow backwards from 0 to T and $$dt$$ is an negative timestep. Here again, it is possible to generate sample once the score of each marginal distribution $$\Delta_x \log{p_t(x)}$$ is well approximated by a time dependant model $$s_{\theta} (x,t)$$. In practice, a sample x is obtained by solving the previous SDE using Euler-Maruyama method, which is the equivalent of Euuleur method for ODEs. 

&nbsp;

<div style="text-align:center">
<img src="/collections/images/score_based/reverse_sde.gif" width=750></div>
<p style="text-align: center;font-style:italic">Figure 9. Illustration of the reverse SDE process.</p>

&nbsp;

In an analogue manner to the discrete case seen before, such a model is optimized though the objective function : 

$$ \mathbb{E}_{t \sim Uniform[0,T], x \sim p_t}  \lambda (t) \left[ \lVert \Delta_x \log{p_t(x)} - s_{\theta}(x,t) \rVert_2^2 \right] $$

#### Link with denoising diffusion probabilistic models (DDPM)

Denoising Diffusion Probabilistic Models(DDPM), also generally called diffusion models, is a new type of models that appeared in 2015 and were popularized in 2020. In a few years, these models have established state-of-the-art results in image generation and they are now used in almost all famous generative model and applied to many tasks.

They aim to reverse a Markov chain of data perturbed data, corrupted with a small noise such that :

$$ x_i = \sqrt{1 - \beta _i} \text{x}_{i-1} + \sqrt{\beta _i}\text{z}_{i-1}, \qquad i=1,...,N $$

with $$ z_k \sim \mathcal{N}(0,1)$$ and $$\beta$$ close to 0.


Just like Score-Based models with Langevin Dynamic(SMLD), it can be seen as a discretization of the Stochastic Differential Equation :

$$ d\text{x} = -\frac{1}{2} \beta (t) \text{x} \, dt +  \sqrt{\beta(t)} dw$$ 

Convertly, the process described earlier with a sequence of increasing noise level $$\sigma_1 < \sigma_2 < ... < \sigma_L $$ can be seen as a Markov chain :

$$ x_i = x_{i-1} + \sqrt{\sigma^2_i - \sigma^2_{i-1}}\text{z}_{i-1}, \qquad i=1,...,N$$

Score-based models and DDPM are thus almost equivalent as they are a discretization of the same continuous stochastic process. The main difference is that the SDE which DDPM are derived from is Variance Preserving while the one from which SMLD comes from is Variance Exploding.

#### Probability flow ODE and density estimation

A Stochastic Differential Equation can be converted to a non-stochastic equation, called **probability flow ODE** :

$$ d\text{x} = \left[ f(\text{x},t) - \frac{1}{2}\sigma(t)^2 \Delta_x \log{p_t(\text{x})}  \right]dt$$

&nbsp;

<div style="text-align:center">
<img src="/collections/images/score_based/probability_flow_ode.jpg" width=750></div>
<p style="text-align: center;font-style:italic">Figure 10. A SDE have an associated probabilistic flow ODE (white lines), that is not stochastic. Both reverser SDE and probability flow ODE can be obtained by estimating score functions</p>

&nbsp;

The most useful application of such transform is to allow exact log-likelihood computaion, levaring change-of-variable for ODEs :

$$ \log{p_\theta} = \log{\pi(\text{x}_T}) - \frac{1}{2} \int_0^T \sigma(t) \text{trace}(\Delta_x s_\theta (\text{x}, t))dt$$

Such models acheived state-of-the-art log-likelihood estimation on CIFAR-10, surpassing normalizing flow and other models.

<div style="text-align:center">
<img src="/collections/images/score_based/likelihood_cifar10.jpg" width=350></div>
<p style="text-align: center;font-style:italic">Figure 11. NLL and FID of diffusion models on CIFAR-10.</p>


### **References**

[^10]: D. Jimenez Rezende, S. Mohamed. [Variational Inference with Normalizing Flows](https://openreview.net/pdf?id=BywyFQlAW). June 2016.


