---
layout: post
title:  "What about the conditional variational autoencoder?"
author: 'Olivier Bernard'
date:   2022-09-28
categories: autoencoder, conditional, variational, VAE
---

# Notes

* This tutorial was mainly inspired by the following [paper1](https://papers.nips.cc/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html) and [paper2](https://proceedings.neurips.cc/paper/2018/file/473447ac58e1cd7e96172575f48dca3b-Paper.pdf).

&nbsp;

- [**Introduction**](#introduction)
  - [VAE](#vae)
  - [cVAE](#cvae)  
- [**Variational inference**](#variational-inference)
  - [Overall strategy](#overall-strategy)
  - [Formulation of the KL divergence](#formulation-of-the-kl-divergence)  
  - [Evidence lower bound](#evidence-lower-bound)
  - [ELBO reformulation](#elbo-reformulation)

&nbsp;

## **Introduction**

Conditional variational autoencoders (cVAE) should not been seen as an extension of conventional VAE! cVAE are also based on variational inference, but the overall objective is different: 
* In the VAE formalism, a pipeline is optimized to produce output as close as possible to the input data in order to build an efficient ***latent space with reduced dimensionality***. This latent space is then used in inference for interpretation purposes.
* In cVAE formalism, a pipeline is optimize to build a latent space that captures ***reference variability***. This latent space is then used in inference to generate a set of plausible outputs for a given input $$x$$.

### VAE

A complete tutorial on VAE can be found [here](https://creatis-myriad.github.io/tutorials/2022-09-12-tutorial-vae.html). The graph below summarizes the overall strategy used in VAE formalism ***during training***.

![](/collections/images/cvae/vae_training.jpg)


>The goal of VAE is to learn an embedding (latent) space that efficiently represents the distribution of the $$x$$ input in a lower dimensional space for easier interpretation.

&nbsp;

![](/collections/images/cvae/vae_inference.jpg)

>During inference, a new input $$x$$ is given as input to the encoder $$p(z/x)$$ and a dedicated analysis can be performed within the latent space.

&nbsp;

### cVAE

In comparison, the two graphs below shows the overall strategy used in the conditional VAE formalism during training and inference, respectively.

![](/collections/images/cvae/cvae_training.jpg)

>The goal of conditional VAE is to learn an embedding (latent) space that efficently captures the reference variability in a space with reduced dimensionality. 

This is achieved by learning the distribution $$p(z/x,y)$$ which generates a latent space that embeds joint effective information from $$x$$ and $$y$$.

In parallel, the prior network learns to match this distribution by learning $$p(z/x)$$ through the Kullback-Liebler (KL) divergence. The interest of this network after training is that we do not need anymore $$y$$ to get the mapping from $$x$$ to the corresponding latent space. This will be very useful for inference.

&nbsp;

![](/collections/images/cvae/cvae_inference.jpg)

>At time of inference, a new sample $$x$$ is given as input to the prior $$p(z/x)$$ and several points $$z_i$$ are sampled in the corresponding latent space to generate a set of plausible outputs $$\hat{y}_i$$ that will represent the learned variability of the references for a given $$x$$.

&nbsp;

## **Variational inference**

### Overall strategy

The goal of conditional VAE is to approximate a $$p(y/x)$$ distribution through a latent space that captures the variability of references by learning the $$p(z/x,y)$$ distribution. In this way, the distribution $$p(y/x,z)$$ will allow to generate multiple plausible references from a given $$x$$. The following scheme is applied:
* for a given observation $$x$$, a set of latent variables $$z_i$$ is generated from $$p(z/x,y)$$ thanks to the sampling of the corresponding latent space.
* The set of latent variables are then combined with the observation and passed through the conditional generative process $$p(y/x,z)$$ to generate samples from the distribution $$y$$.
* The resulting predictive distribution is finally obtained through the following expression:

$$p(y/x) = \int_z{p(y/x,z) \cdot p(z/x) \,dz}$$

&nbsp;

As for the variational autoencoders, the key concept around conditional VAE is the optimization of the computation of the posterior $$p(z/x,y)$$. Indeed, due to intractable properties, the derivation of this distribution is complicated and requires the use of approximation techniques such as variational inference.

In the conditional VAE formalism, the posterior $$p(z/x,y)$$ is approximated by a Gaussian distribution $$q(z/x,y)$$ whose mean $$\mu_{post}$$ and covariance $$\sigma_{post}$$ are defined by two functions $$g(x,y)$$ and $$h(x,y)$$.

$$q(z/x,y) = \mathcal{N}\left(g(x,y),h(x,y)\right)$$

We thus have a family of candidates for variational inference and need to find the best approximation among this family by minimizing the KL divergence between the approximation $$q(z/x,y)$$ and the target $$p(z/x,y)$$. In other words, we are looking for the optimal $$g^∗$$ and $$h^∗$$ such that:

$$\left(g^*,h^*\right) = \underset{(g,h)}{\arg\min} \,\,\, D_{KL}\left(q(z/x,y) \parallel p(z/x,y) \right)$$

>One particularity of the conditional VAE formalism is that the prior $$p(z/x)$$ is also approximated by a Gaussian distribution $$p(z/x)$$ whose mean $$\mu_{prior}$$ and covariance $$\sigma_{prior}$$ are defined by two functions $$k(x,y)$$ and $$l(x,y)$$.

$$p(z/x)$$ is thus modeled as:

$$p(z/x) = \mathcal{N}\left(k(x,y),l(x,y)\right)$$

As we will see later, minimizing the KL divergence between the approximation $$q(z/x,y)$$ and the target $$p(z/x,y)$$ also leads to finding the optimal $$k^*$$ and $$l^*$$.

&nbsp;

### Formulation of the KL divergence

Let's now reformulate the KL divergence expression

$$D_{KL}\left(q(z/x,y) \parallel p(z/x,y) \right) = - \int{q(z/x,y) \cdot log\left(\frac{p(z/x,y)}{q(z/x,y)}\right) \,dz}$$

Using the following ***conditional probability*** relations:

$$
p(x,y,z) = \left\{
  \begin{array}
    pp(y,z/x) \cdot p(x) \\
    p(z/x,y) \cdot p(x,y)
  \end{array}
  \right.
$$

$$p(x,y) = p(y/x) \cdot p(x)$$

the next equations can be easily obtained

$$p(z/x,y) = \frac{p(y,z/x) \cdot p(x)}{p(x,y)}$$

$$p(z/x,y) = \frac{p(y,z/x) \cdot p(x)}{p(y/x) \cdot p(x)}$$

$$p(z/x,y) = \frac{p(y,z/x)}{p(y/x)}$$

&nbsp;

The previous KL divergence expression can thus be rewritten as

$$D_{KL}\left(q(z/x,y) \parallel p(z/x,y) \right) = - \int{q(z/x,y) \cdot log\left(\frac{p(y,z/x)}{p(y/x) \cdot q(z/x,y)}\right) \,dz}$$

$$ = - \int{q(z/x,y) \cdot 
\left[log\left(\frac{p(y,z/x)}{q(z/x,y)}\right) + log\left(\frac{1}{p(y/x)}\right) \right]\,dz}$$

$$ = - \int{q(z/x,y) \cdot 
log\left(\frac{p(y,z/x)}{q(z/x,y)}\right)\,dz} + log\left(p(y/x)\right) \cdot \underbrace{\int{q(z/x,y)\,dz}}_{=1}$$

$$D_{KL}\left(q(z/x,y) \parallel p(z/x,y) \right) \,+\, \mathcal{L} \,=\, log\left(p(y/x)\right)$$

where $$\mathcal{L}$$ is defined as the ***Evidence Lower BOund (ELBO)*** whose expression is given by:

$$\mathcal{L} = \int{q(z/x,y) \cdot log\left(\frac{p(y,z/x)}{q(z/x,y)}\right) \,dz}$$

&nbsp;

### Evidence lower bound

Let's take a closer look at the previous derived equation:

$$D_{KL}\left(q(z/x,y) \parallel p(z/x,y) \right) \,+\, \mathcal{L} \,=\, log\left(p(y/x)\right)$$

The following observations can be made:
* since $$0\leq p(y/x) \leq 1$$, $$log\left(p(y/x)\right) \leq 0$$

* since $$x$$ and $$y$$ are known, $$log\left(p(y/x)\right)$$ is a fixed value

* by definition $$D_{KL}\left(q(z/x,y) \parallel p(z/x,y) \right) \geq 0$$

* since $$\mathcal{L} = -D_{KL}\left(q(z/x,y) \parallel p(y,z/x)\right)$$, $$\mathcal{L} \leq 0$$


The previous expression can thus be rewritten as follows:

$$\underbrace{D_{KL}\left(q(z/x,y) \parallel p(z/x,y) \right)}_{\geq 0} \,+\, \underbrace{\mathcal{L}}_{\leq 0} \,=\, \underbrace{log\left(p(y/x)\right)}_{\leq 0 \,\, \text{and fixed}}$$

>Thus, by tweaking q(z/x,y), we can seek to maximize the ELBO $$\mathcal{L}$$, which will imply the minimization of the KL divergence $$D_{KL}\left(q(z/x,y) \parallel p(z/x,y) \right)$$, and thus to find a distribution $$q(z/x,y)$$ that is close to $$p(z/x,y)$$.

&nbsp;

### ELBO reformulation

The ELBO $$\mathcal{L}$$ should be reformulated so to justify the loss involved in the conditional VAE framework. The corresponding derivation is provided below.

$$\mathcal{L} = \int{q(z/x,y) \cdot log\left(\frac{p(y,z/x)}{q(z/x,y)}\right) \,dz}$$

By using the following ***conditional probability** relations:

$$
p(x,y,z) = \left\{
  \begin{array}
    pp(y,z/x) \cdot p(x) \\
    p(y/x,z) \cdot p(x,z)
  \end{array}
  \right.
$$

$$p(x,y) = p(z/x) \cdot p(x)$$

the next equations can be easily obtained

$$p(y,z/x) = \frac{p(y/x,z) \cdot p(x,z)}{p(x)}$$

$$p(y,z/x) = \frac{p(y/x,z) \cdot p(z/x) \cdot p(x)}{p(x)}$$

$$p(y,z/x) = p(y/x,z) \cdot p(z/x)$$

&nbsp;

The ELBO $$\mathcal{L}$$ expression can thus be rewritten as

$$\mathcal{L} = \int{q(z/x,y) \cdot log\left(\frac{p(y/x,z) \cdot p(z/x)}{q(z/x,y)}\right) \,dz}$$

$$\mathcal{L} = \int{q(z/x,y) \cdot \left[log\left(p(y/x,z)\right) + log\left(\frac{p(z/x)}{q(z/x,y)}\right) \right] \,dz}$$

$$\mathcal{L} = \int{q(z/x,y) \cdot log\left(p(y/x,z)\right) \,dz} \,+\, \int{q(z/x,y) \cdot log\left(\frac{p(z/x)}{q(z/x,y)}\right) \,dz}$$

$$\mathcal{L} =  \mathbb{E}_{z\sim q(z/x,y)} \left[log\left(p(y/x,z)\right)\right] - D_{KL}\left(q(z/x,y)\parallel p(z/x)\right)$$

where $$\mathbb{E}_{z\sim q(z/x,y)}$$ is the mathematical expectation with respect to $$q(z/x,y)$$. 

&nbsp;

Noting that the generative model $$p(y/x,z)$$ is also modeled by a neural network $$f(\cdot)$$, we are finally looking for:

$$\left(f^*,g^*,h^*,k^*,l^*\right) = \underset{(f,g,h,k,l)}{\arg\max} \,\,\, \left( \mathbb{E}_{z\sim q(z/x,y)} [log(\underbrace{p(y/x,z)}_{f})] - D_{KL}(\underbrace{q(z/x,y)}_{g,h}\parallel \underbrace{p(z/x)}_{k,l}) \right)$$

$$\left(f^*,g^*,h^*,k^*,l^*\right) = \underset{(f,g,h,k,l)}{\arg\min} \,\,\, \left( \mathbb{E}_{z\sim q(z/x,y)} [-log(\underbrace{p(y/x,z)}_{f})] + D_{KL}(\underbrace{q(z/x,y)}_{g,h}\parallel \underbrace{p(z/x)}_{k,l}) \right)$$

