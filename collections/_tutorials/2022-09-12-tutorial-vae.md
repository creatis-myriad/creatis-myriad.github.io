---
layout: post
title:  "The variational autoencoder paradigm demystified"
author: 'Olivier Bernard'
date:   2022-09-12
categories: autoencoder, encoder, decoder, VAE
---

# Notes

* Here are links to four video that I used to create this tutorial: [video1](https://www.youtube.com/watch?v=4toWtb7PRH4), [video2](https://www.youtube.com/watch?v=uKxtmkfeuxg), [video3](https://www.youtube.com/watch?v=BxkZcS1pLpw), [video4](https://www.youtube.com/watch?v=uaaqyVS9-rM). 
* I was also strongly inspired by this excellent [post](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73).

&nbsp;

- [**Intuition**](#intuition)
  - [Autoencoder VS PCA](#autoencoder-vs-pca)
  - [Why get hurt with a probabilistic context?](#why-get-hurt-with-a-probabilistic-context)
- [**Fondamental concepts**](#fondamental-concepts)
  - [Information quantification](#information-quantification)
  - [Entropy](#entropy)
  - [Kullback-Liebler divergence](#kullback-liebler-divergence)    
  - [Bayes theorem](#bayes-theorem)
- [**Variational inference**](#variational-inference)
  - [Key concept](#key-concept)
  - [Evidence lower bound](#evidence-lower-bound)
  - [ELBO reformulation](#elbo-reformulation)  
  - [From ELBO to VAE](#from-elbo-to-vae)    
  - [VAE network architecture](#vae-network-architecture)    

&nbsp;

## **Intuition**

### Autoencoder VS PCA

Let's start with the basic representation of an autoencoder

![](/collections/images/vae/autoencoder.jpg)

Autoencoders belong to the family of dimension reduction methods. This method takes as input a vector $$\mathbf{x} \in \mathbb{R}^N$$ and outputs a closed vector $$\mathbf{\hat{x}} \in \mathbb{R}^N$$ with the restriction of passing through a space with reduced dimensionality $$Z \in \mathbb{R}^M$$. This is usually achieved through the minimization of the $$L_2$$ norm function: $$\lVert \mathbf{x} - \mathbf{\hat{x}} \rVert^2$$.

$$\mathbf{e}$$ and $$\mathbf{d}$$ are two different networks that model the (non linear) projections from the input space $$X$$ to the latent space $$Z$$ in both directions. 

Let us now consider the simple case where $$\mathbf{e}$$ and $$\mathbf{d}$$ correspond to two single-layer networks without any non-linearity. The corresponding autoencoder can be represented as follows:

![](/collections/images/vae/simplified_autoencoder.jpg)

where $$\mathbf{e} \in \mathbb{R}^{M \times N}$$ and $$\mathbf{d} \in \mathbb{R}^{N \times M}$$ are two linear projection matrices. In the particular case where $$\mathbf{e} = \mathbf{U}^T$$ and $$\mathbf{d} = \mathbf{U}$$, the autoencoder expressions can be written as:

$$\mathbf{z} = \mathbf{U}^T\mathbf{x} \quad\quad \text{and} \quad\quad \mathbf{\hat{x}} = \mathbf{U}\mathbf{z}=\mathbf{U}\mathbf{U}^T\mathbf{x}$$

This corresponds to the well know PCA (Principal Component Analysis) paradigm. A more formal proof can be found [in this article](https://arxiv.org/pdf/1804.10253.pdf).

>>Autoencoders can thus be seen as a generalization of the dimensionality reduction PCA formalism by evolving more complex projection operations defined through $$\mathbf{e}$$ and $$\mathbf{d}$$ networks.

&nbsp;

### Why get hurt with a probabilistic context?

VAEs can be considered as an extension of autoencoders with the introduction of regularization mechanisms (encoder side) to ensure that the generated latent space has good properties allowing the generative process (decoder side). The regularity that is expected from the latent space in order to make generative process possible can be expressed through two main properties: 
* ***continuity***: two close points in the latent space should give close contents when decoded.
* ***completeness***: a point sampled from the latent space should give “meaningful” content once decoded.

These two aspects are optimized within VAEs thanks to a ***probabilistic framework***. 

&nbsp;

#### **Continuity**

In order to introduce local regularization to structure the latent space, the encoding-decoding process is slightly modified: instead of encoding an input as a single point, we encode it as a Gaussian distribution $$q_x(z) = \mathcal{N}\left(\mu_x,\sigma_x\right)$$ over the latent space. Thus, a point $$x\in \mathbb{R}^N$$ at the input of the encoder will correspond to a Gaussian distribution $$q_x(z)$$ at the output, as shown below:

![](/collections/images/vae/vae_local_regularization_3.jpg)


This will ensure that the sampling of a local region in the latent space should produce results that are close.

![](/collections/images/vae/vae_local_regularization_with_decoder.jpg)


However, this property is not sufficient to guarantee continuity and completeness. Indeed encoder can either learn distributions with tiny variances (which would correspond to classical autoencoders) or return distributions with very different means (which would be far apart from each other in the latent space), as illustrated in the figure below.

![](/collections/images/vae/vae_no_global_regularization.jpg)

&nbsp;

#### **Completeness**

In order to avoid these effects the covariance matrix and the mean of the distributions returned by the encoder need to be also regularized. In practice, this new regularization is done by enforcing distributions to be close to a standard normal distribution $$\mathcal{N}\left(0,I\right)$$. This way, the covariance matrices are required to be close to the identity, preventing punctual distributions, and the mean to be close to 0, preventing encoded distributions to be too far apart from each others, as illustrated in the figure below.

![](/collections/images/vae/vae_with_global_regularization_2.jpg)

&nbsp;

>>Thanks to this regularization strategy, we prevent the model to encode data far apart in the latent space and encourage as much as possible returned distributions to overlap, satisfying this way the expected continuity and completeness conditions!

&nbsp;


Using the previous reasoning, the overall architecture of the VAE can be represented as follows.

![](/collections/images/vae/vae_overall_architecture_2.jpg)

&nbsp;

VAE thus offers two extremely interesting opportunities:
* the mastery of the encoder allows to optimize the projection operation $$p(z/x)$$ to a latent space with reduced dimensionality for interpretation purposes. This corresponds to ***manifold learning paradigm***.

<!--[](/collections/images/vae/encoder_illustration_2.jpg)-->

* the mastery of the decoder allows to optimize the projection operation $$p(x/z)$$ for the generation of data with a complex distribution. This corresponds to ***generative model framework***.

<!--![](/collections/images/vae/decoder_illustration_2.jpg)-->

&nbsp;

>>In the rest of this tutorial, we will see how the VAE formalism allows to optimize these two tasks through the theory of variational inference.

&nbsp;

## **Fondamental concepts**

### Information quantification

Information $$I$$ can be quantified through the following expression:

$$I = -log(p(x))$$

with $$x$$ being an event and $$p(x)$$ the probability of this event. Since $$0\leq p(x) \leq 1$$, $$I$$ is positive and tends to infinity when $$p(x)=0$$.

>>From this equation, one can see that when the probability of an event is high (close to $$1$$), the corresponding information is low, which makes sense. For instance, the probability that the weather will be hot in France during summer is very high, so this sentence does not provide any useful information in a conversation

&nbsp;

Please note that the information $$I$$ comes with a unit. If the log is a natural logarithm, we call it a "nat" and if the log has a base 2, we call it a "bit", just like the binary information stored in a computer. Think about it, if you randomly pick one such binary variable in a computer, its chance of being $$0$$ or a $$1$$ is $$50\%$$. Thus, the information associated to the event of observing a $$0$$ or a $$1$$ in a binary computer variable is:

$$-log_2\left(\frac{1}{2}\right)=log_2(2)=1$$

i.e., one bit.

&nbsp;

### Entropy

Entropy $$H$$ corresponds to the ***average information of a process***. Its expression can be naturally written as:

$$H = -\sum_{i=1}^{N}{p(x_i)\cdot log\left(p(x_i)\right)} \quad \quad \quad \text{or} \quad \quad \quad H = -\int{p(x)\cdot log\left(p(x)\right)}\,dx$$

>>Since the information $$-log\left(p(x)\right)$$ is always positive and that $$p(x)$$ is also positive, then the entropy $$H$$ is also positive!

&nbsp;

### Kullback-Liebler divergence

The Kullback-Liebler (KL) divergence allows to ***measure the distance between two distributions*** through the use of relative entropy concepts. Let's define the following expressions:

$$H_{p} = -\int{p(x)\cdot log\left(p(x)\right)}\,dx$$

$$H_{pq} = -\int{p(x)\cdot log\left(q(x)\right)}\,dx$$

where $$H_{p}$$ corresponds to the entropy relative to the distribution $$p(x)$$ and $$H_{pq}$$ the average information brought by $$g(x)$$ but weighted by $$p(x)$$. Note that one can prove that $$H_{pq}>H_{p}$$ whever $$p\neq q$$.

&nbsp;

From these notations, the KL divergence $$D_{KL}$$ can be expressed as:

$$D_{KL}\left(p \parallel q \right) = H_{pq} - H_{p}$$

$$D_{KL}\left(p \parallel q \right) = -\int{p(x)\cdot log\left(q(x)\right)}\,dx + \int{p(x)\cdot log\left(p(x)\right)}\,dx$$

$$D_{KL}\left(p \parallel q \right) = -\int{p(x)\cdot log\left(\frac{q(x)}{p(x)}\right)}\,dx$$

$$D_{KL}\left(p \parallel q \right) = \int{p(x)\cdot log\left(\frac{p(x)}{q(x)}\right)}\,dx$$

KL divergence allows to measure a distance between two distributions with the following properties:
* $$D_{KL}$$ is always positive (because $$H_{pq}>H_{p}$$):
$$\quad \quad D_{KL}\left(p \parallel q \right) \geq 0$$
* $$D_{KL}$$ is not symmetric:
$$\quad \quad D_{KL}\left(p \parallel q \right) \neq D_{KL}\left(q \parallel p \right)$$

&nbsp;

Here is a graph to intuit the second relationship, i.e. $$D_{KL}$$ is not symmetric.

![](/collections/images/vae/Kl_asymmetric.jpg)

Indeed, from the perspective of the purple distribution, the distance between points A and B appears to be high. However, from the perspective of the green distribution, the same distance appears to be moderate.

&nbsp;

>>$$D_{KL}$$ can thus be used to measure a distance between two distributions. Its is always positive and it is not symmetric.

&nbsp;

### Bayes theorem

Let’s assume a model where data $$x$$ are generated from a probability distribution depending on an unknown parameter $$z$$. Let’s also assume that we have a prior knowledge about the parameter $$z$$ that can be expressed as a probability distribution $$p\left(z\right)$$. Then, when data $$x$$ are observed, we can update the prior knowledge about this parameter using the Bayes theorem as follows:

![](/collections/images/vae/bayes_theorem.jpg)

&nbsp;

## **Variational inference**

### Key concept

In the VAE formalism, we first make the assumption that $$p(z)$$ is a standard Gaussian distribution (to ensure completeness) and that $$p(x/z)$$ is a Gaussian distribution whose mean is defined by a deterministic function $$f$$ of the variable $$z$$ and whose covariance matrix has the form of a positive constant $$c$$ that multiplies the identity matrix $$I$$. We will see that this last assumption allows to keep most of the information of the data structure in the reduced representations.

$$p(z) = \mathcal{N}(0,I)$$

$$p(x/z) = \mathcal{N}(f(z),cI)$$

The key concept around VAE is that we will try to optimize the computation of $$p(z/x)$$. Indeed, it can be demonstrated that the computation of $$p(z/x)$$ is often complicated and requires the use of approximation techniques such as ***variational inference***.

&nbsp;

>>In statistics, variational inference is a technique to approximate complex distributions. The idea is to set a parametrised family of distribution, usuall the family of Gaussians whose parameters are the mean and the covariance, and to look for the best approximation of the target distribution among this family. The best element in the family is one that minimise a given approximation error measurement, most of the time the KL divergence between approximation and target.

&nbsp;

In the VAE formalism, $$p(z/x)$$ is approximated by a Gaussian distribution $$q_x(z)$$ whose mean and covariance are defined by two functions $$g(x)$$ and $$h(x)$$. 

$$q_x(z) = \mathcal{N}\left(g(x),h(x)\right)$$


We thus have a family of candidates for variational inference and need to find the best approximation among this family by minimising the KL divergence between the approximation and the target $$p(z/x)$$. In other words, we are looking for the optimal $$g^*$$ and $$h^*$$ such that:

$$\left(g^*,h^*\right) = \underset{(g,h)}{\arg\min} \,\,\, D_{KL}\left(q_x(z) \parallel p(z/x) \right)$$


<!--Moreover, for simplification purposes, we will also try to project the input data in a Z space with a Gaussian probability density (i.e. $$p(z) = \mathcal{N}(0,I)$$). This will allow us to efficiently structure the latent space by concentrating the information close to the origin while avoiding holes. 

![](/collections/images/vae/vae_latent_space_gaussian.jpg)

>>This aspect of VAE can be seen as manifold learning-->

Let's now reformulate the KL divergence expression.


$$D_{KL}\left(q_x(z) \parallel p(z/x) \right) = - \int{q_x(z) \cdot log\left(\frac{p(z/x)}{q_x(z)}\right) \,dz}$$

By using the ***conditional probability*** relation:

$$p(z/x) = \frac{p(x,z)}{p(x)}$$

where $$p(x,z)$$ is the joint distribution of event $$x$$ and $$z$$, the above expression can be rewritten as:

$$D_{KL}\left(q_x(z) \parallel p(z/x) \right) = - \int{q_x(z) \cdot log\left(\frac{p(x,z)}{p(x) \cdot q_x(z)}\right) \,dz}$$

$$D_{KL}\left(q_x(z) \parallel p(z/x) \right) = - \int{q_x(z) \cdot \left[ log\left(\frac{p(x,z)}{q_x(z)}\right) + log\left(\frac{1}{p(x)}\right) \right] \,dz}$$

$$D_{KL}\left(q_x(z) \parallel p(z/x) \right) = - \int{q_x(z) \cdot log\left(\frac{p(x,z)}{q_x(z)}\right) \,dz} \,+\, log\left(p(x)\right) \cdot \underbrace{\int{q_x(z)\,dz}}_{=1}$$

$$D_{KL}\left(q_x(z) \parallel p(z/x) \right) \,+\, \mathcal{L} \,=\, log\left(p(x)\right)$$

where $$\mathcal{L}$$ is defined as the ***Evidence Lower BOund (ELBO)*** whose expression is given by:

$$\mathcal{L} = \int{q_x(z) \cdot log\left(\frac{p(x,z)}{q_x(z)}\right) \,dz}$$

&nbsp;

### Evidence lower bound

Let's take a closer look at the previous derived equation:

$$D_{KL}\left(q_x(z) \parallel p(z/x) \right) \,+\, \mathcal{L} \,=\, log\left(p(x)\right)$$

The following observations can be made:
* since $$0\leq p(x) \leq 1$$, $$log\left(p(x)\right) \leq 0$$

* since $$x$$ is the observation, $$log\left(p(x)\right)$$ is a fixed value

* by definition $$D_{KL}\left(q_x(z) \parallel p(z/x) \right) \geq 0$$

* since $$\mathcal{L} = -D_{KL}\left(q_x(z) \parallel p(x,z)\right)$$, $$\mathcal{L} \leq 0$$


The previous expression can thus be rewritten as follows:

$$\underbrace{D_{KL}\left(q_x(z) \parallel p(z/x) \right)}_{\geq 0} \,+\, \underbrace{\mathcal{L}}_{\leq 0} \,=\, \underbrace{log\left(p(x)\right)}_{\leq 0 \,\, \text{and fixed}}$$

>>At this point, it is important to remember that $$p(z/x)$$ is the unknown and that the goal is to find the best $$q_x(z)$$, i.e. the one that shall minimize $$D_{KL}\left(q_x(z) \parallel p(z/x) \right)$$.

&nbsp;

With the above observations, the following strategy can be implemented: by tweaking $$q_x(z)$$, we can seek to maximize the ELBO $$\mathcal{L}$$, which will imply the minimization of the KL divergence $$D_{KL}\left(q_x(z) \parallel p(z/x) \right)$$, and thus to find a distribution $$q_x(z)$$ that is close to $$p(z/x)$$. The table below provides an illustration of such a strategy. 

<style>
table th:first-of-type {
    width: 33%;
}
table th:nth-of-type(2) {
    width: 33%;
}
table th:nth-of-type(3) {
    width: 33%;
}
</style>

| $$\downarrow \, D_{KL}$$ | $$\uparrow \, \mathcal{L}$$ | $$log\left(p(x)\right)$$ |
| :---: | :---: | :---:|
| 4 | -8 | -4 |
| 3 | -7 | -4 |
| 2 | -6 | -4 |
| 1 | -5 | -4 |
| 0 | -4 | -4 |

&nbsp;

### ELBO reformulation

The ELBO $$\mathcal{L}$$ should be reformulated so to justify the loss involved in the VAE framework. The corresponding derivation is provided below.

$$\mathcal{L} = \int{q_x(z) \cdot log\left(\frac{p(x,z)}{q_x(z)}\right) \,dz}$$

$$\mathcal{L} = \int{q_x(z) \cdot log\left(\frac{p(x/z)\cdot p(z)}{q_x(z)}\right) \,dz}$$

$$\mathcal{L} = \int{q_x(z) \cdot \left[ log\left(p(x/z)\right) + log\left(\frac{p(z)}{q_x(z)}\right) \right] \,dz}$$

$$\mathcal{L} = \int{q_x(z) \cdot log\left(p(x/z)\right) \,dz} + \int{q_x(z) \cdot log\left(\frac{p(z)}{q_x(z)}\right) \,dz}$$

$$\mathcal{L} =  \mathbb{E}_{z\sim q_x} \left[log\left(p(x/z)\right)\right] - D_{KL}\left(q_x(z)\parallel p(z)\right)$$

$$\mathcal{L} =  \mathbb{E}_{z\sim q_x} \left[-\frac{\|x-f(z)\|^2}{2c}\right] - D_{KL}\left(q_x(z)\parallel p(z)\right)$$

&nbsp;

where $$\mathbb{E}_{z\sim q_x}$$ is the mathematical expectation with respect to $$q_x(z)$$. 

We are finally looking for:

$$\left(f^*,g^*,h^*\right) = \underset{(f,g,h)}{\arg\max} \,\,\, \left( \mathbb{E}_{z\sim q_x} \left[-\frac{\|x-f(z)\|^2}{2c}\right] - D_{KL}\left(q_x(z)\parallel p(z)\right) \right)$$

or 

$$\left(f^*,g^*,h^*\right) = \underset{(f,g,h)}{\arg\min} \,\,\, \left( \mathbb{E}_{z\sim q_x} \left[\frac{\|x-f(z)\|^2}{2c}\right] + D_{KL}\left(q_x(z)\parallel p(z)\right) \right)$$

&nbsp;

### From ELBO to VAE

Here is a summary of what have been done so far.

1. We want to estimate a non-linear projection $$p(z/x)$$ to go from an input space to a space of reduced dimension, and this through a probabilistic framework.

2. To do this, we introduced a third party parametric distribution $$q_x(z)$$ to estimate the target distribution $$p(z/x)$$.

3. We used the KL divergence metric which measures the proximity between the two distributions, the objective being to minimize this metric.

4. The minimization of the KL divergence leads to the maximization of the following ELBO equation:

<!--
<div style="background-color:#d7efd5; text-align:center; vertical-align: middle; padding:5px 0;">
$$\mathcal{L} =  \mathbb{E}_{z\sim q(z)} \left[log\left(p(x/z)\right)\right] - D_{KL}\left(q(z)||p(z)\right)$$
</div>
-->

$$\left(f^*,g^*,h^*\right) = \underset{(f,g,h)}{\arg\max} \,\,\, \left( \mathbb{E}_{z\sim q_x} \left[-\frac{\|x-f(z)\|^2}{2c}\right] - D_{KL}\left(q_x(z)\parallel p(z)\right) \right)$$


&nbsp;

The maximization of the above equation can be handled by the following graph.

![](/collections/images/vae/vae_final_step.jpg)

>>From this graph, we can see that the maximization of the ELBO equation can be handled by a decoder (first part) and an encoder (second part)!

&nbsp;

**Let's work on the decoder** 

Our goal is to output an instance $$\hat{x}$$ that is close to the input $$x$$. Since the decoder is a neural network, the link between $$z$$ and $$\hat{x}$$ is deterministic. We thus have 

$$p(x/z) \equiv p(x/\hat{x})$$

If $$p$$ is a Gaussian distribution, then 

$$p(x/\hat{x}) \propto \exp{\left(-\frac{\|x-\hat{x}\|^2}{\sigma^2}\right)}$$

$$log\left(p(x/\hat{x})\right) = -\|x-\hat{x}\|^2 + Cst$$

So the maximization of $$\mathbb{E}_{z\sim q(z)} \left[log\left(p(x/z)\right)\right]$$ can be obtained through the minimization of $$\|x-\hat{x}\|^2$$, which is the conventional reconstruction error!

>>If we make the assumption that $$\,p$$ follows a Bernouilli distribution, then we can demonstrate that the maximization of $$\,\mathbb{E}_{z\sim q(z)} \left[log\left(p(x/z)\right)\right]$$ leads to the minimization of the cross entropy function!

&nbsp;

**Let's work on the encoder** 

Our goal is to minimize $$D_{KL}\left(q(z/z)\parallel p(z)\right)$$.

In order to make the equation simpler and to structure the latent space, we first force $$p(z)$$ to follow a Gaussian distribution $$\mathcal{N}(0,I)$$. This is a strong choice of VAE formalism. The encoder should thus minimize the following loss:

$$D_{KL}\left(q(z/x)\parallel \mathcal{N}(0,I)\right)$$

A very important point here is that we must think in terms of probability function since we want to fit two distributions. In other words, the encoder must generate the parameters of the distribution that will generate the $$z$$ sample. 

>>The execution of the encoder for the same input $$x$$ will generate different $$z$$ samples whose values should be close to $$\mathcal{N}(0,I)$$.

&nbsp;

Given the needs for distribution modeling, the graph below shows the final network structure used in the VAE formalism.

![](/collections/images/vae/vae_final_representation.jpg)

The following equation is also used as a loss term:

$$\text{loss}=\|x-\hat{x}\|^2 \,+\, D_{KL}\left(\mathcal{N}\left(\mu_x,\sigma_x\right),\mathcal{N}\left(0,I\right)\right) $$

&nbsp;

### VAE network architecture



