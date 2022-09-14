---
layout: post
title:  "The variational autoencoder paradigm demystified"
author: 'Olivier Bernard'
date:   2022-09-12
categories: autoencoder, encoder, decoder, vae
---

# Notes

* Here are links to four video that I used to create this tutorial: [video1](https://www.youtube.com/watch?v=4toWtb7PRH4), [video2](https://www.youtube.com/watch?v=uKxtmkfeuxg), [video3](https://www.youtube.com/watch?v=BxkZcS1pLpw), [video4](https://www.youtube.com/watch?v=uaaqyVS9-rM)

&nbsp;

- [**Intuition**](#intuition)
- [**Fondamental concepts**](#fondamental-concepts)
  - [Information quantification](#information-quantification)
  - [Entropy](#entropy)
  - [Kullback-Liebler divergence](#kullback-liebler-divergence)    
- [**Variational inference**](#variational-inference)
  - [Key concept](#key-concept)
  - [Lower bound](#lower-bound)
  - [Lower bound reformulation](#lower-bound-reformulation)  
  - [From lower bound to vae](#from-lower-bound-to-)    

&nbsp;

## **Intuition**
Let's start with the basic representation of an autoencoder

![](/collections/images/vae/autoencoder.jpg)

Autoencoders belong to the family of dimension reduction methods. This method takes as input a vector $$\mathbf{x} \in \mathbb{R}^N$$ and outputs a closed vector $$\mathbf{\hat{x}} \in \mathbb{R}^N$$ with the restriction of passing through a space with reduced dimensionality $$Z \in \mathbb{R}^M$$. This is usually achieved through the minimization of the $$L_2$$ norm function: $$\lVert \mathbf{x} - \mathbf{\hat{x}} \rVert^2$$.

$$\mathbf{e}$$ and $$\mathbf{d}$$ are two different networks that model the (non linear) projections from the input space $$X$$ to the latent space $$Z$$ in both directions. 

Let us now consider the simple case where $$\mathbf{e}$$ and $$\mathbf{d}$$ correspond to two single-layer networks without any non-linearity. The corresponding autoencoder can be represented as follows:

![](/collections/images/vae/simplified_autoencoder.jpg)

where $$\mathbf{e} \in \mathbb{R}^{M \times N}$$ and $$\mathbf{d} \in \mathbb{R}^{N \times M}$$ are two linear projection matrices. In the particular case where $$\mathbf{e} = \mathbf{U}^T$$ and $$\mathbf{d} = \mathbf{U}$$, the autoencoder expressions can be written as:

$$\mathbf{z} = \mathbf{U}^T\mathbf{x} \quad\quad \text{and} \quad\quad \mathbf{\hat{x}} = \mathbf{U}\mathbf{z}=\mathbf{U}\mathbf{U}^T\mathbf{x}$$

This corresponds to the well know PCA (Principal Component Analysis) paradigm. 

>>Autoencoders can thus be seen as a generalization of the dimensionality reduction PCA formalism by evolving more complex projection operations defined through $$\mathbf{e}$$ and $$\mathbf{d}$$ networks.

&nbsp;


VAE thus offers two extremely interesting opportunities:
* the mastery of the encoder allows to optimize the projection operation $$p(z/x)$$ to a latent space with reduced dimensionality for interpretation purposes. This corresponds to ***manifold learning paradigm***.

![](/collections/images/vae/encoder_illustration.jpg)

* the mastery of the decoder allows to optimize the projection operation $$p(x/z)$$ for the generation of data with a complex distribution. This corresponds to ***generative model framework***.

![](/collections/images/vae/decoder_illustration.jpg)

>>In the rest of this tutorial, we will see how the vae formalism allows to optimize these two tasks through the theory of variational inference.

&nbsp;

## **Fondamental concepts**

### Information quantification

Information $$I$$ can be quantified through the following expression:

$$I = -log(p(x))$$

with $$x$$ being an event and $$p(x)$$ the probability of this event. 

>>From this equation, one can see that when the probability of an event is high (close to $$1$$), the corresponding information is low, which makes sense. For instance, the probability that the weather will be hot in France during summer is very high, so this sentence does not provide any useful information in a conversation


&nbsp;

### Entropy

Entropy $$H$$ corresponds to the ***average information of a process***. Its expression can be naturally written as:

$$H = -\sum_{i=1}^{N}{p(x_i)\cdot log\left(p(x_i)\right)} \quad \quad \quad \text{or} \quad \quad \quad H = -\int{p(x)\cdot log\left(p(x)\right)}\,dx$$

&nbsp;

### Kullback-Liebler divergence

The Kullback-Liebler (KL) divergence allows to ***measure the distance between two distributions*** through the use of relative entropy concepts. Let's define the following expressions:

$$H_{p} = -\int{p(x)\cdot log\left(p(x)\right)}\,dx$$

$$H_{pq} = -\int{p(x)\cdot log\left(q(x)\right)}\,dx$$

where $$H_{p}$$ corresponds to the entropy relative to the distribution $$p(x)$$ and $$H_{pq}$$ the average information brings by $$g(x)$$ but weighted by $$p(x)$$. From these notations, the KL divergence $$D_{KL}$$ can be expressed as:

$$D_{KL}\left(p || q \right) = H_{pq} - H_{p}$$

$$D_{KL}\left(p || q \right) = -\int{p(x)\cdot log\left(q(x)\right)}\,dx + \int{p(x)\cdot log\left(p(x)\right)}\,dx$$

$$D_{KL}\left(p || q \right) = -\int{p(x)\cdot log\left(\frac{q(x)}{p(x)}\right)}\,dx$$

$$D_{KL}\left(p || q \right) = \int{p(x)\cdot log\left(\frac{p(x)}{q(x)}\right)}\,dx$$

KL divergence allows to measure a distance between two distributions with the following properties:
* $$D_{KL}$$ is always positif:
$$\quad \quad D_{KL}\left(p || q \right) \geq 0$$
* $$D_{KL}$$ is not symmetric:
$$\quad \quad D_{KL}\left(p || q \right) \neq D_{KL}\left(q || p \right)$$

&nbsp;

Here is a graph to intuit the second relationship, i.e. $$D_{KL}$$ is not symmetric.

![](/collections/images/vae/Kl_asymmetric.jpg)

Indeed, from the perspective of the purple distribution, the distance between points A and B appears to be high. However, from the perspective of the green distribution, the same distance appears to be moderate.

&nbsp;

>>$$D_{KL}$$ can thus be used to measure a distance between two distributions. Its is always positive and it is not symmetric.

&nbsp;

## **Variational inference**

### Key concept

The key concept around VAE is that we will try to optimize the learning of the non-linear projection operation $$p(z/x)$$ thanks to the variational inference formalism. Indeed, variational inference allows to approximate a complex probability (in our case $$p(z/x)$$) by a simpler model thanks to the use of the KL divergence tool.

Moreover, for simplification purposes, we will also try to project the input data in a Z space with a Gaussian probability density (i.e. $$p(z) = \mathcal{N}(0,1)$$). This will allow us to efficiently structure the latent space by concentrating the information close to the origin while avoiding holes. 

![](/collections/images/vae/vae_latent_space_gaussian.jpg)

>>This aspect of VAE can be seen as manifold learning

&nbsp;

1. Let's say we have a distribution $$p(z/x)$$ that we don't know

2. We use a new (simpler) distribution $$q(z)$$ to estimate $$p(z/x)$$

3. The KL divergence is then exploited to measure the quality in terms of distribution fitting !

$$D_{KL}\left(q(z) || p(z/x) \right) = - \int{q(z) \cdot log\left(\frac{p(z/x)}{q(z)}\right) \,dz}$$

By using the ***conditional probability*** relation:

$$p(z/x) = \frac{p(x,z)}{p(x)}$$

where $$p(x,z)$$ is the joint distribution of event $$x$$ and $$z$$, the above expression can be rewritten as:

$$D_{KL}\left(q(z) || p(z/x) \right) = - \int{q(z) \cdot log\left(\frac{p(x,z)}{p(x) \cdot q(z)}\right) \,dz}$$

$$D_{KL}\left(q(z) || p(z/x) \right) = - \int{q(z) \cdot \left[ log\left(\frac{p(x,z)}{q(z)}\right) + log\left(\frac{1}{p(x)}\right) \right] \,dz}$$

$$D_{KL}\left(q(z) || p(z/x) \right) = - \int{q(z) \cdot log\left(\frac{p(x,z)}{q(z)}\right) \,dz} \,+\, log\left(p(x)\right) \cdot \underbrace{\int{q(z)\,dz}}_{=1}$$

$$D_{KL}\left(q(z) || p(z/x) \right) \,+\, \mathcal{L} \,=\, log\left(p(x)\right)$$

where $$\mathcal{L}$$ is defined as the ***lower bound*** whose expression is given by:

$$\mathcal{L} = \int{q(z) \cdot log\left(\frac{p(x,z)}{q(z)}\right) \,dz}$$

&nbsp;

### Lower bound

Let's take a closer look at the previous derived equation:

$$D_{KL}\left(q(z) || p(z/x) \right) \,+\, \mathcal{L} \,=\, log\left(p(x)\right)$$

The following observations can be made:
* since $$0\leq p(x) \leq 1$$, $$log\left(p(x)\right) \leq 0$$

* since $$x$$ is the observation, $$log\left(p(x)\right)$$ is a fixed value

* by definition $$D_{KL}\left(q(z) \| p(z/x) \right) \geq 0$$

* since $$\mathcal{L} = -D_{KL}\left(q(z) \| p(x,z)\right)$$, $$\mathcal{L} \leq 0$$


The previous expression can thus be rewritten as follows:

$$\underbrace{D_{KL}\left(q(z) || p(z/x) \right)}_{\geq 0} \,+\, \underbrace{\mathcal{L}}_{\leq 0} \,=\, \underbrace{log\left(p(x)\right)}_{\leq 0 \,\, \text{and fixed}}$$

>>At this point, it is important to remember that $$p(z/x)$$ is the unknown and that we have the possibility to play with the expression of $$q(z)$$ to minimize $$D_{KL}\left(q(z) \| p(z/x) \right)$$.

&nbsp;

With the above observations, the following strategy can be implemented: by playing with $$q(z)$$, we can seek to maximize the lower bound $$\mathcal{L}$$, which will imply the minimization of the KL divergence $$D_{KL}\left(q(z) \| p(z/x) \right)$$, and thus to find a distribution $$q(z)$$ which will approach $$p(z/x)$$. The table below provides an illustration of such a strategy. 

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

### Lower bound reformulation

The lower bound $$\mathcal{L}$$ should be reformulated so to justify the loss involved in the VAE framework. The corresponding derivation is provided below.

$$\mathcal{L} = \int{q(z) \cdot log\left(\frac{p(x,z)}{q(z)}\right) \,dz}$$

$$\mathcal{L} = \int{q(z) \cdot log\left(\frac{p(x/z)\cdot p(z)}{q(z)}\right) \,dz}$$

$$\mathcal{L} = \int{q(z) \cdot \left[ log\left(p(x/z)\right) + log\left(\frac{p(z)}{q(z)}\right) \right] \,dz}$$

$$\mathcal{L} = \int{q(z) \cdot log\left(p(x/z)\right) \,dz} + \int{q(z) \cdot log\left(\frac{p(z)}{q(z)}\right) \,dz}$$

$$\mathcal{L} =  \mathbb{E}_{z\sim q(z)} \left[log\left(p(x/z)\right)\right] - D_{KL}\left(q(z)||p(z)\right)$$

&nbsp;

### From lower bound to vae



