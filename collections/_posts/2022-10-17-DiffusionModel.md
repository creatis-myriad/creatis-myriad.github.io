---
layout: review
title: "Deep Unsupervised Learning using Nonequilibrium Thermodynamics"
author: "Sophie Carneiro Esteves"
cite:
    authors: "Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan and Surya Ganguli"
    title:   "Deep Unsupervised Learning using Nonequilibrium Thermodynamics"
    venue:   "International Conference on Machine Learning"
pdf: "http://proceedings.mlr.press/v37/sohl-dickstein15.pdf"
---
<style> {text-align: justify}</style>


# Notes
- Code is available on GitHub: [https://github.com/Sohl-Dickstein/
Diffusion-Probabilistic-Models](https://github.com/Sohl-Dickstein/
Diffusion-Probabilistic-Models)

# Introduction
The authors propose a new probalistic model that aims to link a known distribution $$ \pi(.) $$ to another one $$ q(.) $$ based on diffusion. To achieve it, they iteratively added small amount of noise on the target distribution until it is converted into the known distribution. This process can be seen has a Markov chain that can be learn reversly. 


# Highlights
- Flexible and Tractable method
- Possibility to multiply distributions (ex : inpainting)


# Diffusion probalistic model

> A Markov chain or Markov process is a stochastic model describing a sequence of possible events in which the probability of each event depends only on the state attained in the previous event. [^1]

A diffusion probalistic model consists in building a markov chain which allows to switch from a simple known distribution $$ \pi(.) $$ ( ex :  a gaussian distribution ) to target distribution  $$ q(.) $$ (ex : a kind of image distribution).
To obtain that chain they start from the target distribution and add small perturbations a thousand of times (eq. time steps) until we obtain the known distribution. This is the **forward trajectory**. 

#### Forward trajectory

Let 

- $$q(\mathbf{x}^{(0)})$$ be the data distribution
- $$T_{\pi}(\mathbf{y}\mid\mathbf{y}';\beta)$$ be a Markov diffusion kernel
- $$\beta_t$$ be a diffusion rate

We can define $$q\left(\mathbf{x}^{(t)} \mid \mathbf{x}^{(t-1)}\right)$$ with $$t$$ a time step in the Markov chain such as : 


$$
q\left(\mathbf{x}^{(t)} \mid \mathbf{x}^{(t-1)}\right)=T_\pi\left(\mathbf{x}^{(t)} \mid \mathbf{x}^{(t-1)} ; \beta_t\right)
$$

Let's assume that at $$T$$ steps, $$q(\mathbf{x}^{(0...T)})$$ is converted into the simple distribution $$ \pi(.) $$.
We can define anatycally $$q(\mathbf{x}^{(0...T)})$$ such as: 

$$
q\left(\mathbf{x}^{(0 \cdots T)}\right)=q\left(\mathbf{x}^{(0)}\right) \prod_{t=1}^T q\left(\mathbf{x}^{(t)} \mid \mathbf{x}^{(t-1)}\right) = \pi (\mathbf{x}^T) 
$$


However, the interests in the method is to be able to convert a simple distribution $$\pi(.)$$ into an element included in  the unknown  $$ q(.) $$. That is the **reverse trajectory**. 

#### Reverse diffusion

The Markov chain has been analytically created, therefore the forward trajectory form is known. As changes applied from a time step to another are small, diffusion kernels have the same functional form in both trajectories. Thus we can define : 

$$
p\left(\mathbf{x}^{(T)}\right)=\pi\left(\mathbf{x}^{(T)}\right)
$$

$$
p\left(\mathbf{x}^{(0 \cdots T)}\right)=p\left(\mathbf{x}^{(T)}\right) \prod_{t=1}^T p\left(\mathbf{x}^{(t-1)} \mid \mathbf{x}^{(t)}\right)
$$

| *Figure 1: Scheme of the method and representation of its Markov chain proposed for denoising diffusion Probalistic Model [^2]*|
|:----------------------------------------------------------------------------:|
|![](/collections/images/DiffusionModel/diffusion_model-markov.png)|


#### Training
During the training, the purpose is to learn the parameters of the diffusion kernels. Here, we will present only the case where the known distribution is gaussian. Associated formulas are given in the following table. 

| Distribution and kernels | Formulas          | 
| :---------------: |:---------------:|
|  $$\pi (\mathbf{x}^T) $$ |  $$\mathcal{N}\left(\mathbf{x}^{(T)} ; \mathbf{0}, \mathbf{I}\right)$$| 
| $$q\left(\mathbf{x}^{(t)} \mid \mathbf{x}^{(t-1)}\right)$$  |     $$\mathcal{N}\left(\mathbf{x}^{(t)} ; \mathbf{x}^{(t-1)} \sqrt{1-\beta_t}, \mathbf{I} \beta_t\right)$$     | 
| $$p\left(\mathbf{x}^{(t-1)} \mid \mathbf{x}^{(t)}\right)$$  |     $$\mathcal{N}\left(\mathbf{x}^{(t-1)} ; \mathbf{f}_\mu\left(\mathbf{x}^{(t)}, t\right), \mathbf{f}_{\Sigma}\left(\mathbf{x}^{(t)}, t\right)\right)$$     |   


>$$\mathbf{f}_\mu\left(\mathbf{x}^{(t)}, t\right)$$, $$\mathbf{f}_{\Sigma}\left(\mathbf{x}^{(t)}, t\right)$$ and $$\beta_{1...T}$$ are learned.

The probability the generative model assigns to the data is 

$$p\left(\mathbf{x}^{(0)}\right)=\int d \mathbf{x}^{(1 \cdots T)} p\left(\mathbf{x}^{(0 \cdots T)}\right)$$




### Multiple distributions

# Data
- swiss roll
- binary heartbeat distribution
- MNIST
- CIFAR-10
- Dead Leaf Images
- Bark Texture Images

# Results


# Conclusions

It works well and really interesting


# References

[^1]:  [https://en.wikipedia.org/wiki/Markov_chain](https://en.wikipedia.org/wiki/Markov_chain)

[^2]: Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *Advances in Neural Information Processing Systems*
