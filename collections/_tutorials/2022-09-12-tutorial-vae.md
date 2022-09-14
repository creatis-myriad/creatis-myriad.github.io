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
- [**Fondamental knowledge**](#fondamental-knowledge)
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
Let's start with the basic representation of an auto-encoder

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

## **Fondamental knowledge**
TODO

&nbsp;

### Information quantification

TODO

&nbsp;

### Entropy

TODO

&nbsp;

### Kullback-Liebler divergence

TODO



