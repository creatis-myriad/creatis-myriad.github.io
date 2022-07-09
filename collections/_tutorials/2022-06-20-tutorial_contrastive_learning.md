---
layout: post
title:  "The beauty of contrastive learning: a new step toward efficient unsupervised learning"
author: 'Olivier Bernard'
date:   2022-07-09
categories: contrastive learning, unsupervised learning
---

# Notes

* Here is a link to an interesting video to better understand the contrastive learning paradigm: [video](https://www.youtube.com/watch?v=IEiytaXnggI&ab_channel=DeepMindELLISUCLCSMLSeminarSeries)

&nbsp;

- [**Introduction**](#introduction)
- [**SimCLR framework**](#simclr-framework)
  - [Overall scheme](#overall-scheme)
  - [Contrastive loss](#contrastive-loss)
  - [Training tricks](#training-tricks)
- [**Transformer encoder**](#transformer-encoder)
  - [Norm step](#norm-step)
  - [MLP step](#mlp-step)  
  - [Multi-Head Attention block](#multi-head-attention-block)  
  - [Self-Attention module](#self-attention-module)    

&nbsp;

## **Introduction**
The goal of unsupervised-based model is to learn effective representation without human supervision. Most mainstream approaches fall into one of the two classes: generative or discriminative.  

>> Generative approaches learn to generate samples in the input space, such as GAN [[1]](https://arxiv.org/abs/1406.2661?context=cs). 
	
>> Discriminative approaches learn representation by performing pretext tasks where both the input and label are derived from an unlabeled dataset [[2]](https://arxiv.org/abs/1803.07728).

Discriminative approaches based on contrastive learning in the latent space have recently shown great promise, achieving state-of-the-art results. The purpose of this tutorial is to introduce the key steps / concepts of this formalism, as well as the results that follow from it. 

&nbsp;

## **SimCLR framework**

To illustrate the contrastive learning mechanism, we will study the now famous simple framework for contrastive learning of visual representation, called [SimCLR](https://arxiv.org/pdf/2002.05709.pdf). The corresponding architecture is given below:

![](/collections/images/contrastive_learning/simCLR_overview.jpg)

&nbsp;

### Overall scheme

As illustrated in the figure above, SimCLR comprises four major components: 

* A stochasitc data augmentation module that transforms any given data sample $$x$$ randomly resulting in two correlated views of the same sample, denoted as $$\tilde{x}_i$$ and $$\tilde{x}_j$$ and considered as positive pair. **Random cropping**, **random color distortions** and **random Gaussian blur** are classically combined to transform data samples. 

* A neural network base encoder $$f(\cdot)$$ that extracts representation vectors from augmented data samples. This step can be realized by any standard network architecture, such as ResNet. This step is modeled as follows:

$$h_i = f(\tilde{x}_i)=\texttt{ResNet}(\tilde{x}_i)$$

&nbsp; &nbsp; &nbsp; &nbsp; where $$h_i \in \mathbb{R}^{d}$$ is the output vector after average pooling layer. 

* A simple neural network projection head $$g(\cdot)$$ that maps representations to the space where the contrastive loss is applied. A simple MLP with one hidden layer is  used for this step which outputs a vector $$z_i \in \mathbb{R}^{d'}$$ with $$d'=128$$. This allows to project the representation to a 128-dimensional latent space.

* A contrastive loss function defined for a contrastive prediction task. Given a set of $$\{\tilde{x}_k\}$$ including a positive pair of samples $$\tilde{x}_i$$ and $$\tilde{x}_j$$, the contrastive prediction task aims to identify $$\tilde{x}_j$$ in $$\{\tilde{x}_k\}_{k \neq i}$$ for a given $$\tilde{x}_i$$.

&nbsp;

### Contrastive loss

* A minibatch of $$N$$ samples is randomly selected.

* Each sample is used to create a pair of augmented examples, resulting in $$2N$$ data points.

* For each minibatch, one pair is considered as positive and the others $$2(N-1)$$ as negative examples. 

* The cosine similarity between two samples $$u$$ and $$v$$ is computed from the conventional dot product $$sim(u,v)=u^{T}v/\left(\|u\|\|v\|\right)$$ . The corresponding value varies between 0 (when u and v are orthogonal) and 1 (when u and v are aligned).

* The following loss function for a positive pair of samples $$(i,j)$$ is defined as:

$$l(i,j)=-\log{\left(\frac{ \exp\left(sim(z_i,z_j)/\tau\right) }{ \sum_{k=1}^{2N}{\mathbb{1}_{[k \neq i]}\exp\left(sim(z_i,z_k)/\tau\right) } }\right)}$$

&nbsp; &nbsp; &nbsp; &nbsp; where $$\mathbb{1}_{[k \neq i]} \in [0,1]$$ is an indicator function evaluating to 1 iff $$k \neq i$$ and $$\tau$$ is a parameter.

>> Since both the numerator and denominator involve exponential terms, the values inside the $$-\log(\cdot)$$ function varie between 0 and 1. As a reminder, we display above the corresponding curve. From this figure, on can see that the loss function will tend to its minimum when the numerator and the denominator will be close, i.e. when the set $$\{\exp\left(sim(z_i,z_k)/\tau\right)\}_{[k \neq (i,j)]}$$ will be as low as possible and when $$\exp\left(sim(z_i,z_j)/\tau\right)$$ will be high, making the two points $$z_i$$ and $$z_j$$ to be as close as possible and in the meanwhile the other points to be as orthogonal as possible to these two points. The minimization of the contrastive loss thus allows to structure the latent space according to visual representation.

<p align = "center"><img src ="/collections/images/contrastive_learning/minus_log.jpg" alt="Trulli" style="width:40%"></p>

* The final loss is computed across all positive pairs, both $$(i,j)$$ and $$(j,i)$$, in a mini-batch as follows:

$$ \mathcal{L} = \frac{1}{2N} \sum_{k=1}^{N}{ \left[ l(2k-1,2k) + l(2k,2k-1) \right] } $$

&nbsp;

### Training tricks

* The model was trained at batch size N=4096 for 100 epochs.

* The training was stabilized thanks to the use of the [LARS](https://arxiv.org/abs/1708.03888) optimizer for all batch sizes.

* The model was trained with 128 TPU v3 core.

* It took around 1.5 hours to train a SimCRL model composed of ResNet-50 with a batch size of 4096 for 100 epochs. 

&nbsp;

## **Tokenization process**
The first step is to convert the input data into a sequence of tokens, *i.e.* a sequence of vectors of dimensions $$\mathbb{R}^{1 \times (P^2 C)}$$, with $$P^2 C$$ being the length of each flattened patch. This step is called the tokenization process and involves a simple linear projection (*i.e.* a multiplication with a matrix of dimensions $$\mathbb{R}^{(P^2 C) \times D}$$) and a position embedding step which encodes spatial information. The same linear projection and position embedding operation are shared to encode each patch. This process is modeled as follows: 

$$z_0 = [x_{class}; \, x^1_p\mathbf{E}; \, x^2_p\mathbf{E}; \, \cdots; \, x^N_p\mathbf{E}] + \mathbf{E}_{pos}$$

with <br>
&nbsp; &nbsp; &nbsp; &nbsp;$$\mathbf{E}$$ being the linear projection matrix of size $$\mathbf{E} \in \mathbb{R}^{(P^2 C) \times D}$$ <br>
&nbsp; &nbsp; &nbsp; &nbsp;$$\mathbf{E}_{pos}$$ being the output of the position embedding operation with dimensions $$\mathbf{E}_{pos} \in \mathbb{R}^{(N+1) \times D}$$

&nbsp;

The figure below illustrates the tokenization process used in ViT.

![](/collections/images/transformers/vit_tokenization.jpg)

&nbsp;

## **Transformer encoder**

The figure below presents an overview of the encoding layer of a transformer. It is composed of two main steps.

![](/collections/images/transformers/vit_encoding_layer.jpg)

&nbsp;

The goal of the first step is to compute attention maps between tokens. The output corresponds to $$z^{'}_l$$ which is modeled as follows:

$$z^{'}_l = \text{MHA}(\text{LN}(z_{l-1})) + z_{l-1} \quad \quad \quad \quad l=1 \cdots L$$

The goal of the second step is to introduce non-linearities to compute more relevant sequence of tokens. The output corresponds to $$z_l$$ which is modeled as follows:

$$z_l = \text{MLP}(\text{LN}(z^{'}_{l})) + z^{'}_{l} \quad \quad \quad \quad l=1 \cdots L$$

&nbsp;

### Norm step

Normalization was inserted into the encoding layer to control the dynamics of the token values before each of the two key steps. The corresponding diagram is given below:

![](/collections/images/transformers/vit_norm_step.jpg)

&nbsp;

### MLP step

The diagram of the $$\text{MLP}$$ procedure is given below:

![](/collections/images/transformers/vit_mlp_step.jpg)

&nbsp;

### Multi-Head Attention block

The Multi-Head Attention (MHA) block is the key element of the encoding layer. It is based on the "QKV" paradigm, but what does "QKV" mean ? 
Before answering this question, let's zoom in on the MHA block and get an overview of its structure.

![](/collections/images/transformers/vit_mha_overview.jpg)

From this figure, we can see that the key element is the Self-Attention module which outputs $$k$$ head matrices of size $$\mathbb{R}^{(N+1) \times D_h}$$, where $$D_h$$ is usually computed as $$D_h=D/k$$. These head matrices contain useful information computed from attention mechanisms. The second part of the MHA block uses a linear projection to optimally merge the different heads and to output a new sequence of tokens of the same size as the input matrix, *i.e.* $$\mathbb{R}^{(N+1) \times D}$$. This operation is modeled as:

$$ \text{MHA}(z) = \left[ \text{SA}_1(z), \, \text{SA}_2(z), \cdots, \, \text{SA}_k(z) \right] \cdot \mathbf{U}_{msa} \quad \quad \quad \quad \mathbf{U}_{msa} \in \mathbb{R}^{(k D_h) \times D}$$

where $$\text{SA}_i(z)$$ represents the output of Head $$i$$ ($$\text{SA}$$ stands for Self Attention). Now it is time to investigate the Self-Attention module !

&nbsp;

### Self-Attention module

The Self-Attention module is the core element of the MHA block. The figure below provides an overview of such a module.

![](/collections/images/transformers/vit_self_attention_module.jpg)

&nbsp;

It involves the generation of 3 matrices $$\mathbf{Q}$$, $$\mathbf{K}$$, $$\mathbf{V}$$ of size $$\mathbb{R}^{(N+1) \times D_h}$$ computed from three different linear projection matrices $$\mathbf{U}_{qkv} \in \mathbb{R}^{D \times D_h}$$. This step is modeled as follows:

$$ [\mathbf{Q}, \mathbf{K}, \mathbf{V}] = z \mathbf{U}_{qkv} \quad \quad \quad \quad \mathbf{U}_{qkv} \in \mathbb{R}^{D \times D_h}$$

&nbsp;

$$\mathbf{Q}$$ and $$\mathbf{K}$$ are used to create a self-attention matrix $$\mathbf{A} \in \mathbb{R}^{(N+1) \times (N+1)}$$. Since this matrix is computed from scalar products, it expresses the proximity between token $$i$$ and all token $$j$$. The Softmax operation is applied to each row of $$\mathbf{A}$$ individually to ensure that the sum of the elements of each row is equal to $$1$$. The corresponding operation is:

$$ \mathbf{A} = softmax\left( \frac{\mathbf{Q} \cdot \mathbf{K}^T}{\sqrt{D_h}} \right)$$

&nbsp;

Finally, this self-attention matrix $$\mathbf{A}$$ is used to compute the final Head matrix $$\text{SA}(z) \in \mathbb{R}^{(N+1)\times D_h}$$ whose rows correspond to a weighted sum of the tokens obtained in the $$\mathbf{V}$$ matrix. This is done by using the following simple multiplication operation:

$$\text{SA}(z) = \mathbf{A} \cdot \mathbf{V}$$


