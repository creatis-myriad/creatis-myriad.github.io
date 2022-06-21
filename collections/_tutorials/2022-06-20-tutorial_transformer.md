---
layout: post
title:  "Transformers demystified"
author: 'Olivier Bernard'
date:   2022-06-20
categories: transformer, encoder
---

- [**Introduction**](#introduction)
- [**Tokenization process**](#tokenization-process)
- [**Transformer encoder**](#transformer-encoder)
  - [Norm step](#norm-step)
  - [MLP step](#mlp-step)  
  - [Multi-Head Attention block](#multi-head-attention-block)  

&nbsp;

## **Introduction**
Transformers paradigm has been first successfully applied in Natural Langage Processing (NLP) and is currently highly investigated for image processing. 

The basic concept of transformers lies in the generation of self attention maps to guide the decision making mechanism of the underlying architecture.

In order to illustrate the bascis of transformer, we will open the black box of the famous [Vision Transformer (ViT)](https://creatis-myriad.github.io/2022/06/01/VisionTransformer.html) method. The overall architecture is given below:

![](/collections/images/transformers/vit_overview.jpg)

&nbsp;

## **Tokenization process**
The first step performed by the transformers is to convert the input data into a sequence of tokens, i.e. a sequence of vectors of dimensions $$\mathbb{R}^{1 \times (P^2 \cdot C)}$$, with $$P^2 C$$ being the length of each flattened patch. This step is referred to as the tokenization process and involves a simple linear projection (i.e. a multiplication with a matrix of dimensions $$\mathbb{R}^{(P^2 \cdot C) \times D}$$ and a position embedding step with encodes spatial information. The same linear projection and position embedding are shared to encode each patch. This process is modeled as follows: 

$$z_0 = [x_{class}; \, x^1_p\mathbf{E}; \, x^2_p\mathbf{E}; \, \cdots; \, x^N_p\mathbf{E}] + \mathbf{E}_{pos}$$

with <br>
&nbsp; &nbsp; &nbsp; &nbsp;$$\mathbf{E}$$ being the linear projection matrix of size $$\mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}$$ <br>
&nbsp; &nbsp; &nbsp; &nbsp;$$\mathbf{E}_{pos}$$ being the output of the position embedding operation with dimensions $$\mathbf{E}_{pos} \in \mathbb{R}^{(N+1) \times D}$$


The figure below illustrates the tokenization process used in ViT.

![](/collections/images/transformers/vit_tokenization.jpg)

&nbsp;

## **Transformer encoder**

The figure below presents an overview of the encoding layer of transformers. It is composed of two main steps.

![](/collections/images/transformers/vit_encoding_layer.jpg)

&nbsp;

The goal of the first step is to compute attention maps between tokens. The output corresponds to $$z^{'}_l$$ which is modeled as follows:

$$z^{'}_l = MSA(LN(z_{l-1})) + z_{l-1} \quad \quad \quad \quad l=1 \cdots L$$

The goal of the second step is to introduce non-linearities to compute more relevant sequence of tokens. The output corresponds to $$z_l$$ which is modeled as follows:

$$z_l = MLP(LN(z^{'}_{l})) + z^{'}_{l} \quad \quad \quad \quad l=1 \cdots L$$

&nbsp;

### Norm step

Normalization was inserted into the encoding layer to control the dynamics of the token values before each of the two key steps. The corresponding diagram is given below:

![](/collections/images/transformers/vit_norm_step.jpg)

&nbsp;

### MLP step

The diagram of the MLP procedure is given below:

![](/collections/images/transformers/vit_mlp_step.jpg)

&nbsp;

### Multi-Head Attention block

The Multi-Head Attention (MHA) block is the key element of the encoding layer. It is based on the "qkv" paradigm, but what does "qkv" mean ? 
Before answering to this question, let's have a zoom to the MHA block and have an overview of the structure.







