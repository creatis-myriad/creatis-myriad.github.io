---
layout: post
title:  "Transformers demystified"
author: 'Olivier Bernard'
date:   2022-06-20
categories: transformer, encoder
---

- [**Introduction**](#introduction)
- [**Tokenization process**](#tokenization-process)


&nbsp;

## **Introduction**
Transformers paradigm has been first successfully applied in Natural Langage Processing (NLP) and is currently highly investigated for image processing. 

The basic concept of transformers lies in the generation of self attention maps to guide the decision making mechanism of the underlying architecture.

In order to illustrate the bascis of transformer, we will open the black box of the famous [Vision Transformer (ViT)](https://creatis-myriad.github.io/2022/06/01/VisionTransformer.html) method. The overall architecture is given below:

![](/collections/images/transformers/vit_overview.jpg)

&nbsp;

## **Tokenization process**
The first step performed by the transformers is to convert the input data into a sequence of tokens, i.e. a sequence of vectors of dimensions $$\mathbb{R}^{1 \times (P^2 \cdot C)}$$, with $$P^2 \cdot C$$ being the length of each flattened patch. This step is referred to as the tokenization process and involves a simple linear projection (i.e. a multiplication with a matrix of dimensions $$\mathbb{R}^{(P^2 \cdot C) \times D}$$ and a position embedding step with encodes spatial information. The same linear projection and position embedding are shared to encode each patch. This process is modeled as follow: 

$$z_0 = [x_{class}; \, x^1_p\mathbf{E}; \, x^2_p\mathbf{E}; \, \cdots; \, x^N_p\mathbf{E}] + \mathbf{E}_{pos}$$.

with <br>
&nbsp; &nbsp; &nbsp; &nbsp;$$\mathbf{E}$$ being the linear projection matrix of size $$\mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}$$. <br>
&nbsp; &nbsp; &nbsp; &nbsp;$$\mathbf{E}_{pos}$$ being the output of the position embedding operation with dimensions $$\mathbf{E}_{pos} \in \mathbb{R}^{(N+1) \times D}$$.


The figure below illustrates the tokenization process used in ViT.

![](/collections/images/transformers/vit_tokenization.jpg)



