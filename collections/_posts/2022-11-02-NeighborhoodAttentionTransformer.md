---
layout: review
title: "Neighborhood Attention Transformer"
tags: deep-learning CNN transformer segmentation classification object-detection attention
author: "Pierre Rougé"
cite:
    authors: "Ali Hassani, Steven Walton, Jiachen Li, Shen Li, Humphrey Shi"
    title:   "Neighborhood Attention Transformer"
pdf: "https://arxiv.org/abs/2204.07143v1"
---

# Notes

* Code is available on [github](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer)
* This work was done by the same team that did the Compact Convolutional Transformer (CCT) reviewed in this [post](https://creatis-myriad.github.io./2022/06/13/CompactConvolutionalTransformer.html)
* So they use the same method of Convolutional Tokenization

# Highlights

* Similar to Swin Transformer the idea is to reduce the computational cost of the attention mechanism 
* The authors introduce the Neighborhood Attention (NA) and the Neighborhood Attention Transformer (NAT)
* With the Neighborhood Attention the attention is only compute on a neighborhood around each token
* This method not only allow to reduce the computational cost of the attention mechanism but also helps to introduce local inductive biases
*  The drawback is that it reduces the receptive field

![](/collections/images/NeighborhoodAttentionTransformer/receptive_fields.jpg)

# Neighborhood Attention

![](/collections/images/NeighborhoodAttentionTransformer/NeighborhoodAttention.jpg)

* Neighborhood attention on a single pixel $$(i, j)$$ is defined as follows:

$$ NA(X_{i, j}) = softmax(\frac{Q_{i,j}K^T_{\rho(i,j)} + B_{i,j}}{scale})V_{\rho_{(i,j)}} $$

where $$Q, K, V$$ are linear projection of $$X$$

$$B_{i,j}$$ denotes the relative positional bias

with ρ(i, j), which is a fixed-length set of indices of pixels nearest to (i, j), for a neighborhood of size $$ L * L $$, $$ \lVert \rho(i,j) \rVert = \lVert L² \rVert$$	

> However, if the function ρ maps each pixel to all pixels ($$L²$$is equal to feature map size), this will be equivalent to self attention.

- The complexity of the neighborhood attention is linear with respect to resolution unlike self attention's.
- The function $$\rho$$ which maps a pixel to a set of neighboring pixels is realized with a sliding window.
- For corner pixels that cannot be centered, the neighborhood is expanded to maintain receptive field size. As illustrated in the image below.

![](/collections/images/NeighborhoodAttentionTransformer/corner_pixels.jpg)

# Neighborhood Attention Transformer

![](/collections/images/NeighborhoodAttentionTransformer/architecture.jpg)

- For the tokenization they use the overlapping convolution method introduced in the  [Compact Convolutional Transformer](https://creatis-myriad.github.io./2022/06/13/CompactConvolutionalTransformer.html)

- The rest of the architecture is a succession of blocks containing a token merging layer to reduce dimension and standard multi head attention block but with the self attention replace by the neighborhood attention

  > Note : Similar to the Swin Transformer this architecture build hierarchical features maps by using token merging layers

# Results

## Classification

![](/collections/images/NeighborhoodAttentionTransformer/results_classification.jpg)

## Object Detection

![](/collections/images/NeighborhoodAttentionTransformer/results_object_detection.jpg)

## Semantic Segmentation

![](/collections/images/NeighborhoodAttentionTransformer/results_segmentation.jpg)

## Ablation studies



# Conclusion

