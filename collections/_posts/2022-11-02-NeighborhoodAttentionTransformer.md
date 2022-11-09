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

# Highlights

* Similar to Swin Transformer the idea is to reduce the computational cost of the attention mechanism 
* Introduce the Neighborhood Attention (NA) and the Neighborhood Attention Transformer (NAT)
* Here the attention is only compute on a neighborhood around each token
* It also helps to introduce local inductive biases but it reduces the receptive field

![](/collections/images/NeighborhoodAttentionTransformer/receptive_fields.jpg)

# Methods

![](/collections/images/unetr/overview_method.jpg)

## Neighborhood Attention

![](/collections/images/NeighborhoodAttentionTransformer/NeighborhoodAttention.jpg)

* Neighborhood attention on a single pixel $$(i, j)$$ is defined as follows:

$$ NA(X_{i, j}) = softmax(\frac{Q_{i,j}K^T_{\rho(i,j)} + B_{i,j}}{scale})V_{\rho_{(i,j)}} $$

where $$Q, K, V$$ are linear projection of $$X$$

$$B_{i,j}$$ denotes the relative positional bias

with ρ(i, j), which is a fixed-length set of indices of pixels nearest to (i, j), for a neighborhood of size $$ L * L $$, $$ \lVert \rho(i,j) \rVert = \lVert L² \rVert$$	

> However, if the function ρ maps each pixel to all pixels ($$L²$$is equal to feature map size), this will be equivalent to self attention.

- The complexity of the neighborhood attention is liner with respect to resolution unlike self attention's.
- The function $$\rho$$ which maps a pixel to a set of neighboring pixels is realized with a sliding window.
- For corner pixels that cannot be centered, the neighborhood is expanded to maintain receptive field size

![](/collections/images/NeighborhoodAttentionTransformer/corner_pixels.jpg)

## Neighborhood Attention Transformer

![](/collections/images/NeighborhoodAttentionTransformer/architecture.jpg)

## Experiments

* Loss is a combination of soft dice and cross-entropy 
* Method is evaluated on BTCV and MSD datasets
* BTCV : 30 patients with abdominal CT scans where 13 organs are annotated (13 class segmentation problem)
* MSD :  484 multi-modal and multi-site MRI (Flair, T1w, T1gd, T2w) for the brain tumor segmentaion task and 41 CT scan for the spleen segmentation task
* Dice and 95% Hausdorff Distance (HD) are used as evaluation metrics

* Transformer parameters used : $$L=12$$ transformer block, embedding size of $$K=768$$, patch size of $$ 16 * 16 * 16$$ 	
* Average training time : 10 hours for 20 000 epochs

* Note : the transformer backbone is not pre-trained at all

# Results

As seen in the table below, UNETR outperforms the state-of-the-art methods on the BTCV leaderboard ( which are CNN or transformer based methods[^1][^2][^3]) 

![](/collections/images/unetr/results_BTCV.jpg)

Same for the MSD dataset

![](/collections/images/unetr/results_MSD.jpg)

Some visual results on the BTCV dataset::

![](/collections/images/unetr/visual_results_BTCV.jpg)

# Ablation studies

Authors compare their decoder architecture with three other designs called Naive UPsampling (NUP), Progressive UPsampling (PUP) and MuLti-scale Aggregation (MLA) [^1]

![](/collections/images/unetr/ablation_decoder.jpg)

They also compare model complexity with other architectures:

![](/collections/images/unetr/parameters.jpg)

# Conclusions

UNETR has taken a first step towards transformer based models for segmentation

# References

[^1]: [Sixiao Zheng et al, *Rethinking semantic segmentation from a sequence-to-sequence perspective with transformers*, Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (2021)](https://arxiv.org/abs/2012.15840)
[^2]: [Jieneng Chen et al, *Transunet: Transformers make strong encoders for medical image segmentation*, arXiv preprint (2021)](https://arxiv.org/abs/2102.04306)
[^3]: [Yutong Xie et al, *Cotr: Efficiently bridging cnn and transformer for 3d medical image segmentation*, International conference on medical image computing and computer-assisted intervention  (2021)](https://arxiv.org/abs/2103.03024)