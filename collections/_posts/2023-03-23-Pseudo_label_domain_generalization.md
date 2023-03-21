---
layout: review
title: "Enhancing Pseudo Label Quality for Semi-Supervised Domain-Generalized Medical Image Segmentation "
tags: deep-learning domain-generalization semi-supervised
author: "Maylis Jouvencel"
cite:
    authors: "Huifeng Yao, Xiaowei Hu, Xiaomeng Li"
    title:   "Enhancing Pseudo Label Quality for Semi-Supervised Domain-Generalized Medical Image Segmentation"
    venue:   "AAAI 2022"
pdf: "https://arxiv.org/pdf/2201.08657.pdf"
---



# Highlights
* The goal of the paper is to improve segmentation pseudo-labels generated in a semi-supervised setting for domain generalization
* The contributions of the paper include a data augmentation scheme based on the Fourier Transform and a confidence-aware cross pseudo supervision network 

# Introduction

Pb: labelled/ unabbeled ...

# Method

[comment]: <> (![](/collections/images/pseudo_labels_DG/pipeline.jpg))

*Figure 1: pipeline proposed by the authors*

## Data Augmentation by Fourier Transformation





**Architecture-wise:** one MLP network with 8 fully-connected layers is learned for each scene. This MLP learns to map a position $$\boldsymbol{x}$$ and a direction $$\boldsymbol{d}$$ to a density $$\sigma$$ and color values $$RGB$$ :
$$ F_\Theta : (\boldsymbol{x},\boldsymbol{d}) \rightarrow (RGB,\sigma)$$




## Confidence-Aware Cross Pseudo Supervision 



## Implementation

Gradient descent is used to optimize the model.

The loss used is : 



# Results


![](/collections/images/NeRF/results_table.jpg)

*Table 1: quantitative comparaison with state-of-the-art method [metrics: SNR/SSIM (higher is better), LPIPS (lower is better))*



# Conclusion

- Bla