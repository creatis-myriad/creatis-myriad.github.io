---
layout: review
title: "Segment Anything for Medical Imaging"
tags: deep-learning medical image segment-anything
author: "Robin Trombetta"
cite:
    authors: "ZJun Ma, Yuting He, Feifei Li, Lin Han, Chenyu You, Bo Wang"
    title:   "Segment Anything in Medical Images"
pdf: "https://arxiv.org/pdf/2304.12306.pdf"
---

# Note

This is a combined review of the four following papers. The aim is to give ea idea about how Segment Anything has been used and adapted to medical images since its released, in April 2023.

* J. Ma, Y. He, F. Li, L. Han, C. You, B. Wang, **Segment Anything in medical imaging**, April 2023
* K. Zhang, D. Liu, **Customized Segment Anything Model for Medical Image Segmentation**, April 2023
* J. Wu, R. Fu, Y. Zhang, H. Fang, Y. Liu, Z.Wang, Y. Xu, Y. Jin, **Medical SAM Adapter: Adapting Segment Anything Model for Medical Image Segmentation**, April 2023
* X. Lin, Y. Xiang, L. Zhang, X. Yang, Z. Yan, L. Yu, **SAMUS: Adapting Segment Anything Model for Clinically-Friendly and Generalizable Ultrasound Image Segmentation**, September 2023

# Highlights

* Segment Anything[^1] performs badly on medical images as it has only been trained on natural images
* Several adaptations have been proposed to leverage SAM and finetune it on medical datasets
* Most if them rely on Adaptation Modules introduced in SAM's Transformer architecture

&nbsp;

# unCLIP overview

<div style="text-align:center">
<img src="/collections/images/DALLE2/dalle2_overview.jpg" width=800></div>
<p style="text-align: center;font-style:italic">Figure 1.Overview of unCLIP model.</p>

> Even if unCLIP is called 'DALL-E 2', the way it works is very different from the first version of DALL-E (based on VQ-VAE).

&nbsp;

# Prior

# References
[^1]: A. Kirillov, E. Mintun, N. Ravi,H. Mao, C. Rolland,L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg,  W.-Y. Lo, P/ Dollar, R. Girshick, *Segment Anything*, ICCV 2023