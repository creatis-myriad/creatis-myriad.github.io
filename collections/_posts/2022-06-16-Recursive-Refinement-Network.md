---
layout: review
title: Recursive refinement network for deformable lung registration
tags: deep-learning CNN registration medical lung CT
cite:
    authors: "He X, Guo J, Zhang X, Bi H, Gerard S, Kaczka D, Motahari A, Hoffman E, Reinhardt J, Barr RG, Angelini E."
    title:   "Recursive Refinement Network for Deformable Lung Registration between Exhale and Inhale CT Scans"
    venue:   "arXiv preprint arXiv:2106.07608. 2021 Jun 14"
pdf: "https://arxiv.org/pdf/2106.07608"
---

# Introduction

In this paper a recursive refinement network (RRN) is proposed for end-to-end unsupervised deformable registration. The performance of the algorithm is tested on [DirLab COPDGene](https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/index.html) dataset including 10 inhale/exhale lung CT images of COPD patients with almost 300 landmarks on each image. The network achieved a state-of-the-art average target registration error (TRE) of 0.83 mm.

# Methods
1. Multi-level features are extracted from fixed and moving images through 3D convolutional layers. 
2. Features are normalized (to avoid feature vanishing at higher levels, as intermediate deformation vector fields are not supervised).
3. The moving features are warped with the 2x upsampled DVF predicted at previous level (no warping for the topmost level).
4. Local cost correlation volumes are computed in a memory efficient way (which are the inner dot product of fixed and moving features within a small radius).
5. DVF is estimated at the topmost level by using *fixed features and cost volumes* then it is refined, level by level, by using *fixed features, cost volumes, context, and previous DVF*.
6. Final DVF is estimated through a 7-layer dilated convolutional network (with a large receptive field).

<p align = "center"><img src ="/collections/images/RRN/image1.png" alt="Trulli" style="width:90%"></p>
<figcaption align = "center">Fig.1 - The global architecture of RRN</figcaption>

<p align = "center"><img src ="/collections/images/RRN/image2.png" alt="Trulli" style="width:100%"></p>
<figcaption align = "center">Fig. 2. The initial (a), intermediate (b) and final (c) DVF estimators. (d-e) are the network architectures of intermediate (d) and final (e) DVF estimators. Feature 1 represents features of the fixed image and feature 2 represents features of the moving image. </figcaption>

### Loss Function
* **Similarity metric:** Normalized local (patch-wise) cross correlation
* **Regularization:** Total variation

# Results

<p align = "center"><img src ="/collections/images/RRN/image3.png" alt="Trulli" style="width:90%"></p>
<figcaption align = "center">Fig.3 - Example of RRN registration. From left to right: moving image, warped image, fixed image and deformation vector fields.</figcaption>

<p align = "center"><img src ="/collections/images/RRN/image4.png" alt="Trulli" style="width:100%"></p>
<figcaption align = "center">Fig.4 -  Comparison with state of the art classic registration methods. </figcaption>

<p align = "center"><img src ="/collections/images/RRN/image5.png" alt="Trulli" style="width:100%"></p>
<figcaption align = "center">Fig.5 -  Ablation experiment using VoxelMorph as a benchmark learning-based method. </figcaption>

# Conclusions
The light weight Recursive Refinement Network (RRN) can handle large inhale-exhale deformations and outperforms state of the art pTV and VoxelMorph methods in terms of TRE on the DirLab dataset.
 
# Remarks
The authors have generously made their code available on [github](https://github.com/Novestars/Recursive_Refinement_Network). I was able to achieve mean TRE ~ 1.0mm on the same dataset. On our local dataset, I observed that by choosing a wider intensity range for the input images, the output deformation field does not follow the sliding motion of the lungs. 