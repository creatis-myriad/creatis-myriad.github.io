---
layout: review
title: "GraphRegNet: deep graph regularisation networks on sparse keypoints for dense registration of 3D lung CTs"
tags: deep-learning CNN segmentation medical essentials
author: Mehdi Shekarnabi
cite:
    authors: "Hansen, Lasse and Heinrich, Mattias P"
    title:   "GraphRegNet: deep graph regularisation networks on sparse keypoints for dense registration of 3D lung CTs"
    venue:   "IEEE Transactions on Medical Imaging 2021, p.2246-2257"
pdf: "https://ieeexplore.ieee.org/abstract/document/9406964"
---

# Context
## Problems
- **Large 3D deformations** estimation of inhale to exhale lung CT do not allow an acceptable registration accuracy (due to local optima). 
- Extremely high **memory requirement** for regularization of the 3D displacement field in the case of discrete displacements search.
- U-Net-like **network architectures** even with multi-level strategies have not reached the accuracy of conventional frameworks.

## Proposed solutions
- Formulate the registration task as the prediction of displacement vectors on a **sparse irregular grid of distinctive keypoints**.
- Introduce a combination of **convolutional and graph neural network** architecture for displacement regularization.

# Methods
The methodology consists in:
- Preprocessing of the input images
- Selection of distinctive keypoints
- Feature extraction
- Calculation of cost tensor
- Regularization by GraphRegNet
- Turn sparse into a dense deformation field
<p align="center"><img src="/collections/images/GraphRegNet/main.png"></p>

Fig1. Overview of the learning framework for keypoint-based dense deformable image registration. Feature maps $F_F$ and $F_M$ are extracted from both, the fixed ($F_F$) and moving ($F_M$) image. Additionally, a set of sparse keypoints P is identified at distinctive locations in the fixed image using the Foerstner operator. Correlating the sampled MIND features at the keypoints in the fixed image and dense displaced locations L in the moving image, yields a cost tensor C for each keypoint. We then predict displacement vectors with our proposed GraphRegNet $\theta$, that consists of three neural network modules. First, an encoder CNN $\theta_E$ learns a low-dimensional displacement embedding for each cost tensor, then a GCN $\theta_G$ is employed that distributes the learned embeddings across the kNN graph of the keypoints to achieve spatial regularization. Final displacement vectors are obtained via integration over the predefined displacement space L, weighted by probabilities of the predicted softmax map of the decoder CNN $\theta_D$. All displacement vectors of the sparse keypoints are accumulated in the displacement field tensor D using trilinear extrapolation (+ spatial smoothing), which makes this densification operation fully differentiable and enables the use of an MSE loss L on the fixed and warped moving MIND image, guiding the training process in an unsupervised fashion.
## Pre-processing
- Segmentation (bounding box + crop extra image margins)
- Affine alignment of BBs
- Resampling to a fixed volume size + grayscale values are clipped between 0 and 1

## Distinctive keypoints
- Foerstner interest operator
$$ S(I_F) = \frac{1}{Tr((G_{\sigma_1} \ast (\nabla I_F \nabla I_F^T))^{-1})}  $$

- High responses of S correspond to distinctive locations. 
- **Non-max suppression:** 
  - $S_{MAX}$ = maxpooling($S$, kernel_size = d, stride = 1)
  - Choose points where $S == S_{MAX}$ ($p = (p_x,p_y,p_z) \to P$) 
- Selected voxels are limited to the lung region (mask).
- Adapt to a fixed number of points ($N_P$):
  - Furthest point sampling if $|P|\geq N_P$
  - Random insertion if $|P|< N_P$
- $S$ is only calculated for the inspi (fixed) image. 

<p align="center"><img src="/collections/images/GraphRegNet/Foerstner.png" height="400"><figcaption>Fig2. Visualisation of extracted keypoints on an exemplary saggital lung CT slice.</figcaption></p>

## Feature extraction
- Handcrafted MIND features (Modality independent neighbourhood descriptor)
- A 12 channel feature map.
- They will be used by the loss function too.

## Feature correlation
- Similarity search for $F_F(p)$ and $F_M(p+l)$ (for $p\in P$)
- $l \in L = 2 . \{-14,-13,...,0,...,13,14\}^3$ (Dense displacements)
- Cost tensor:
$$ C(p,l) = \frac{1}{12} \sum_{i = 0}^{11} (F_F^i(p)-F_M^i(p+l)) $$
- Shape = $(2048,3,29^3)$
- $C = G_{\sigma_2} \ast C$ (along displacements dim)
  
## GraphRegNet
- Sparse displacement field $D_S = \theta(C)$
- $\theta(C) = \theta_{E} + \theta_{G} + \theta_{D}$
### Encoder
- displacement embeddings = $CNN(C)$
### GCN
- $KNN$ graph of $P$ (K=15)
  $$ f_{i}\prime = ReLU\underset{(i,j)\in E}{(avg\;e_{ji})} $$
- $e_{ij} = h_{\theta}(f_i,f_j-f_i) = (\theta_1,\theta_2,...,\theta_{|f_i\prime|})\cdot cat(f_i,f_j-f_i)$

### Decoder
- heatmap = $H_p = CNN(f\prime)$
- $\tilde H_p = softmax(H_p)$
$$d=\sum_{l \in L} l\cdot \tilde H_p(l)$$

<p align="center"><img src="/collections/images/GraphRegNet/architecture.png"><figcaption>Fig3. Block diagram of the GraphRegNet architecture.</figcaption></p>

## Sparse to Dense Supervision
- $Loss = MSE(F_F,D(F_M))$
- D : dense displacement field
- Accumulating all displacements $d \in D_s$ in a dense, low resolution (1/3) tensor (initialized with zeroes) at respective keypoints $P$
- 3 $\times$ average pooling(kernel_size = 5, stride=1)
- Upsampling to obtain D

# Results
<p align="center"><img src="/collections/images/GraphRegNet/results.png"></p>
Table1. Registration results on the DIR-Lab 4D CT and COPDGene Datasets. The average landmark distance in millimeters for all individual cases, as well as the average distance and standard deviation over all cases of a dataset is reported. Results for comparison methods (with exception of VM+, LapIRN, FE+, PDD+ and MST) were taken from literature. For all other methods/experiments a test for statistical significance with respect to the proposed registration framework was conducted using the Wilcoxon signed rank test (calculated over all 3000 landmark pairs of a dataset). Significance levels are defined as * p < 0.05, ** p < 0.01 and *** p < 0.001

<p align="center"><img src="/collections/images/GraphRegNet/accumulative.png" width="50%"></p>

Fig4. Cumulative distribution of target registration errors in millimeters for all keypoint based methods on all landmark pairs of the COPDgene dataset. In addition, the dotted lines visualize the 75th percentiles of the TRE, which are 1.61 mm (ours), 1.73 mm (uniform), 1.80 mm (MST), 2.09 mm (sl), 2.27 mm (PDD+), 2.43 mm (RW), 4.38 mm (coords) and 5.25 mm (noreg).

# Conclusion
- State of the art results in deep learning based lung CT registration algorithms 
- Future research:
  - Image features: learn them in and end-to-end training
  - Keypoint graph: a more descriptive keypoint graph (e.g. from vessel trees) for more targeted graphical message passing
  - Test on other data: inter-patient abdominal CT
