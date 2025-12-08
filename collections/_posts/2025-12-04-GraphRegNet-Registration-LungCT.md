---
layout: review
title: "GraphRegNet: Deep Graph Regularisation Networks on Sparse Keypoints for Dense Registration of 3D Lung CTs"
tags: Lung, Computed tomography,Three-dimensional displays,Feature extraction
author: "María Márquez-Sosa"
cite:
    authors: "Tristan S.W. Stevens, Oisin Nolan, Jean-Luc Robert and Ruud J.G. van Sloun"
    title:   "Nuclear Diffusion Models for Low-rank Background Supression in Videos"
    venue:   "IEEE Transactions on Medical Imaging"
pdf: "https://arxiv.org/abs/2509.20886"
---

<br/>


# Highlights
- The authors propose a graph-based deep learning (DL) model for 3D lung registration in computed tomography (CT) images: **GraphRegNet** [1].
- Uses sparse keypoints instead of dense voxel grids to handle large deformations.
- Combines CNNs for displacement embedding and GNNs for spatial regularization.
- Achieves state-of-the-art (SOTA) accuracy on inhale-exhale lung CT benchmarks.
- Inference time under 2 seconds with very low memory usage.

<br/>

---
# Background

Automated analysis of chest CT is central to the **diagnosis and treatment planning of pulmonary diseases** such as COPD, emphysema, pneumonia, and lung cancer. 
- A **key component** of this analysis is accurate **deformable registration** between **inhale and exhale CT scans**. 
- This registration enables voxel-wise **estimation of lung ventilation and tissue mechanics**, which are otherwise only accessible through complex functional imaging modalities.

## Challenges on CT Lung Registration

Lung CT registration is particularly difficult due to three factors:

- **Large non rigid motion of fine structures**
During breathing, vessels and airways undergo displacements that are often larger than their own diameter.

- **Sliding motion at anatomical boundaries**
The lungs slide along the chest wall and between lobes. This breks the smooth and continuous motion assumptions of most regularization models and introduces physically incorrect constraints near the pleural surface.

![Sliding motion](Sliding_motion.png)

Figure 1: From left to right, inhalation, exhalation, and overlay. The overlay highlights sliding motion, mainly at the diaphragm-lung, inter-lobar and lung-pleural cavity interfaces [2].

- **Respiration-induced intensity changes**
Lung expansion and compression alter local tissue density. Consequently, voxel values differ even when anatomical correspondences are correct
  - Most similarity metrics asume brightness constancy: sum of squared differences (SSD), cross-correlation (CC), and mutual information (MI).
  - Quantitative assesment uses approaches relying on local spatial context.

## Related Work

Conventional optimization-based registration methods with strong regularization can reach high accuracy. However, they require long computation times and careful parameter tuning.

In contrast, DL approaches allow near real time inference. Most rely on dense voxel displacement prediction with multi resolution architectures (U-Net variants being the most common).
- While they perform well for small and moderate deformations, their performance reduces significantly for large lung motion.

These limitations motivated a **shift from dense voxel-based regression to sparse and graph-regularized displacement estimation**.

<br/>

---
# Methods

The fixed image **$I_F$** and moving image **$I_M$** are defined as the inhale and exhale CT scans, respectively. The registration objective is to **calculate a displacement field** $D$:  $\mathbb{R}^3 \to \mathbb{R}$ that **best aligns the inhale and exhale images**.

![GraphRegNet](GraphRegNet.png)
Figure 2: Overview of keypoint based deformable registration framework: **GraphRegNet**. Sparse keypoints are detected in $I_F$. MIND features are extracted from $I_F$ and $I_M$. Feature correlation across candidate displacements builds a cost tensor per keypoint. The model predicts displacements using a CNN encoder, a GCN for spatial regularization, and a CNN decoder. Sparse displacements are densified by trilinear extrapolation. Training is unsupervised using an MSE loss on fixed and warped MIND images [1].

The registration is performed in two stages, a **main stage** that estimates the large scale respiratory motion and a **refinement stage** that increases keypoint density and reduces the displacement search range to improve local alignment.

## Preprocessing

1) Segmentation using thresholding and morphological operations.
2) Resampling to a fixed spatial size of 192 × 160 × 192 voxels.
3) Clipping to the range −1000 to 1500 HU, followed by normalization.
4) Affine alignment of the masks.

### Distinctive Keypoint Extraction

Keypoints are extracted from the $I_F$ using the **Förstner operator** [3].
  - **Main idea**: A point is salient if the local gradient distribution is strong in multiple directions.
  - **Computation**:
    1) Compute the spatial gradients $\nabla I_F$, and apply Gaussian filtering ($\sigma = 1.4$).
    2) A distinctive score $S$ is given for each voxel in $I_F$: 
        ```math
            S(I_F) = \frac{1}{Tr((G_\sigma{_1} * (\nabla I_F \nabla I_F^T))^{-1})}
        ```
        High responses in $S$ correspond to distinctive locations.
    3) Max pooling to obtain a well (uniform) distributed set of keypoints $P$.
    4) Restrict the location of the keypoints to the lung region given by the lung mask.
    5) Adapt the number of keypoints in $P$ to a fixed number $N_p$ by farthest point sampling (if $|P| >= N_p$) or insertion of random points already present in $P$ (if $|P| < N_p$).

  

- Registration stage:  2048 keypoints. 
- Refinement stage:  3072 keypoints.


## Image Feature Extraction

For both $I_F$ and $I_M$, the **modality independent neighborhood descriptor (MIND)** [4] is computed at every voxel. 

- **Main idea:** For every voxel, MIND describes the local structure surrounding that voxel. 
- **Computation:** At each voxel $x$, $r$ is an element from the search space $\mathbb{R}$, the descriptor is calculated as:
    ```math
    \text{MIND}(I, x, r) = \frac{1}{n}
    \exp \left(
    - \frac{D_p(I, x, x + r)}{V(I, x)}
    \right)
    ```
    Where $D_p(I, x, x + r)$ is the patch distance between the descriptor voxel and the search space voxel, and $V(I,x)$ is the local variance estimate. 


-The result is a 12-dimensional vector per voxel. Each channel encodes how similar the voxel is to one neighboring offset.

## Feature Correlation and Cost Tensor Construction

For each keypoint $p \in P$ in the $I_F$, a discrete 3D displacement search is performed within a cubic search region by comparing fixed features $F_F(p)$ with moving features $F_M(p + l)$ at candidate displacement locations $l \in L = q \cdot \left\{-l_{\max}, \ldots, -1, 0, 1, \ldots, l_{\max}\right\}^3$

The displacement space is defined using a quantization step size $q$ and a maximum expected displacement $l_{max}$.

 - Registration stage:  $q=2$ and $l_{max} = 14$. 
 - Refinement stage:  $q=1$ and $l_{max} = 8$.

At each candidate displacement, similarity is computed using SSD between $F_F(p)$ and $F_M(p + l)$, resulting in a discrete cost tensor $C$ per keypoint, which is then smoothed with a Gaussian kernel ($\sigma = 1$) along the displacement dimensions. 

$C(p,l)= \frac{1}{12}\sum_{1=0}^{11}(F_F^i(p)-F_M^i(p+l))^2$

where $F_F^i$ and $F_M^i$ denote the i-th channel of the respective 12 channgel feature map.

-The spatial dimensions of $C$ remain sparse and are defined only at the keypoint locations.

## GraphRegNet Architecture

**Aim:** Predict a sparse displacement field $D_S$ that assigns a displacement vector $d = (dx , dy, dz)$ to each keypoint $p \in P$. 

### Encoder

Each keypoint cost tensor $C(p)$ is processed independently by a convolutional encoder that consist of. 

- Three convolutional layers with kernel size 3 and stride 2. 
- The number of feature channels increases at each layer. 
- Each convolution is followed by instance normalization and a leaky ReLU activation.

The encoder outputs a **low-dimensional displacement embedding**  for each keypoint.

### Graph Neural Network

- A k-nearest-neighbor graph is constructed using the spatial coordinates of the keypoints ($ k = 15$). 
- The displacement embeddings are concatenated with the corresponding 3D keypoint coordinates, and used as the initial node features.

Three graph convolution layers based on **edge convolutions (EdgeConv)** [5] are applied in a DenseNet-fashion (the input features of all previous layers are concatenated with the current layer output) while keeping the output feature channels constant.  

The EdgeConv operation is defined as

$f_i' = \mathrm{ReLU}\left( \mathrm{avg}_{(i,j)\in E}  e_{ij} \right)$

where $f_i'$ denotes the updated feature vector at node $i$, and $E$ is the edge set of the kNN graph. The edge features are computed as 

$e_{ij} = h_\theta(f_i, f_j - f_i)$

where $h_\theta$ is a fully connected layer.


![Graph](gcn.png)
Figure 3: Edge convolutions. Left: Computing an edge feature, $e_ij$, from a point pair $(x_i, y_j)$. $h\theta()$ is instantiated using a fully connected layer, and the learnable parameters are its associated weights. Right: The EdgeConv operation. The output of EdgeConv is calculated by aggregating the edge features associated with all the edges coming from each connected vertex [5].

These layers propagate information between neighboring keypoints and perform **learned spatial regularization** of the displacement embeddings.


### Decoder

The regularized displacement embeddings are passed to a convolutional decoder that consists of:
- Two upconvolutions (trilinear upsampling + convolution) 
- A single convolutional layer output a single channel feature map $H_p$ for each keypoint. 
- Softmax

The final displacement vector $d$ is determined by the integration over the displacement search region $L$ weighted by the normalized predictions $\hat{H}_p$ as

$d = \sum_{l \in L}l \cdot \hat{H}_p(l)$

### Sparse-to-dense Supervision

- The sparse displacement vectors predicted are converted into a dense displacement field through trilinear extrapolation and spatial smoothing. 

- Supervision is applied at the dense level by warping the moving feature image $F_M$ with the reconstructed displacement field $D$ and minimizing the MSE with respect to the fixed feature image $F_M$ inside the lung region.

$L = MSE(F_F, D(F_M))$

# Data

The method is evaluated on two widely used public lung registration benchmarks.

1) **DIR-Lab 4D CT dataset**
Consists of 10 inhale-exhale scan pairs acquired under controlled breathing conditions. Each scan pair includes 300 manually annotated landmark correspondences [6].

1) **DIR-Lab COPDgene dataset**
Contains 10 inhale-exhale CT scans with large respiratory motion. It also provides 300 expert-annotated landmarks per case [7].

   - 25 additional lung CT scans from public datasets are included. All experiments are conducted using five-fold cross-validation. 

![COPDGene](copd8-render.png)
Figure 4: 3D rendering of case 8: inspiratory and expiratory CT from the DIR-Lab COPDGene study. Landmark displacement vectors with colors indicating distances in millimeters are shown.

# Results

- Contains only about 33,000 trainable parameters. The total inference time including refinement is less than 2 seconds on a NVIDIA Titan RTX GPU, with a memory usage of less than 4GB. 11GB used for training. 

![Figures-GraphRegNet](figures-graphregnet.png)
Figure 5: Qualitative results of GraphRegNet on COPDGene scan pairs in sagittal view. From left to right: initial and final color overlays of inhale and exhale scans, two views of the predicted displacement field, and the Jacobian within the lung. 

![Results-GraphRegNet](results_graphregnet.png)
Figure 6: Registration results on the DIR-LAB 4D CT and COPDGene datasets. Mean landmark distance in mm per case. Baseline results are taken from SOTA. Statistical significance versus GraphRegNet is tested with the Wilcoxon signed-rank test over 3000 landmark pairs. Significance levels are * p < 0.05, ** p < 0.01, *** p < 0.001.


# Conclusion

- By **shifting from dense regression to keypoint based correspondence estimation** with learned spatial regularization, **GraphRegNet achieves high accuracy, robustness to large motion, and very low computational cost.**

# References 

[1] Hansen, L. and Heinrich, M.P. (2021) ‘GraphRegNet: Deep graph regularisation networks on sparse keypoints for dense registration of 3D Lung CTS’, IEEE Transactions on Medical Imaging, 40(9), pp. 2246–2257. doi:10.1109/tmi.2021.3073986. 

[2] Yuan, P. et al. (2026) ‘GraphMorph: Equilibrium Adjustment regularized dual-stream GCN for 4D-CT lung imaging with Sliding Motion’, Neurocomputing, 664, p. 132022. doi:10.1016/j.neucom.2025.132022. 

[3] Förstner, W., & Gülch, E. (1987). 'A fast operator for detection and precise location of distinct points, corners and centres of circular features'. In Proc. ISPRS intercommission conference on fast processing of photogrammetric data, Interlaken, Switzerland, 6, pp. 281-305.

[4] Heinrich, M.P. et al. (2012) ‘MIND: Modality independent neighbourhood descriptor for multi-modal deformable registration’, Medical Image Analysis, 16(7), pp. 1423–1435. doi:10.1016/j.media.2012.05.008. 

[5] Wang, Y. et al. (2019) ‘Dynamic graph CNN for learning on point clouds’, ACM Transactions on Graphics, 38(5), pp. 1–12. doi:10.1145/3326362. 

[6] Castillo, R. et al. (2009) ‘A framework for evaluation of deformable image registration spatial accuracy using large landmark point sets’, Physics in Medicine and Biology, 54(7), pp. 1849–1870. doi:10.1088/0031-9155/54/7/001. 

[7] Castillo, R. et al. (2013) ‘A reference dataset for deformable image registration spatial accuracy evaluation using the COPDGENE Study Archive’, Physics in Medicine and Biology, 58(9), pp. 2861–2877. doi:10.1088/0031-9155/58/9/2861. 