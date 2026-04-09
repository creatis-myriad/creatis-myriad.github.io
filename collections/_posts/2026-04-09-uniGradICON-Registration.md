---
layout: review
title: "UniGradICON: A Foundation Model for Medical Image Registration"
tags: Medical Image Registration, Foundation Models, Deep Learning, Registration Properties
author: "María Márquez-Sosa"
cite:
    authors: "Lin Tian, Hastings Greer, Roland Kwitt, François-Xavier Vialard, Raúl San José Estépar, Sylvain Bouix, Richard Rushmore and Marc Niethammer"
    title:   "UniGradICON: A Foundation Model for Medical Image Registration"
    venue:   "Medical Image Computing and Computer Assisted Intervention – MICCAI 2024: 27th International Conference, Marrakesh, Morocco, October 6–10, 2024, Proceedings, Part II"
pdf: "https://arxiv.org/abs/2403.05780"
---

<br/>

# Highlights

- **uniGradICON** [1] is presented as a first step towards a foundation model for medical image registration.
- The central claim is that a single deep learning (DL) registration model can cover multiple registration settings that normally require separate task specific networks.
- The paper focuses on three capabilities: 
  1) Performance across several in-distribution datasets.
  2) Zero shot transfer to out-of-distribution tasks.
  3) Use as a pre-trained initialization for fine-tuning on unseen tasks  
- The approach aims to bridge classical registration methods, which are generic but slow, and learning-based models, which are fast and accurate but typically specialized.
- The model is trained and evaluated on 12 public datasets.

---

# Extras

- Code and model available: [Github](https://github.com/uncbiag/uniGradICON) 
-  **uniGradICON** has been used as a baseline in [LUMIR Brain MRI Registration Challenge](https://github.com/JHU-MedImage-Reg/LUMIR_L2R) 
- An extention of this work:
  - **multiGradICON: A Foundation Model for Multimodal Medical Image Registration** [2] won best oral presentation at 2024 MICCAI Workshop for Biomedical Image Registration (WBIR) 

---

# Background
 
**Image registration** is a fundamental problem in medical image analysis, aimed at **estimating physically plausible spatial correspondences between image pairs**, with a **wide range of downstream tasks**. 

<figure style="text-align:center;">
  <img src="/collections/images/uniGradICON/Transform.gif" width=1000 alt="Illustration of image registration showing source image, deformation field, warped image, and target image">
  <figcaption><i>Figure 1. Illustration of image registration. From left to right: source image, deformation field, warped image, and target image. The transformation is physically plausible, ensuring a one to one mapping without folding.</i></figcaption>
</figure>

## Classical registration methods

Classical approaches formulate registration as an optimization problem solved independently for each image pair [3].

These methods can be described along three dimensions:

1) **Transformation model**
   * Defines the admissible class of deformations
   * Ranging from low dimensional parametric models (e.g., rigid, affine) to high dimensional non-parametric formulations 
   * **Diffeomorphic transformations are often preferred**
     - They enforce strong geometric and physical constraints on the deformation
2) **Similarity measure**
   * Quantifies alignment quality between images
   * Depends on modality and image appearance
     - e.g., intensity based or statistical measures
3) **Solution strategy**
   * Defines how parameters are estimated

### Limitations

- Strong dependence on application specific assumptions
    1) Anatomical region
    2) Imaging modality and acquisition protocol
    3) Expected deformation patterns
- Requires careful design of the transformation model and regularization
- Sensitive to hyperparameter tuning
- High computational cost due to per pair optimization

## Learning based registration methods

Learning based approaches replace per pair optimization by training models, typically neural networks, that predict transformations directly and can be applied to new image pairs

These methods can also be characterized along several key dimensions:
1) **Learning paradigm**
    - Supervised, semi-supervised or unsupervised training
2) **Network architecture**
    - Typically encoder-decoder structures such as U-Net [4]
    - Multi-scale or coarse to fine designs for large deformations
    - Multi-step formulations with iterative refinement
3) **Generalization capability**
    - Task specific (most methods), multitask, or foundation models
4) **Inference strategy**
    - Single forward pass
      - Direct prediction of the transformation
    - Refinement strategies
      - Optional instance optimization (IO) at inference to improve alignment
  
### Advantages

- Inference time
- Competitive or improved accuracy compared to classical methods
  
### Limitations

- Often trained for a specific dataset or task
  - Limited Flexibility
- Additional design complexity for architecture and training
- Loss functions remain SIMILAR to classical formulations, combining similarity and regularization 
- Hyperparameter tuning remains necessary

## Key Challenge

The main challenge is to design a registration framework that:

- Preserves the generality of classical optimization based methods
- Achieves the efficiency and accuracy of learning based, but task specific, approaches

_The trade-off between generality, efficiency, and robustness motivates the development of more flexible registration frameworks._

--- 

# Problem formulation

Let $I_A : \Omega \to \mathbb{R}$ and $I_B : \Omega \to \mathbb{R}$ denote the source and the target images on a spatial domain $\Omega \to \mathbb{R^d}$.  

The **goal of image registration** is to estimate a spatial transformation $\Phi_{AB} : \mathbb{R}^d \to \mathbb{R}^d$ such that the warped source image aligns with the target:

$$
I_A \circ \Phi_{AB} \sim I_B.
$$

The transformation map $\Phi_{AB}$ is a **diffeomorphism** if it is differentiable, bijective, and its inverse is differentiable as well. 
 - This ensures **smooth, invertible, and topology preserving mappings**.

## Optimization based formulation

Classical registration methods estimate the transformation parameters independently for each image pair by solving:

$$
\tau^* = \arg\min_{\tau} \; \mathcal{L}_{\text{sim}}(I_A \circ \varphi_{\tau}^{-1}, I_B) + \lambda \, \mathcal{L}_{\text{reg}}(\tau),
$$

- $\mathcal{L}_{\text{sim}}(\cdot, \cdot)$: similarity measure between the warped source and the target
- $\mathcal{L}_{\text{reg}}(\cdot)$: regularizer
- $\tau$: transformation parameters
- $\lambda \geq 0$: weighting parameter

## Learning based formulation

Learning based approaches replace per pair optimization with a parametric model $\Phi_\theta$, over the parameters $\theta$ of a neural network, that directly predicts the transformation from image pairs.

Given a dataset of image pairs

$$
\mathcal{I} = \{(I_A^i, I_B^i)\}_{i=1}^N
$$

the model is trained by solving:

$$
\theta^* = \arg\min_{\theta} \; \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}_{\text{sim}}\left(I_A^i \circ \Phi^{AB}_{\theta,i}, I_B^i\right) + \lambda \, \mathcal{L}_{\text{reg}}(\Phi^{AB}_{\theta,i}),
$$

- $\Phi^{AB}_{\theta,i}$ is as shorthand for $\Phi_{\theta}[I_A^i, I_B^i]$: predicted transformation for the $i$-th input image pair.
- The loss retains the same structure as classical registration, combining similarity and regularization.

---

# Previous Work

A simple and straightforward non-rigid transformation model in learning based registration is the displacement vector field (DVF):

$$
\Phi^{AB} = Id + D
$$

where $D$ is a dense displacement field predicted by the network, mapping each spatial location to a new position.

To enforce physically plausible deformations, explicit regularization is applied to the displacement field. Common choises include **bending energy** 

$$
\mathcal{L}_{\text{reg}} = \sum_i \left\| \nabla^2 \left( (\Phi^{AB} - \mathrm{Id})_i \right) \right\|_F^2
$$

or **difussion regularization** 

$$
\mathcal{L}_{\text{reg}} = \left\| \nabla (\Phi^{AB} - \mathrm{Id}) \right\|_F^2
$$ 

However, these approaches have several **limitations**.
- Strong regularization limits the ability to model large and complex deformations
- Requires careful tuning of the similarity regularization trade off
- Does not guarantee invertibility and may produce foldings 

To address these issues, implicit regularization strategies have been proposed. 

## Inverse consistency (ICON)

ICON [5] introduces inverse consistency as a regularization mechanism:

$$
\Phi_{AB} \circ \Phi_{BA} \approx \text{Id}
$$

This enforces that forward and backward transformations are approximate inverses, encouraging smoothness, invertibility, and topology preservation.

The corresponding loss is: 

$$
\mathcal{L}_{\text{ICON}} = \left\| \Phi_{AB} \circ \Phi_{BA} - \text{Id} \right\|^2
$$

#### Limitations

  - Training convergence can be slow  
  - Requires careful balancing of the consistency term (positions)
  - Difficult to maintain stability at high resolution  

## Gradient inverse consistency (GradICON)

GradICON [3] extends ICON by enforcing inverse consistency at the differential level. Instead of penalizing the transformation directly, it **penalizes on the Jacobian of the inverse consistency**:

$$
\mathcal{L}_{\text{GradICON}} = \left\| \nabla \left( \Phi_{AB} \circ \Phi_{BA} \right) - I \right\|_F^2
$$

It acts as an **implicit first-order regularization** (see paper for demonstration of $H^1$ type regularization). It penalizes high-frequency distortions while allowing low-frequency deformations, which provides a balance between smoothness and flexibility.

#### Intuition  
  - Enforces consistency at the local (differential) level rather than globally 
  - Constrains the Jacobian of the composed transformations to remain close to identity
  - Encourages smooth, locally invertible deformations while allowing global flexibility 

#### Advantages  
  - Faster convergence during training  
  - Improved numerical stability at higher resolutions
  - Better control of local deformations  
  - Produces smooth and approximately diffeomorphic mappings  
  - Weaker regularization allows the network learn transformations directly from the data, enabling stable training with the same hyperparameters across datasets


<figure style="text-align:center;">
  <img src="/collections/images/uniGradICON/results_gradicon.jpg" width=700 alt="Results using the uniGradICON regularizer showing source image, target image, and overlap between warped source and deformation field">
  <figcaption><i>Figure 2. Results obtained using the uniGradICON regularizer, source (left), target (middle) and overlap between the warped source and the deformation field (right). Gradient Inverse Consistency enforces spatially regular transformations and achieves accurate alignment across knee, brain, and lung datasets [3].</i></figcaption>
</figure>

--- 

# uniGradICON contributions

uniGradICON extends the GradICON framework toward a **general purpose registration model**.

Its main contributions are:

1) **Foundation model for registration**
   * Trained on diverse datasets covering multiple anatomies and modalities
   * Aims to generalize to unseen registration tasks without retraining

2) **Single training protocol**
   * Uses the same architecture, loss function, and hyperparameters across datasets
   * Avoids task specific tuning

3) **Strong generalization capability**
   * Supports zero shot inference on new datasets
   * Provides a good initialization for further finetuning

## Evaluation protocol

These properties are evaluated under three settings:

1) In-distribution performance
2) Out-of-distribution generalization with zero-shot inference
3) Fine-tuning on unseen datasets

--- 
# Materials and Methods

## Data

A composite training dataset was created from 12 publicly available datasets with different:
1) Anatomical regions
   - Lung, knee, brain, and abdomen
2) Modalities
   - CT, CBCT, and MRI
3) Deformation patterns
   - Intra-subject and inter-subject variations
   - Physiological motion such as lung inspiration and expiration

<figure style="text-align:center;">
  <img src="/collections/images/uniGradICON/data.jpg" width=800 alt="Summary of datasets used for training and evaluation">
  <figcaption><i>Table 1. Summary of datasets used for training and evaluation.</i></figcaption>
</figure>

The performance of uniGradICON is evaluated across four dataset categories, as defined in Table 2.

<figure style="text-align:center;">
  <table style="margin:auto; border-collapse:collapse;">
    <thead>
      <tr>
        <th></th>
        <th>Anatomical region</th>
        <th>Deformation</th>
        <th>Acquisition</th>
        <th>Modality</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>In-distribution</td>
        <td>✓</td>
        <td>✓</td>
        <td>✓</td>
        <td>✓</td>
      </tr>
      <tr>
        <td>Out-distribution (Type 1)</td>
        <td>✓</td>
        <td>-</td>
        <td>✗</td>
        <td>✓</td>
      </tr>
      <tr>
        <td>Out-distribution (Type 2)</td>
        <td>✗</td>
        <td>✗</td>
        <td>-</td>
        <td>✓</td>
      </tr>
      <tr>
        <td>Out-distribution (Type 3)</td>
        <td>✓</td>
        <td>-</td>
        <td>-</td>
        <td>✗</td>
      </tr>
    </tbody>
  </table>
  <figcaption><i>Table 2. Types of generalization. ✓ and ✗ indicate whether the corresponding data are included in the composite training dataset. − indicates cases where generalization is not explicitly evaluated for uniGradICON.</i></figcaption>
</figure>

## Training protocol

* The intra-patient dataset (Dataset 1 - COPDGene) contains 899 lung CT pairs acquired at inspiration and expiration
* The inter-patient datasets contain 2532 (Dataset 2 - OAI), 1076 (Dataset 3 - HCP), and 30 images (Dataset 4 - Abdomen), respectively

For inter-patient data, image pairs are formed by randomly sampling two images within each dataset, leading to many possible combinations.

To reduce bias caused by the imbalance in pair across datasets, a fixed number of samples is used during training:

* 1000 image pairs are randomly sampled from each dataset per epoch -> Results in 4000 3D image pairs per training epoch

## Preprocessing

### Intensity 

<figure style="text-align:center;">
  <table style="margin:auto; border-collapse:collapse;">
    <thead>
      <tr>
        <th style="text-align:center;">Step</th>
        <th style="text-align:center;">CT Images</th>
        <th style="text-align:center;">MRI Images</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>1</td>
        <td>Clip Hounsfield Units (HU) to [-1000, 1000]</td>
        <td>Clip maximum intensity at the 99th percentile</td>
      </tr>
      <tr>
        <td>2</td>
        <td>Normalize [0, 1]</td>
        <td>Normalize [0, 1]</td>
      </tr>
    </tbody>
  </table>
  <figcaption><i>Table 3. Preprocessing steps applied to the different imaging modalities</i></figcaption>
</figure>

### Spacing 

All images were resized to a fixed resolution of [175, 175, 175] using trilinear interpolation.

- Image spacing of the network input images may not be isotropic.

## Registration Network

The registration network follows a **multi-step, multi-resolution architecture** based on the GradICON framework. It estimates a dense deformation field by progressively refining spatial transformations between a $I^A$ and $I^B$.

<figure style="text-align:center;">
  <img src="/collections/images/uniGradICON/model_gradicon.jpg" width=700 alt="Illustration of the GradICON framework showing atomic registration networks, downsampling, and transformation composition">
  <figcaption><i>Figure 2. Illustration of the GradICON framework. The model is constructed from atomic registration networks $\Psi_i$ using downsampling (Down) and sequential transformation composition (TS) [3].</i></figcaption>
</figure>

The model consists of four identical U-Net modules $\Psi_1$, $\Psi_2$, $\Psi_3$, and $\Psi_4$, which each predict a displacement field. These modules are combined through two key operators:

#### **Downsampling (Down / DS)**
  Applies average pooling to process images at lower resolution:  

$$
DS\{\Psi_\theta\}[I^A​,I^B​]=\Psi_\theta[avgPool(I_A​,2),avgPool(I_B​,2)]
$$

#### **Two-Step composition (TS)**
 Sequentially composes two registration networks, where the second refines the deformation predicted by the first:

$$
TS\{\Psi_\theta^1, \Psi_\theta^2\}[I^A, I^B] 
= \Psi_\theta^1[I^A, I^B] \circ \Psi_\theta^2\big[I^A \circ \Psi_\theta^1[I^A, I^B], I^B\big]
$$

### Network Architecture

The full deformation model $\Psi_\theta$ is defined as:

$$
\Psi_\theta = TS\left\{
  TS\left\{
    DS\left\{
      TS\left\{
        DS\left\{\Psi_\theta^1\right\}, \Psi_\theta^2
      \right\}
    \right\}, \Psi_\theta^3
  \right\}, \Psi_\theta^4
\right\}
$$

_This formulation explicitly encodes a coarse-to-fine refinement strategy with repeated composition of transformations._

### Stage 1, coarse to fine estimation

- $\Psi_1$ predicts a deformation at 1/4 resolution  
- $\Psi_2$ refines it at 1/2 resolution  
- $\Psi_3$ refines it at full resolution  

At each step, the moving image is warped using the current estimate before being passed to the next network. Each network therefore learns only the **residual deformation**.

### Stage 2, final refinement

- $\Psi_4$ operates at full resolution  
- It refines the deformation produced by Stage 1  

This final step **improves alignment of fine anatomical details**.

## Training loss

Combines image similarity and gradient inverse consistency:

$$
\mathcal{L}=
\mathcal{L}_{\text{sim}}\left(I^A \circ \Phi^{AB}, I^B\right)
+
\mathcal{L}_{\text{sim}}\left(I^B \circ \Phi^{BA}, I^A\right)
+
\lambda \left\| \nabla \left(\Phi^{AB} \circ \Phi^{BA}\right) - \mathbf{I} \right\|_F^2
$$

- $\Phi^{AB}=\Psi_\theta[I^A,I^B]$ maps $I^A$ to $I^B$
* $\Phi^{BA}=\Psi_\theta[I^B,I^A]$ is obtained by swapping the input pair
* The similarity term is computed symmetrically in both directions
* The regularization term enforces gradient inverse consistency

Localized normalized cross correlation (LNCC) is used as the similarity measure.

## Training strategy

The network is trained in two stages:

* Stage 1 trained for 800 epochs
* Stage 2 trained for 200 epochs
* Learning rate set to $5 \times 10^{-5}$
* Regularization weight $\lambda = 1.5$ 

**These settings are kept fixed across datasets, defining unified training protocol.**

## Instance optimization

At inference time, the predicted transformation can be further refined by **instance optimization** (IO).

* The network output $\Psi_\theta [I^A,I^B]$ is used as initialization
* The same symmetric loss function is optimized for a specific image pair
* Gradient descent is applied for a few iterations

# Results

<figure style="text-align:center;">
  <img src="/collections/images/uniGradICON/images1.jpg" width=700 alt="uniGradICON registration results showing source, target, warped image, overlay, and difference image">
  <img src="/collections/images/uniGradICON/images2.jpg" width=700 alt="Additional uniGradICON registration results showing source, target, warped image, overlay, and difference image">
  <figcaption><i>Figure 3. Visualization of uniGradICON registration results for zero-shot inference. From left to right: source image, target image, warped image, overlay between the source and the displacement field, and difference image [1].</i></figcaption>
</figure>

## Performance on in-distribution tasks

The in-distribution performance of uniGradICON was evaluated on lungs (dataset 5 - COPDGene), knee (dataset 6 - OAI), brain (dataset 7 - HCP), and abdomen (dataset 4 - L2R-Abdomen).

<figure style="text-align:center;">
  <img src="/collections/images/uniGradICON/results1.jpg" width=700 alt="Comparison between task specific and universal registration models based on VoxelMorph, LapIRN and uniGradICON">
  <figcaption><i>Table 4. Comparison between task-specific (top) and universal (bottom) models, including VoxelMorph, LapIRN and GradICON [1].</i></figcaption>
</figure>

_uniGradICON achieves performance comparable to models trained specifically for each dataset, while using a single unified model._

## Performance on out-of-distribution tasks

### **Type I**: same anatomy and modality, different source

Zero-shot inference of uniGradICON was evaluated on one lung dataset (dataset 8 - L2R-NLST), and two brain datasets (dataset 9 - L2R-OASIS and dataset 10 IXI). 

- Evaluation was performed on the validation sets, as test sets were not available
- It outperforms SyN across all three registration tasks and achieves performance within the range of the top 5 Learn2Reg methods. 
  - This comparison assumes similar distributions between validation and test sets
<figure style="text-align:center;">
  <img src="/collections/images/uniGradICON/results2.jpg" width=700 alt="Zero-shot performance of uniGradICON on Type I out-of-distribution tasks with and without instance optimization">
  <figcaption><i>Table 5. Zero-shot performance of uniGradICON on Type I out-of-distribution tasks, with and without IO [1].</i></figcaption>
</figure>

**Instance optimization further improves the results in all cases.** 

### **Type II**: unseen anatomical regions, same modality

Generalization is evaluated by excluding L2R-Abdomen (dataset 4) from the training dataset and testing the model on these unseen images.

- It performs worse than the top 5 Learn2Reg methods, even with IO, but still outperforms SyN. 
- It provides reasonable alignment despite the anatomical shift.

<figure style="text-align:center;">
  <img src="/collections/images/uniGradICON/results3.jpg" width=700 alt="Zero-shot performance of uniGradICON on Type II out-of-distribution tasks with and without instance optimization">
  <figcaption><i>Table 6. Zero-shot performance of uniGradICON on Type II out-of-distribution tasks, with and without IO [1].</i></figcaption>
</figure>

### **Type III**:  same anatomical region, unseen modalities

Generalization is evaluated on two datasets with unseen modality combinations: lung CT/CBCT (dataset 11 - L2R-CBCT) and abdomen CT/MRI (dataset 12 - L2R-CTMR).

<figure style="text-align:center;">
  <img src="/collections/images/uniGradICON/results4.jpg" width=700 alt="Zero-shot performance of uniGradICON on Type III out-of-distribution tasks with and without instance optimization">
  <figcaption><i>Table 7. Zero-shot performance of uniGradICON on Type III out-of-distribution tasks, with and without IO [1].</i></figcaption>
</figure>

uniGradICON can generalize to unseen modalities and handle multi-modal registration, although performance depends on the difficulty of the modality shift. 

## Performance on finetuning on out-of-distribution dataset

The performance of uniGradICON was evaluated when used as an **initialization** and fine-tuned on a lung CT/CBCT registration task (Dataset 11) for 4000 epochs, using the same hyperparameters.

<figure style="text-align:center;">
  <img src="/collections/images/uniGradICON/results5.jpg" width=700 alt="Evaluation of uniGradICON on Type III out-of-distribution tasks with zero-shot inference, instance optimization, and target task finetuning">
  <figcaption><i>Table 8. Evaluation of uniGradICON on Type III out-of-distribution tasks with zero-shot inference, IO, and target task fine-tuning [1].</i></figcaption>
</figure>


# Limitations

- Limited number of in-distribution evaluation tasks, which restricts the assessment of performance on seen data
- Training data diversity remains constrained, despite combining multiple datasets
- Limited multi-modal generalization abilities
  - Current approach relies on LNCC, which is not optimal for cross-modality alignment
  - Could benefit from modality-agnostic representations
  - Alternative similarity measures could improve performance, such as $1 - LNCC^2$, normalized mutual information (MI) or MIND (Modality independent neighbourhood descriptor).

# Conclusions

- The paper concludes that a foundation registration network is feasible, provided that the deformation model, similarity measure, and regularization strategy are chosen so that one common training protocol can survive across heterogeneous data.
- Deep registration can move beyond the one network per task paradigm. 
- uniGradICON performs on par with task specific SOTA methods on in-distribution tasks, gives competitive zero-shot transfer on new datasets and some unseen modalities, and provides a strong initialization for later fine-tuning.

<div style="border-left: 4px solid #007acc; background-color: #f5f5f5; padding: 12px; margin: 16px 0;">
  <strong>Limitations mainly covered in multiGradICON [2]</strong>
  <ul style="margin-top: 8px;">
    <li>Uses a multimodal similarity measure</li>
    <li>Incorporates multimodal registration tasks into training</li>
    <li>Explores the impact of different factors:
      <ul>
        <li>Training similarity loss: (1 − LNCC) or (1 − LNCC²)</li>
        <li>Instance optimization loss: (1 − LNCC²) or MIND-SSC</li>
        <li>Training loss strategy: baseline or label randomization</li>
      </ul>
    </li>
  </ul>

  <figure style="text-align:center; margin-top:16px;">
    <img src="/collections/images/uniGradICON/comparison.jpg" width=700 alt="Comparison between uniGradICON and multiGradICON on T1w MRI and mean diffusivity registration">
    <figcaption><i>Figure 4. Comparison of uni- and multiGradICON on T1w MRI-mean diffusivity (MD) registration from ABCD. Note the improved matching of the ventricles for multiGradICON [2].</i></figcaption>
  </figure>
</div>


# References 

[1] Tian, L. et al. (2024). uniGradICON: A Foundation Model for Medical Image Registration. In: Linguraru, M.G., et al. Medical Image Computing and Computer Assisted Intervention – MICCAI 2024. MICCAI 2024. Lecture Notes in Computer Science, vol 15002. Springer, Cham. https://doi.org/10.1007/978-3-031-72069-7_70

[2] Demir, B. et al. (2024). MultiGradICON: A Foundation Model for Multimodal Medical Image Registration. In: Modat, M., Simpson, I., Špiclin, Ž., Bastiaansen, W., Hering, A., Mok, T.C.W. (eds) Biomedical Image Registration. WBIR 2024. Lecture Notes in Computer Science, vol 15249. Springer, Cham. https://doi.org/10.1007/978-3-031-73480-9_1

[3] Tian, L., Greer, H., Vialard, F. X., Kwitt, R., Estépar, R. S. J., Rushmore, R. J., Makris, N., Bouix, S., & Niethammer, M. (2023). GradICON: Approximate Diffeomorphisms via Gradient Inverse Consistency. Proceedings. IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2023, 18084–18094. https://doi.org/10.1109/cvpr52729.2023.01734

[4] Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. In: Navab, N., Hornegger, J., Wells, W., Frangi, A. (eds) Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015. MICCAI 2015. Lecture Notes in Computer Science(), vol 9351. Springer, Cham. https://doi.org/10.1007/978-3-319-24574-4_28 

[5] Greer, H., Kwitt, R., Vialard, F. X., & Niethammer, M. (2021). ICON: Learning Regular Maps Through Inverse Consistency. Proceedings. IEEE International Conference on Computer Vision, 2021, 3376–3385. https://doi.org/10.1109/iccv48922.2021.00338

