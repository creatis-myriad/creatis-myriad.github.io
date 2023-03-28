---
layout: review
title: "Deep label fusion: A generalizable hybrid multi-atlas and deep convolutional neural network for medical image segmentation"
tags: deep-learning CNN transformer segmentation classification object-detection attention
author: "Emile Saillard"
cite:
    authors: "Long Xie, Laura E.M. Wisse, Jiancong Wang, Sadhana Ravikumar, Pulkit Khandelwal, Trevor Glenn, Anica Luther, Sydney Lim, David A. Wolk, Paul A. Yushkevich"
    title:   "Deep label fusion: A generalizable hybrid multi-atlas and deep convolutional neural network for medical image segmentation"
    venue: Elsevier
pdf: "https://reader.elsevier.com/reader/sd/pii/S1361841522003115?token=CD4B7AB27E42E9CE6A6FAD29EF34D9ED4C7E505ECA20F6B363AAA2C43B59D9085D8BD62754D8DCC825DCB8F9A95594CC&originRegion=eu-west-1&originCreation=20230314135712"
---
# Notes

* The model was implemented in PyTorch using functionalities from the MONAI project
* The code for Deep Label Fusion is available on [github](github.com/LongXie/DeepLabelFusion)

# Reminder: Multi-Atlas Segmentation

* Multiple atlases (images with their manual segmentations) are warped into the space of target image via linear and deformable registration
* The warped atlases segmentations are combined into a consensus segmentation, using a **label fusion** algorithm

# Highlights

*  Deep Convolutional Neural Networks (DCNN) offer great results for medical image segmentation, but generalizability on new data that is not well represented in the training set can be underwhelming

*  Multiple Atlas Segmentation (MAS), while showing sub-optimal performance for segmentation, can offer a greater generalizability to new datasets

* **This article proposes a hybrid end-to-end MAS & DCNN segmentation pipeline, called Deep Label Fusion (DLF), as well as a dedicated data augmentation scheme for multi-modal datasets in order to combine the strengths of MAS and DCNN** 


# Deep Label Fusion Pipeline
![](/collections/images/DeepLabelFusion/DLF_pipeline.jpg) 
Figure 1 : Complete pipeline of DLF

* Input : Target image + set of registered atlases  $$A=\left\{A^i,i=1,2,...,N_{atlas} \right\},S=\left\{S^i,i=1,2,...,N_{atlas} \right\} $$
* Output : Segmentation of target image

> Note : The prior registration process is made by warping the atlases (images with their corresponding manual segmentation) into the space of the target image via non-linear diffeomorphic transformations

* The DLF pipeline consists of 3 distinct parts :

1. A weighted voting subnet
2. Label fusion computation
3. A fine-tuning subnet
  
## Weighted voting subnet

* This subnet is a 3 level U-Net designed to estimate the similarity between registered atlases and target image

![](/collections/images/DeepLabelFusion/subnets.jpg) 
![](/collections/images/DeepLabelFusion/subnet_legende.jpg) 
Figure 2 : Architecture of the weighted voting network

* Input : Pair of target/atlas image with coordinate maps (x,y,z) for spatial context
* Output : Label-specific weight maps $$W^i=\left\{W^i_l,l=1,2,...,N_{label} \right\} $$ for atlas i with value $$w_{ln}$$ at voxel n

* This network replaces conventional similarity metrics used in MAS 

 * The maps obtained are used to assign different weights to each atlas depending on the pixel considered

## Label Fusion computation

* Once the weight maps are acquired, the candidate segmentations $$S_i$$ are fused into initial consensus segmentation $$S^{init}$$ in 3 steps :
  - Candidate segmentations are converted to one-hot encoding segmentations $$S^i=\left\{p^i_l,l=1,2,...,N_{label} \right\} $$
  - Vote maps $$V^i=\left\{V^i_l,l=1,2,...,N_{label} \right\} $$ are computed by elementwise multiplying $$W^i$$ and $$S^i$$ for each labels and spatial location n
  - For each label, the vote maps of all the atlases are averaged to generate the initial segmentation

## Fine-tuning subnet

* This subnet is a 4 level U-Net designed to correct remaining residual errors after label fusion 

* Input : $$S^{init}$$ and associated coordinate maps
* Output : Set of feature maps of the same size as $$S^{init}$$

* A label-specific mask is then generated taking the union of all candidate segmentations and multiplying it by the corresponding channel of the output to get the final segmentation.

> Note : This label-masking operation assumes that the truth label is contained inside the region that has atlas votes of that label


# Data augmentation strategies

* In addition to usual data augmentation (Random flips, rotations, patching, elastic deformation and additive gaussian noise), DLF has 3 dedicated data augmentation methods :

  - A random selection of $$N_{Atlas}$$ is made for each target image. $$N_{Atlas}$$ can be superior to the total number of available datasets. This random selection allows for duplicate atlases to help handle similar votes, as well as making the network more robust to atlas variability

  - An extended Random Histogram Shift (RHS) is applied to atlas & target independantly, which corresponds to a change in contrast between the atlas image and the target image. This helps the weighted voting subnet to be more sensitive to image structure rather than intensity distribution
  
  - In case of multi-modality (T1 & T2 here), **Modality Augmentation (ModAug)** is implemented. The idea is to randomly replace a modality during training with white noise to force the model to base its prediction on individual modalities



# Experiments & Results

## Datasets
![](/collections/images/DeepLabelFusion/datasets.jpg) 
Figure 3 : Different datasets used for the experiments

## Cross-validation experiments

* Comparison with other MAS algorithms with various label fusion methods (Majority Voting (MV), Spatially Varying Weighted Voting (SVWV), Joint Label Fusion (JLF))

* Comparison with a 3D U-Net with a similar architecture as the fine-tuning model of DLF, as well as with nnUNet

* Oracle(10%) results consider that a pixel is rightly segmented after registration if at least 10% of atlases have the correct segmentation. Those results are used to evaluate the upper bound of label fusion performance

>Note : The oracle results are especially useful to characterize the registration difficulty for a specific task

![](/collections/images/DeepLabelFusion/cross-val.jpg) 
Figure 4 : Cross-validation results

* DLF outperforms conventional MAS methods in almost all tasks

* Compared to a conventional U-Net, DLF has similar or better results

* DLF shows comparable results to nnUNet, depending on the task

## Generalization experiments

* Inference on unseen datasets with different characteristics and image resolution :
  - 7T instead of 3T for MRI images
  - Presence of lesions for the lumbar-CT dataset (fracture, osteoporosis...)
  
* Comparison with JLF-CL, standard 3D U-Net and nnUNet

![](/collections/images/DeepLabelFusion/generalizability.jpg) 
Figure 5 : Generalizability results

* MAS methods show better generalizability than DCNN methods

* DLF has significantly better generalizability compared to all other methods

## Modality augmentation

* The effect of ModAug was tested on DLF as well as a standard 3D U-Net

![](/collections/images/DeepLabelFusion/ModAug.jpg) 
Figure 6 : Segmentation results using ModAug 

* ModAug brings very large improvements in both cases when the primary modality is missing 

* Segmentation results are better for the standard U-Net when using ModAug when the two modalities are present for both experiments (cross-validation and generalization)

* The results for DLF with ModAug and both modalities are slightly below the original results without ModAug

## Ablation study

* An ablation study was conducted to investigate the contribution of the weighted voting subnet, the fine-tuning subnet as well as the atlas masking

![](/collections/images/DeepLabelFusion/Ablation.jpg) 
Figure 7 : Results of the ablation study

* Results show that the fine-tuning subnet is the most important contributor

* The weighted voting subnet also brings important improvements

* The atlas masking operation only adds marginal improvements to the overall pipeline

## Limitations

* The registration quality has a great impact on the resulting segmentation. DLF is therefore not applicable when good correspondance cannot be achieved via deformable registration

* The processing/training time is longer than when using a classical U-Net

* To evaluate the clinical application of DLF, they would need to evaluate it on other various datasets

# Conclusion

* DLF is the first end-to-end hybrid MAS / DCNN pipeline

* DLF shows segmentation results comparable to nnUNet, while having a better generalizability on unseen datasets

* The Modality Augmentation method is a promising way of dealing with missing modalities when using multimodal datasets
