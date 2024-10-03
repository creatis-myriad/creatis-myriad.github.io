---
layout: review
title: "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture"
tags: self-supervised
author: "Celia Goujat"
cite:
    authors: "Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, Nicolas Ballas"
    title: "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture"
    venue: "Conference on Computer Vision and Pattern Recognition (CVPR), 2023"
pdf: "https://openaccess.thecvf.com/content/CVPR2023/papers/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.pdf"
---

# Highlights

*   Introduce a joint-embedding predictive architecture for images (I-JEPA).
*  The goal is to learn highly semantic image representations without the use of hand-crafted view augmentations.

&nbsp;

# Introduction

Self-supervised learning is a method of representation learning where the model attempts to understand the relationships between its inputs.
Currently, there are two main approaches:

* Invariance-based pretraining (DINO[^1], SimCLR v2[^2]) can provide high-level semantic representations but may lead to decreased performance in certain downstream tasks 
(like segmentation) or with new data distributions.
* Generative pretraining (MAE[^3],iBOT[^4], SimMIM[^5]) requires less prior knowledge and offers greater generalizability but results in lower semantic-level representations 
and underperforms in off-the-shelf evaluations.

Some recent methods are hybrid (MSN[^6], data2vec[^7]), using mask generation and contrastive loss. 
However, most of the methods rely on hand-crafted image transformations.

The goal of the authors is to enhance the semantic quality of self-supervised representations while ensuring applicability to a broader range of tasks.
I-JEPA does not rely on additional prior knowledge encoded through image transformations, thereby reducing bias.

&nbsp;

# Methodology

I-JEPA is similar to the generative masked autoencoders (MAE) method, with two main differences:
- I-JEPA is non-generative: it focuses only on predicting the representations of target blocks from context blocks, rather than reconstructing the original data.
- Predictions are made in an abstract representation space (or feature space) rather than directly in the pixel or token space.

<div style="text-align:center">
<img src="/collections/images/SSL_with_joint_embedding/method.jpg" width=600></div>
<p style="text-align: center;font-style:italic">Figure 1. I-JEPA: The Image-based Joint-Embedding Predictive
Architecture.</p>

* **Targets:** From a sequence of N non-overlapping patches from an input image y, the target-encoder $$f_{\bar{\theta}}$$
obtains a corresponding patch-level representation $$s_y = \{s_{y_1} , \dots , s_{y_N} \}$$. Randomly sample M (possibly overlapping) blocks from the target representation $$s_y$$. 
$$B_i$$ is the mask of the block i and $$s_y(i)=\{s_{y_j}\}_{j \in B_i}$$ is its representation. These M representation blocks are masked, they are the target of the model.

* **Context:** Sample a single block x from the image y and remove overlapping regions with target blocks. 
Then use context-encoder $$f_{\theta}$$ to get the context representation. $$B_x$$ is the mask of the block x and $$s_x=\{s_{x_j}\}_{j \in B_x}$$ is its representation.

* **Predictions:** For each target $$s_y(i)$$ use output of the context encoder $$s_x$$ along with a mask token for each patch of the target. 
The predictor $$g_{\Phi}$$ generates patch-level predictions $$s_{\hat{y}(i)}=\{s_{\hat{y}_j}\}_{j \in B_i}$$. 

* **Loss:** $$\mathcal{L} = \frac{1}{M} \sum_{i=1}^{M} \mathcal{D}(s_{\hat{y}(i)},s_y(i)) =  \frac{1}{M} \sum_{i=1}^{M} \sum_{j \in B_i} \lVert s_{\hat{y}(j)} - s_y(j) \lVert_2^2$$
The parameters of $$\Phi$$ and $$\theta$$ are learned through gradient-based optimization while the parameters of $$\bar{\theta}$$ are updated using EMA (Exponential Moving Average) of the $$\theta$$ parameters. 

A Vision Transformer (ViT) architecture is used for the context encoder, target encoder, and predictor.

&nbsp;

# Results

## Image classification

* After self-supervised pretraining, the model weights are frozen and a linear classifier is trained on top using the full ImageNet-1K training set:

<div style="text-align:center">
<img src="/collections/images/SSL_with_joint_embedding/results_ImageNet_classif.jpg" width=600></div>
<p style="text-align: center;font-style:italic">Figure 2.  Linear-evaluation on ImageNet-1k (the ViT-
H/16 448 is pretrained at a resolution of 448 x 448, the others at 224 x 224).</p>

<div style="text-align:center">
<img src="/collections/images/SSL_with_joint_embedding/results_lowShotImageNet_classif.jpg" width=600></div>
<p style="text-align: center;font-style:italic">Figure 3. Semi-supervised evaluation on ImageNet-1K using only 1% of the available labels (12 or 13 images per class). Models are
adapted via fine-tuning or linear-probing, depending on whichever works best for each respective method. The ViT-
H/16 448 is pretrained at a resolution of 448 x 448, the others at 224 x 224.</p>

<div style="text-align:center">
<img src="/collections/images/SSL_with_joint_embedding/results_linearProbeTransfer_classif.jpg" width=600></div>
<p style="text-align: center;font-style:italic">Figure 4. Linear-evaluation on downstream image classification tasks.</p>

## Local prediction tasks

<div style="text-align:center">
<img src="/collections/images/SSL_with_joint_embedding/results_linearProbeTransfer_localPredictionTasks.jpg" width=600></div>
<p style="text-align: center;font-style:italic">Figure 5. Linear-evaluation on downstream low-level tasks consisting of object counting (Clevr/Count) and depth prediction (Clevr/Dist).</p>

## Ablation study

<div style="text-align:center">
<img src="/collections/images/SSL_with_joint_embedding/ablation_datasetModelSize.jpg" width=800></div>
<p style="text-align: center;font-style:italic">Figure 6. Evaluating impact of pre-training dataset size and model size on transfer tasks. </p>

<div style="text-align:center">
<img src="/collections/images/SSL_with_joint_embedding/ablation_maskingStrategies.jpg" width=800></div>
<p style="text-align: center;font-style:italic">Figure 7. Linear evaluation on ImageNet-1K using only 1% of the available labels after I-JEPA pretraining of
a ViT-B/16 for 300 epochs. Comparison of proposed multi-block masking strategy. In "rasterized masking" the image is split into
four large quadrants; one quadrant is used as a context to predict the other three quadrants. In "block masking", the target is a single image
block and the context is the image complement. In "random masking", the target is a set of random image patches and the context is the
image complement.</p>

<div style="text-align:center">
<img src="/collections/images/SSL_with_joint_embedding/ablation_targetsStrategies.jpg" width=600></div>
<p style="text-align: center;font-style:italic">Figure 8. Linear evaluation on ImageNet-1K using only 1% of the available labels. The semantic level of the
I-JEPA representations degrades significantly when the loss is applied in pixel space, rather than representation space.</p>

## General performances

<div style="text-align:center">
<img src="/collections/images/SSL_with_joint_embedding/performance_gpuHours.jpg" width=600></div>
<p style="text-align: center;font-style:italic">Figure 9. Semi-supervised evaluation on ImageNet-1K
1% as a function of pretraining GPU hours. I-JEPA requires less
compute than previous methods to achieve strong performance.</p>


<div style="text-align:center">
<img src="/collections/images/SSL_with_joint_embedding/performance_vizualizationRepresentations.jpg" width=800></div>
<p style="text-align: center;font-style:italic">Figure 10. Visualization of I-JEPA predictor representations. For each image: first column contains the original image; second column
contains the context image, which is processed by a pretrained I-JEPA ViT-H/14 encoder. Green bounding boxes in subsequent columns contain samples from a 
generative model. The generative model decodes the output of the predictor, conditioned on positional mask tokens corresponding to the location of the green bounding box.</p>

&nbsp;

# Conclusions

*  In contrast to view-invariance-based methods, I-JEPA learns semantic image representations without relying on hand-crafted data augmentations. 
*  By predicting in representation space, the model converges faster than pixel reconstruction methods and achieves high-level semantic representations.

&nbsp;

# References

[^1]: Caron, Mathilde, et al. "Emerging properties in self-supervised vision transformers." Proceedings of the IEEE/CVF international conference on computer vision. 2021, [link to paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.pdf)
[^2]: Chen, Ting, et al. "Big self-supervised models are strong semi-supervised learners." Advances in neural information processing systems 33 (2020): 22243-22255, [link to paper](https://proceedings.neurips.cc/paper/2020/file/fcbc95ccdd551da181207c0c1400c655-Paper.pdf)
[^3]: He, Kaiming, et al. "Masked autoencoders are scalable vision learners." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022, [link to paper](https://openaccess.thecvf.com/content/CVPR2022/papers/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.pdf)
[^4]: Zhou, Jinghao, et al. "ibot: Image bert pre-training with online tokenizer." arXiv preprint arXiv:2111.07832 (2021). [link to paper](https://arxiv.org/pdf/2111.07832)
[^5]: Xie, Zhenda, et al. "Simmim: A simple framework for masked image modeling." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022, [link to paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Xie_SimMIM_A_Simple_Framework_for_Masked_Image_Modeling_CVPR_2022_paper.pdf)
[^6]: Assran, Mahmoud, et al. "Masked siamese networks for label-efficient learning." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2022, [link to paper](https://arxiv.org/pdf/2204.07141)
[^7]: Baevski, Alexei, et al. "Data2vec: A general framework for self-supervised learning in speech, vision and language." International Conference on Machine Learning. PMLR, 2022, [link to paper](https://proceedings.mlr.press/v162/baevski22a/baevski22a.pdf)
