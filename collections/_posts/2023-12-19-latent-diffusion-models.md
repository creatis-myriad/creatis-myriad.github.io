---
layout: review
title: "High-resolution image synthesis with latent diffusion models"
tags: diffusion model, generative model
author: "Celia Goujeat, Olivier Bernard"
cite:
    authors: "Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer"
    title:   "High-resolution image synthesis with latent diffusion models"
    venue: "IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2022"
pdf: "https://arxiv.org/pdf/2112.10752.pdf"
---

# Notes

* Code is available on this [GitHub repo](https://github.com/CompVis/latent-diffusion)


# Highlights

* Diffusion models (DMs) are applied in the latent space of powerfull pretrained autoencoders
* Allows to reach a good compromise between complexity reduction and details preservation
* Introduce cross-attention layers into the model architecture for general conditioning inputs such as text

* The model significantly reduces computational requirement compared to pixel-based DMs
* The model achieves new SOTA for image inpainting and class-conditional image synthesis
* The model achieves highly competitive results on text-to-image synthesis, unconditional image generation and super-resolution


&nbsp;

# Introduction

* DMs are computationally demanding since it requires repeated  function evaluation in the high-dimensional space of RGB images

![](/collections/images/ddpm/ddpm_overview_complete.jpg)

<!--
&nbsp;

![](/collections/images/ddpm/ddpm_scheme_for_deep_learning.jpg)

&nbsp;

![](/collections/images/ddpm/ddpm_architecture_1.jpg)

&nbsp;

![](/collections/images/ddpm/ddpm_architecture_2.jpg)
-->

See [the tutorial on DDPM](https://creatis-myriad.github.io/tutorials/2023-11-30-tutorial-ddpm.html) for more information.

&nbsp;

The figure below shows the rate-distorsion trade-off of a trained model. Learning can be divided into two stages:

<div style="text-align:center">
<img src="/collections/images/latent-DM/distortion-rate.jpg" width=400></div>
<p style="text-align: center;font-style:italic">Figure 1. Illustrating perceptual and semantic compression.</p>


* Most bits of a digital image correspond to imperceptible details, leading to unnecessary expensive optimization and inference  

> If a latent space without imperceptible details can be learned independently, DMs can then be applied efficiently from this space to only focus on semantic properties

&nbsp;

# Methodology

## Perceptual image compression

* A perceptual compression model based on previous work [1] is used to efficiently encode images
* It consists in an auto-encoder trained by combinaison of a perceptual loss and a patch-based adversarial objective
* The overall objective of the cited paper is more complex than only computed an efficient latent space (high-resolution image synthesis based on transformer), but as far as I understand the pre-trained encoder/decoder parts are available and directly used in latent DM formalism. This paper should be the subject of a future post!

![](/collections/images/latent-DM/perceptual-image-compression.jpg)

&nbsp;

## Model artchitecture

The latent diffusion model is composed of 3 main parts:
* an encoder $$E$$ / decoder $$D$$ module which allows to go from the pixel space to a latent space which is perceptually equivalent, but offers significantly reduced computational complexity
* a time-conditional U-Net for the denoising parts $$\epsilon_{\theta}(z_t,t)$$.
* a conditioning module to efficiently encode and propagate an additional source of information $$y$$ through cross-attention mechanisms


![](/collections/images/latent-DM/latent-DM-architecture.jpg)

&nbsp;

The learning of the LDM without the cross-attention module can be modeled as:

$$\mathcal{L}_{LDM} := \mathbb{E}_{z \sim E(x), \epsilon \sim \mathcal{N}(0,\mathbf{I}), t \sim [1,T]} \left[ \| \epsilon_t - \epsilon _{\theta}(z_t,t)\|^2 \right]$$

&nbsp;

## Conditioning mechanisms

* a domain specific encoder $$\tau_{\theta}$$ that projects $$y$$ to an intermediate representation $$\tau_{\theta}(y)$$ is introduced
* this intermediate representation is then mapped to the intermediate layers of the Unet via a cross-attention layer as illustrated below:

![](/collections/images/latent-DM/cross-attention.jpg)

&nbsp;

The learning of the LDM with the cross-attention module can be modeled as:

$$\mathcal{L}_{LDM} := \mathbb{E}_{z \sim E(x), y, \epsilon \sim \mathcal{N}(0,\mathbf{I}), t \sim [1,T]} \left[ \| \epsilon_t - \epsilon _{\theta}(z_t,t,\tau_{\theta}(y))\|^2 \right]$$

&nbsp;

# Results

* Qualitative and quantitative evaluation on COCO 2017 dataset. 118K training images and 5K validation images.
* They are 7 instances per image on average and up to 63 instances in a single image of the training set.
* The maximum number of predictions, $$N$$, is consequently set to 100.
* Results are compared to Faster R-CNN, the strongest baseline for real-time object detection on natural images

<div style="text-align:center">
<img src="/collections/images/detr/performances_coco.jpg" width=600></div>
<p style="text-align: center;font-style:italic">Table 1. Performances comparison with Faster R-CNN on COCO 2017 dataset.</p>

Recall that FFNs prediction heads and Hungarian loss were used at each decoder stage during training. The authors computed the performances at each decoder layer, showing that the performances of the deepest layers are not improved by bounding boxes post-proccessing techniques such as Non-Maximum Suppression.

<div style="text-align:center">
<img src="/collections/images/detr/ablation_layers.jpg" width=400></div>
<p style="text-align: center;font-style:italic">Figure 3. Performances at each decoder stage.</p>

&nbsp;

The two figures below show the attention of the encoder and the decoder for a reference point or for the final predicted object.

<div style="text-align:center">
<img src="/collections/images/detr/attention_encoder.jpg" width=600></div>
<p style="text-align: center;font-style:italic">Figure 4. Encoder self-attention for several reference points (in red).</p>

<div style="text-align:center">
<img src="/collections/images/detr/attention_decoder.jpg" width=600></div>
<p style="text-align: center;font-style:italic">Figure 5. Decoder attention for every predicted object.</p>

&nbsp;

With small modifications that create a FPN-like network, DETR can also work for panoptic segmentation (details not explained here, we only provide an exemple of visual results below)

<div style="text-align:center">
<img src="/collections/images/detr/segmentation.jpg" width=600></div>
<p style="text-align: center;font-style:italic">Figure 6: Qualitative results for panoptic segmentation generated by DETR-R101.</p>

&nbsp;

# Conclusion

DEtection TRansformer (DETR) is a new transformer-based model to perform object detection in an end-to-end fashion. The main idea of the paper is to force unique predictions via bipartite matching that finds the optimal assignment between the predicted bounding boxes and the ground truth, avoiding the use of surrogate components or post-processing like non-maximum suppression or anchors.

&nbsp;

# References
\[1\] P. Esser, R. Rombach, B. Ommer, *Taming transformers for high-resolution image synthesis*, CoRR 2022, [link to paper](https://arxiv.org/pdf/2012.09841.pdf)




