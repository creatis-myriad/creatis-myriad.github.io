---
layout: review
title: "Adding Conditional Control to Text-to-Image Diffusion Models"
tags: Diffusion, autoencoders, representation, conditioning
author: "Romain Deleat-besson"
cite:
    authors: "Lvmin Zhang, Anyi Rao, Maneesh Agrawala"
    title: "Adding Conditional Control to Text-to-Image Diffusion Models"
    venue: "CVPR 2023"
pdf: "https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.pdf"
---



# Highlights

* ControlNet is a method to spatially conditioned text-to-image Diffusion or Latent Diffusion models.
* Used a lot in different context (even in medical field).
* The method is not complex and already implemented on MONAI [here](https://github.com/Project-MONAI/GenerativeModels/blob/main/tutorials/generative/2d_controlnet/2d_controlnet.ipynb)


# Introduction

Diffusion models can create visually stunning images by typing in a text prompt, but they are limited in the control they provide over the spatial composition of the image.
In particular, it is difficult to accurately express complex layouts and poses using text prompts alone.

ControlNet is a neural network architecture that can add spatial control (e.g. edge maps, human pose skeletons, segmentation maps, depth, normals, etc.) to large pre-trained text-to-image diffusion models.



# Methods

ControlNet is a neural network architecture designed to maintain the quality and robustness of a large pretrained model, specifically a text-to-image Stable Diffusion model [1].

To achieve this, the parameters of the original model are locked while a trainable copy of the encoding layers is created, leveraging Stable Diffusion as a strong backbone. The trainable copy is connected to the original locked model through zero convolution layers, where the weights are initialized to zero and gradually adapt during training. This design prevents the introduction of disruptive noise into the deep features of the diffusion model at the early training stages, thereby preserving the integrity of the pretrained backbone in the trainable copy.

They demonstrate that ControlNet effectively conditions Stable Diffusion using various input modalities, including Canny edges, Hough lines, user scribbles, human key points, segmentation maps, shape normals, and depth information.


<div style="text-align:center"><img src="/collections/images/ControlNet/Fig_1.jpg" width=1500></div>


The complete ControlNet then computes:

$$ y_c = \mathcal{F}(x; \Theta) + \mathcal{Z} \big( \mathcal{F} \big(x + \mathcal{Z}(c; \Theta_{z1}); \Theta_c \big); \Theta_{z2} \big) $$

* $$ \mathcal{Z}(.;.) $$ is a $$ 1 \times 1 $$ convolution layer with both weight and bias initialized to zeros.
* $$ \Theta_{z1} $$ and $$ \Theta_{z2} $$ are the parameters of the first and second *zero convolution* layer.


<div style="text-align:center"><img src="/collections/images/ControlNet/Fig_1bis.jpg" width=1500></div>



The objective function is:

$$ \mathcal{L} = \mathbb{E}_{z_0, t, c_t, c_f, \epsilon \sim \mathcal{N}(0,1)} \left[ \left\| \epsilon - \epsilon_{\theta} (z_t, t, c_t, c_f) \right\|_2^2 \right] $$

* $$ \epsilon $$ is the real noise sampled from a standard normal distribution
* $$ \epsilon_{\theta} $$ is the noise predicted by the neural network, parameterized by $$ \theta $$
* $$ z_t $$ is the feature map $$ z $$ at timestep $$ t $$
* $$ t $$ is the timestep
* $$ c_t $$ is the text prompt conditioning
* $$ c_f $$ is the ControlNet conditioning



# Results

## CFG Resolution Weighting and Composition

<div style="text-align:center"><img src="/collections/images/ControlNet/Fig_2.jpg" width=1500></div>

Classifier-free guidance was used in Stable Diffusion model to generate high quality images.
It is formulated as:

$$ \epsilon_{\text{prd}} = \epsilon_{\text{uc}} + \beta_{\text{cfg}} (\epsilon_c - \epsilon_{\text{uc}}) $$ 

* $$ \epsilon_{\text{prd}} $$ is the model’s final output.
* $$ \epsilon_{\text{uc}} $$ is the unconditional output.
* $$ \epsilon_c $$ is the conditional output.
* $$ \beta_{\text{cfg}} $$ is user-specified weight.


They had to change the original CFG (as shown in figure 5) by adding the conditioning image to $$ \epsilon_c $$ and multiply a weight $$ w_i $$ (based on the resolution of each block) to each connection between Stable Diffusion and ControlNet.

$$ w_i = 64/h_i $$ where $$ h_i $$ is the $$ i^{th} $$ block. ($$ h_1 = 8 $$, $$ h_2 = 16 $$ ...)

By doing so, they reduced the strengh of the CFG guidance and called their approch: CFG Resolution Weighting.


## Examples of spatial conditioning and Quantitative results

Examples:

<div style="text-align:center"><img src="/collections/images/ControlNet/Fig_3.jpg" width=1500></div>


Quantitative results:

<div style="text-align:center"><img src="/collections/images/ControlNet/Tab_1.jpg" width=1500></div>


## Dataset size and Input interpretation

<div style="text-align:center"><img src="/collections/images/ControlNet/Fig_4.jpg" width=1500></div>


# Conclusion


ControlNet is a neural network framework designed to facilitate conditional control in large pre-trained text-to-image diffusion models. It integrates the pre-trained layers of these models to construct a high-capacity encoder for learning specific conditioning signals, while employing "zero convolution" layers to mitigate noise propagation during training. 

Empirical evaluations confirm its effectiveness in modulating Stable Diffusion under different conditioning scenarios, both with and without textual prompts. 
The model proposed by the authors is likely to be applicable to a wider range of conditions, and facilitate relevant applications.



# References

[1] [Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (CVPR 2022). High-resolution image synthesis with latent diffusion models.](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf)

