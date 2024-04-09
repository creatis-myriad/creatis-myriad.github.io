---
layout: review
title: "RePaint: Inpainting using Denoising Diffusion Probabilistic Models"
tags: diffusion-model inpainting
author: "Gabriel Jacquinot"
cite:
    authors: "Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte and Luc Van Gool"
    title: "RePaint: Inpainting using Denoising Diffusion Probabilistic Models"
    venue: "(CVPR) 2022"
pdf: "https://arxiv.org/pdf/2201.09865.pdf"
---

# Notes

* Paper's code is available at this [GitHub repo](https://github.com/andreas128/RePaint)
* A re-implementation of the algorithm is available at the following [Monai prototyping repo](https://github.com/Project-MONAI/GenerativeModels/blob/main/tutorials/generative/2d_ddpm/2d_ddpm_inpainting.py)
* An introduction tutorial about denoising diffusion probabilistic models is available [Here](https://creatis-myriad.github.io/tutorials/2023-11-30-tutorial-ddpm.html) 

&nbsp;

# Highlights

* Image inpainting done thanks to Denoising Diffusion Probabilistic Model (DDPM)
* Free-form inpainting, i.e., works with a wide variety of masks
* Inpainting is done during the inference stage, by modifying the generation process, thus not requiring specific training
* The Repaint method outperforms state-of-the-art autoregressive and GAN approaches for at least five out of the six mask distributions tested

&nbsp;

# Introduction

The state-of-the art of inpainting methods mostly use a distribution of arbitrary masks to train their models. Resulting in poor generalization capabilities. The paper proposes a new method to condition the generation process. The Repaint method only takes place during the complete generation process and does not require specific training; a classic DDPM is enough!

![](/collections/images/Repaint/masks_used.jpg)

<center style="font-style:italic">
Figure 1: Illustration of the variety of masks used.
</center>

&nbsp;

# Method
## Inpainting algorithm and general idea

A quick reminder about diffusion models is needed to understand the following parts: Diffusion models are basically U-net networks, which, through training, try to predict the noise added to an image. We define a range of noise levels from $$0$$ to $$t$$, $$x_0$$ denoting an input image with no noise added and $$x_t$$ denoting a noisy image $$\sim \mathcal{N}(0, I)$$. The process of gradually adding noise to an image is called the **forward diffusion process**, and the action of gradually removing it is called the **reverse diffusion process**.
The forward diffusion process is modelized as a Markov chain with noise added at each step, we can use the following equation to generate a picture at a specific noise level:

&nbsp;

$$
q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha_t}}x_0; (1 − \bar{\alpha_t})I)\\
$$

&nbsp;

As previously mentioned, the innovation of the Repaint paper is due to its method of generation, which is conditioned by the known region of the image. They introduce some notations: the ground truth image is denoted as $$x$$ and the mask corresponding to the unknown area to inpaint as $$m$$. They define their new reverse diffusion process as follows:

&nbsp;

$$
\begin{aligned}
x^{known}_{t−1} & ∼ \mathcal{N} (\sqrt{\bar{\alpha_t}}x_0, (1 − \bar{\alpha_t})I)\\
x^{unknown}_{t−1} & ∼ \mathcal{N} (\mu_\theta(x_t, t), \Sigma_\theta(x_t, t))\\
x_{t−1} & = m \odot x^{known}_{t−1} + (1 − m) \odot x^{unknown}_{t−1}
\end{aligned}
$$

&nbsp;

Basically, the idea boils down to sampling from known and unknown regions at a given timestep. Since this sampling is done through all steps of the reverse diffusion process, the distribution of the intermediaries $$x_{t-1}$$ images matches the propriety of the distributions from a regular reverse diffusion process, thus filling the unknown region with a texture that matches the neighboring regions. The final result is an image with the known regions intact, and the unknown region filled with generated texture.

![](/collections/images/Repaint/reverse_diffusion.gif)

<center style="font-style:italic">
Figure 2: Reverse diffusion process with known region conditioning.
</center>

&nbsp;

## Resampling

As you may have noticed, the previously described reverse diffusion process does not directly restrict diffusion process generation. This can result in a coherent generated texture by itself, but incoherent regarding the general image semantic, with non-matching boundaries, for instance (cf. figure 3, with n from 1 to 5).

To overcome this problem, the authors leverage the fact that DDPM is trained to create images that lie within a data distribution and diffuse the output $$x_{t−1}$$ back to $$x_t$$. The effect of this is to incorporate some information in the $$x^{unknown}_t$$ and add noise to the image, thus creating new $$x^{known}_t$$ and $$x^{unknown}_t$$ regions that are better harmonized.

![](/collections/images/Repaint/resampling.jpg)

<center style="font-style:italic">
Figure 3: Inpainting of a dog's snout with different amounts of resampling.
</center>

&nbsp;

As you can see, the more resamples are made, the better harmonized the resulting image is. The authors noted that after about 10 resamplings, the benefits of resampling are negligible (it still adds noise to the image, so it's not completely useless). Adding those resampling steps makes the inferring process look like this:

![](/collections/images/Repaint/algorithm.jpg)

<center style="font-style:italic">
Figure 4: Reverse diffusion process pseudo-code.
</center>

&nbsp;

# Results
## Metrics

To put into perspective their results, the authors compared their method to others, which are autoregressive-based or GAN-based approaches.
All the methods were trained on two datasets: CelebA-HQ and ImageNet.
The comparisons are done using the perceptual metric LPISP, learned on the deep feature space of AlexNet, and a user study.

The user study is done by showing each voter the reference image to inpaint and two inpainted images, one using Repaint and the other using another method. The five people who had participated in the voting had to vote for the most realistic inpainting, resulting in 1000 votes per method-to-method comparison in each dataset and mask setting (Wide, Narrow, Every Second Line, Half Image, Expand, and Super-Resolve). The vote distribution shows a 95% confidence interval toward the mean vote.

![](/collections/images/Repaint/sample_test.jpg)

<center style="font-style:italic">
Figure 5: ImageNet qualitative results.
</center>

&nbsp;

![](/collections/images/Repaint/metrics.jpg)

<center style="font-style:italic">
Figure 6: CelebA-HQ and ImageNet quantitative results.
</center>

&nbsp;

As we can see, the Repaint inpainting method was most of the time chosen by voters as the most fitting inpainting method and has a correct LPIPS regarding the other methods.

&nbsp;

## Ablation Study

A huge con of the Repaint method is its computational cost. The diffusion model inference process is not fast in the first place, but if you need $$T$$ denoising steps to denoise an image, with Repaint you will need $$T \times n$$ steps to inpaint an image.

To overcome that problem, the researchers made an ablatation study, doing a trade-off between the number of resampling and the number of denoising steps for a given number of steps.

![](/collections/images/Repaint/ablation_table.jpg)

<center style="font-style:italic">
Figure 7: Quantitative results of the computational trade-off.
</center>

&nbsp;

![](/collections/images/Repaint/ablation_example.jpg)

<center style="font-style:italic">
Figure 8: Quatlitative results of the computational trade-off.
</center>

&nbsp;

Through that study, they show that for the same computational budget, it is better to increase the number of resamplings to the detriment of the number of denoising steps.

&nbsp;

# Conclusions

* The proposed inpainting method was the first one using DDPM at the time.
* The quality of the images inpainted is correct, and the method challenges state-of-the-art in this domain.
* Other promising methods have been developed since Repaint ([see also the Copaint method](https://arxiv.org/pdf/2304.03322.pdf))
* The inpainting is correct regarding the fact training a diffusion model is basically training a U-net to predict noise, instead of training a GAN.