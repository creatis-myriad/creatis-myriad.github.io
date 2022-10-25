---
layout: review
title: "A hierarchical probabilistic U-Net for modeling multi-scale ambiguities"
tags: UNet, conditional VAE, variational autoencoders
author: "Olivier Bernard"
cite:
    authors: "Simon A. A. Kohl, Bernardino Romera-Paredes, Klaus H. Maier-Hein, Danilo Jimenez Rezende, S. M. Ali Eslami, Pushmeet Kohli, Andrew Zisserman, Olaf Ronneberger"
    title:   "A hierarchical probabilistic U-Net for modeling multi-scale ambiguities"
    venue:   "NeurIPS workshop 2019"
pdf: "https://proceedings.neurips.cc/paper/2018/file/473447ac58e1cd7e96172575f48dca3b-Paper.pdf"
---

# Notes

* Here are some (highly) useful links: [video](https://crossminds.ai/video/a-hierarchical-probabilistic-u-net-for-modeling-multi-scale-ambiguities-6070a767fa08279acdb21414/), [repo](https://github.com/deepmind/deepmind-research/tree/master/hierarchical_probabilistic_unet)

# Highlights

* The objective of this paper is to develop a generative model for semantic segmentation able to learn complex-structured conditional distribution.
* The innovation comes from the modelling of a coarse-to-fine hierarchy of latent variables to improve fidelity to fine structures in the models' samples and reconstructions.
* The proposed framework is capable of modelling distributions over segmentations with factors of variations across space and scale.


# Method

## Appetizer

The following animation illustrates the capacity of the method in modelling complex structured distributions across scales.

![](/collections/images/hierarchical_probabilistic_unet/animation.gif)

## Architecture

* The architecture is based on the ***conditional VAE*** whose details are provided in the following [tutorial](https://creatis-myriad.github.io/tutorials/2022-09-12-tutorial-cvae.html).
* The main innovation comes from the following hierarchical modelling of the latent space:

$$p\left(\boldsymbol{z} \vert x\right) = p\left(z_0,\ldots,z_L \vert x\right) = p\left(z_L \vert z_{<L},x\right) \cdot \, \ldots \, \cdot p\left(z_0 \vert x\right)$$

$$q\left(\boldsymbol{z} \vert x,y\right) = q\left(z_0,\ldots,z_L \vert x,y\right) = q\left(z_L \vert z_{<L},x,y\right) \cdot \, \ldots \, \cdot q\left(z_0 \vert x,y\right)$$

In the particular case where $$L=2$$, the above equation can be put in the following form:

$$p\left(\boldsymbol{z} \vert x\right) = p\left(z_2 \vert z_1,z_0,x\right) \cdot p\left(z_1 \vert z_0,x\right) \cdot p\left(z_0 \vert x\right)$$

$$q\left(\boldsymbol{z} \vert x,y\right) = q\left(z_2 \vert z_1,z_0,x,y\right) \cdot q\left(z_1 \vert z_0,x,y\right) \cdot q\left(z_0 \vert x,y\right)$$

* Taking into account the hierarchical modelling, a new ELBO objective with a relative weighting factor $$\beta$$ was formulated as follows:

$$\mathcal{L}_{ELBO} = \mathbb{E}_{z\sim q(z \vert x,y)} [CE\left( y,\hat{y}\right)] + \beta \cdot \sum_{i=0}^{L} \mathbb{E}_{z_i\sim q(z_i \vert z_{<i},x,y)} [D_{KL}(q(z_i \vert z_{<i},x,y) \parallel p(z_i \vert z_{<i},x))]$$

* The authirs observed that the minimization of $$\mathcal{L}_{ELBO}$$ leads to sub-optimally results. For this reason, they used the recently proposed $$GECO$$ loss:

$$\mathcal{L}_{GECO} = \lambda \cdot \left( \mathbb{E}_{z\sim q(z \vert x,y)} [CE\left( y,\hat{y}\right)] - \kappa \right) + \sum_{i=0}^{L} \mathbb{E}_{z_i\sim q(z_i \vert z_{<i},x,y)} [D_{KL}(q(z_i \vert z_{<i},x,y) \parallel p(z_i \vert z_{<i},x))]$$

where $$\kappa$$ is chosen as the desired reconstruction error and $$\lambda$$ is a Lagrange multiplier that is updated as a function of the exponential moving average of the reconstruction contraint.

> This formulation initially puts high pressure on the reconstruction and once the desired $$\kappa$$ is reached it increasingly moves the pressure over on the KL-term.

* Finally, the prior and the generator networks are based on the same U-Net architecture, which results in parameter and run-time savings.

The overall architecture is given below:

![](/collections/images/hierarchical_probabilistic_unet/overall_architecture.jpg)

![](/collections/images/hierarchical_probabilistic_unet/inference_phase.jpg)

&nbsp;

This architecture can be difficult to understand at first sight. Therefore I show below the different parts of the network with reference to the formalism of the [conditional VAE](https://creatis-myriad.github.io/tutorials/2022-09-12-tutorial-cvae.html)

![](/collections/images/hierarchical_probabilistic_unet/prior_network.jpg)

![](/collections/images/hierarchical_probabilistic_unet/posterior_network.jpg)

![](/collections/images/hierarchical_probabilistic_unet/generative_network.jpg)

&nbsp;

## Implementation details

* U-Nets are composed by res-blocks. 

> Without the use of res-blocks, the KL-terms between distributions at the begining of the hiearchy often become $$\, 0$$ early in the training, essentially resulting in uninformative and thus unused latents. 

* The number of latent scales is chosen empirically such as to allow for a sufficiently granular effect of the latent hierarchy. For the tasks and image resolutions considered, the authors found 3 to 5 latent scales to work well.

## Performance measures

* TODO

&nbsp;

# Results

## LIDC-IDRI dataset

* 1010 2D+slices CT scan of lungs with lesions 
* For each scan, 4 radiologists (from a total of 12) provided annotation masks for lesions that they independently detected
* the CT scans were resampled to $$0.5 \, \text{mm} \times 0.5 \, \text{mm}$$ in-plane resolution and cropped 2D images ($$180 \times 180$$ pixels) centered at the lesion positions.
* This resulted in $$8882$$ images in the training set, $$1996$$ images in the validations set and $$1992$$ images in the test set.
* Because the experts can disagree, up to 3 masks per image can be empty.

TODO

&nbsp;


# Conclusions

* TODO


