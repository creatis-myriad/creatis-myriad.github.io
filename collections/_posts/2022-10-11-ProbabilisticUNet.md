---
layout: review
title: "A probabilistic U-Net for the segmentation of ambiguous images"
tags: UNet, conditional VAE, variational autoencoders
author: "Olivier Bernard"
cite:
    authors: "Simon A. A. Kohl, Bernardino Romera-Paredes, Clemens Meyer, Jeffrey De Fauw, Joseph R. Ledsam, Klaus H. Maier-Hein, S. M. Ali Eslami, Danilo Jimenez Rezende, Olaf Ronneberger"
    title:   "A probabilistic U-Net for the segmentation of ambiguous images"
    venue:   "NeurIPS 2018"
pdf: "https://proceedings.neurips.cc/paper/2018/file/473447ac58e1cd7e96172575f48dca3b-Paper.pdf"
---

# Notes

* Here are some (highly) useful links: [video](https://www.youtube.com/watch?v=-cfFxQWfFrA&ab_channel=SimonKohl), [repo](https://github.com/stefanknegt/Probabilistic-Unet-Pytorch), [NeurIPS](https://proceedings.neurips.cc/paper/2018/hash/473447ac58e1cd7e96172575f48dca3b-Abstract.html)

# Highlights

* TODO

## Method

* The architecture is based on the ***conditional VAE*** whose details are provided in the following [tutorial](https://creatis-myriad.github.io/tutorials/2022-09-12-tutorial-cvae.html).
* The innovation comes from the use of a U-Net architecture to model the distribution $$p(y \vert x,z)$$.
* The latent vector $$z$$ is first passed to a decoder to produce a $$N$$-channel feature map with the same spatial dimensions as the segmentation map. This feature map is then concatenated with the last activation map of a U-Net before being convolved by a last layer to produce the final segmentation map with the desired number of classes.

The image below provides an overview of the architecture deployed during the training. The distributions $$p(z \vert x,y)$$, $$p(z \vert x)$$ and $$p(y \vert x,z)$$ displayed in blue are modeled by three distinct neural networks.

![](/collections/images/probabilistic_unet/proba_unet_training.jpg)


The following image illustrates the use of the architecture during inference.

![](/collections/images/probabilistic_unet/proba_unet_inference.jpg)

# Results

TODO

# Conclusions

TODO


