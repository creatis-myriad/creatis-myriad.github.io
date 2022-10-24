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
* The main innovation concerns the design of a generative segmentation model with a conditional VAE that uses a hierarchical latent space decomposition.
* The proposed framework is capable of modelling distributions over segmentations including independently varying scales and locations.
* The proposed framework is capable of modelling factors of variations across space and scale.


# Method

## Appetizer

The following animation illustrates the capacity of the method in modelling factors of variations across space and scale:

![](/collections/images/hierarchical_probabilistic_unet/animation.gif)

## Architecture

* The architecture is based on the ***conditional VAE*** whose details are provided in the following [tutorial](https://creatis-myriad.github.io/tutorials/2022-09-12-tutorial-cvae.html).
* The innovation comes from the modelling of hiearchy in the latent space

&nbsp;

## Implementation details

* TODO

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


