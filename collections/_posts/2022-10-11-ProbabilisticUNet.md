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

* The objectif if this article is to learn a distribution over segmentations given an input.
* The main innovation concerns the design of a generative segmentation model based on a combination of a U-Net with a conditional VAE.
* The proposed framework is able to also learn hypotheses that have a low probability and to predict them with the corresponding frequency .


# Method

## Architecture

* The architecture is based on the ***conditional VAE*** whose details are provided in the following [tutorial](https://creatis-myriad.github.io/tutorials/2022-09-12-tutorial-cvae.html).
* The innovation comes from the use of a U-Net architecture to model the distribution $$p(y \vert x,z)$$.
* The latent vector $$z$$ is first passed to a decoder to produce a $$N$$-channel feature map with the same spatial dimensions as the segmentation map. This feature map is then concatenated with the last activation map of a U-Net before being convolved by a last layer to produce the final segmentation map with the desired number of classes.

The image below provides an overview of the architecture deployed during training. The distributions $$p(z \vert x,y)$$, $$p(z \vert x)$$ and $$p(y \vert x,z)$$ displayed in blue are modeled by three distinct neural networks.

![](/collections/images/probabilistic_unet/proba_unet_training.jpg)


The following image illustrates the use of the architecture during inference.

![](/collections/images/probabilistic_unet/proba_unet_inference.jpg)


## Implementation details

* The dimension of the latent space has been experimentally fixed at $$M=6$$ and is kept fixed in all experiments.
* The difference between the predicted segmentation and the ground truth is optimized using a Cross Entropy loss.
* The training is done from scratch with randomly initialized weights 
* When drawing $$m$$ samples for the same input image, the prior network output and U-Net feature activations can be reused without the need for recomputation, making the overall strategy computationally efficient.

## Performance measures

* The performance of the methods was evaluated by comparing distributions of segmentations
* This was done through the ***generalized energy distance*** whose expression is given below:

$$D^2_{GED}(P_{gt},P_{out}) = 2 \, \mathbb{E} \left[d\left(S,Y\right) \right] - \mathbb{E} \left[d\left(S,S'\right) \right] - \mathbb{E} \left[d\left(Y,Y'\right) \right]$$

* $$d$$ is a distance measure, $$Y$$ and $$Y'$$ are independent samples from the ground truth distribution $$P_{gt}$$, $$S$$ and $$S'$$ are independent samples from the predicted distribution $$P_{out}$$. 
* The distance measure is based on the $$IoU$$ metric and is defined as follows: $$d(x,y)=1-IoU(x,y)$$.

&nbsp;

# Results

## LIDC-IDRI dataset

* 1010 2D+slices CT scan of lungs with lesions 
* For each scan, 4 radiologists (from a total of 12) provided annotation masks for lesions that they independently detected
* the CT scans were resampled to $$0.5 \, \text{mm} \times 0.5 \, \text{mm}$$ in-plane resolution and cropped 2D images ($$180 \times 180$$ pixels) centered at the lesion positions.
* This resulted in $$8882$$ images in the training set, $$1996$$ images in the validations set and $$1992$$ images in the test set.
* Because the experts can disagree, up to 3 masks per image can be empty.


As shown in the figure below, the probabilistic U-Net network outperforms state-of-the-art methods in effectively representing the ground truth distribution.

![](/collections/images/probabilistic_unet/LIDC_results.jpg)

## Analysis of the latent space

* The authors proposed to visualize the latent space by arranging the samples to represent their corresponding position in a 2D plane, i.e. the dimension of the latent space is set to $$2$$ and each latent variable is normalized by the inferred mean and standard deviation computed from the tested image: $$\hat{z}=\left(z-\mu\right)/\sigma$$.
* This allows to interpret how the model ends up structuring the space to solve the given tasks.

![](/collections/images/probabilistic_unet/latent_space_lidc.jpg)


# Conclusions

TODO


