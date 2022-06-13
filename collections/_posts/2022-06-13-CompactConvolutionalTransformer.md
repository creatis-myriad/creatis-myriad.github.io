---
layout: review
title: "Escaping the big data paradigm with compact transformers"
tags: CNN transformer classification
author: "Olivier Bernard"
cite:
    authors: "Ali Hassani, Steven Watson, Nikhil Shah, Abulikemu Abuduweili, Jiachen Li, Humphrey Shi"
    title:   "Escaping the big data paradigm with compact transformers"
    venue:   "arXiv"
pdf: "https://arxiv.org/abs/2104.05704"
---

# Notes

* This paper was accepted in a workshop at CVPR 2021
* Here are some (highly) useful links: [video](https://www.youtube.com/watch?v=AEWhf_hMBgs), [repo](https://github.com/SHI-Labs/Compact-Transformers), [blog](https://medium.com/pytorch/training-compact-transformers-from-scratch-in-30-minutes-with-pytorch-ff5c21668ed5)

# Highlights

* The objective of this work is to propose a transformer-based architecture that does not require a huge amount of data during training (just as a reminder, [VIT architecture](https://creatis-myriad.github.io/2022/06/01/VisionTransformer.html) is based on weights that have been pre-trained on over 100 million images !)
* The proposed architecture, "CCT" (Compact Convolutional Transformer) is shown to perform as well or better than CNNs for image classification on various scale datasets (ex: CIFAR10/100, Fashion-MNIST, MNIST and ImageNet)
* The proposed architecture brings a reduction of parameters of a factor of 30 during training (85 million to 3.7 million) compared to standard transformer architecture (VIT) for a better performance
* The proposed architecture brings a reduction of parameters of a factor of 3 during training (10 million to 3.7 million) compared to CNNs (ResNet1001, MobileNetV2) for a given performance
* The CCT architecture needs less than 30 minutes to be trained over their experiments on small datasets

# Methods

![](/collections/images/cct/main_diagram.jpg)

## Architecture

* Tokens are built from the input image thanks to a simple convolutional strategy. This step replaces the previous "tokenization" procedure (i.e., input image split into patches + linear projection).
* The class token strategy used in the VIT architecture is replaced by a sequence pooling procedure whose output is used as input for a simple MLP to perform classification.
* Several tests have also been done to reduce as much as possible the number of final parameters, in particular the number of blocks in the encoder, the kernel size of the input convolutional layers and the number of convolutional blocks. As an example, CCT-12/7x2 means CCT architecture with an encoder of 12 layers and 2 convolutional blocks with 7x7 convolutions to generate the input sequence of tokens.
* Although its influence is less pronounced, positional embedding is retained.

## 1st innovation: convolutional block

![](/collections/images/cct/convolutions.jpg)

* Be careful, my feeling is that this figure is partially right (feature maps should be flatten to provide an information for each token)
* $$d$$ filters (i.e., convolutions kernels) have to be learned to produce a sequence of $$d$$-dimensional kernels 
* $$ x_i = MaxPool\left( ReLU\left( Conv2d(x) \right) \right) $$
* $$x_i$$ is a feature map whose individual value corresponds to the $$i$$ component for each token.
* The number of tokens is directly linked to the image size and the size of the MaxPool operation.
* The output of this tokenization procedure is of size $$\mathbb{R}^{b \times n \times d}$$, where $$b$$ is the mini-batch size, $$n$$ is the number of tokens and $$d$$ is the embedding dimension.
* This step allows the embedding of the image into a latent representation that should be more efficient for the transformer !

## 2nd innovation: sequence pooling

The experiments have been run on a number of datasets of image classification:

* ImageNet
* ImageNet ReaL ("Reassessed Labels", from Beyer et al. 2020) [Code](https://github.com/google-research/reassessed-imagenet)
* CIFAR10/100
* Oxford Pets, Oxford Flowers
* VTAB (Zhai et al., 2019b) ("VTAB evaluates low-data transfer using 1 000 examples to diverse tasks.  The tasks are divided into three groups: Natural– tasks like the above, Pets, CIFAR, etc. Specialized– medical and satellite imagery, and Structured– tasks that require geometric understanding like localization.") [Blog post](https://ai.googleblog.com/2019/11/the-visual-task-adaptation-benchmark.html)

# Results

As seen in the table below, ViT performs slightly better than a very large ResNet, and does so using significantly less FLOPS (for the fine-tuning phase).

![](/collections/images/vit/tab2.jpg)

However, the pre-training has to involve a very large number of training samples: when this number exceeds 100 million, ViT starts to shine. Else, the ResNet performs better. See below:

![](/collections/images/vit/fig3-4.jpg)

ViT also compares favourably in terms of pre-training FLOPS, as seen below:

![](/collections/images/vit/fig5.jpg)

The "Hybrid" approach uses CNN feature vectors as tokens; it is not considered very important by the authors.

The figures below serve to inspect the vision transformer architecture:

![](/collections/images/vit/fig7.jpg)

# Conclusions

The Vision Transformer is an architecture that can outperform CNNs given datasets in the 100M-image range. It required less FLOPS to train than the CNNs used in this paper.


