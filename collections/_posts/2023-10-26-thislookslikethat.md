---
layout: review
title: "This Looks Like That: Deep Learning for Interpretable Image Recognition"
tags: interpretability classification uncertainty ultrasound
author: "Thierry Judge"
cite:
    authors: "Chaofan Chen, Oscar Li, Chaofan Tao, Alina Jade Barnett, Jonathan Su, Cynthia Rudin"
    title:   "This Looks Like That: Deep Learning for Interpretable Image Recognition"
    venue:   "Neurips 2019"
pdf: "https://arxiv.org/pdf/1806.10574.pdf"
---

# Notes
* Spotlight presentation (top 3% of papers) at NeurIPS 2019. 
* Link to the code [here](https://github.com/cfchen-duke/ProtoPNet).

# Highlights
* The authors propose a interpretable classification model to classify aortic stenosis (AS). 
* The model is a prototype-based model which analyzed the similarity between the input and a set of prototypes. 

# Introduction

Most interpretable models fall under two categories: posthoc interpretability and attention-based interpretability. 


![](/collections/images/thislookslikethat/fig1.jpg)
*Figure 1: Example.*


# Method 

## Architecture 

![](/collections/images/thislookslikethat/fig2.jpg)

*Figure 2: ProtoPNet architecture.*


The model architecture is illustrated in figure 2. It consists of three main blocks. 

* $$f$$ is a convolutional neural network with weights $$w_{conv}$$.
* $$g_{\mathbf{P}}$$ is a prototype layer.
* $$h$$ is a fully connected layer with a weight matrix $$w_h$$.

Given an input image $$x \in R^{224 \times 224 \times 3}$$, convolutional features $$z = f(x) \in R^{H \times W \times D }$$ 
are computed ($$H = W = 7$$, $$D$$ is chosen from {128, 256, 512} with cross-validation). 

The prototype layer contains $$m_k$$ prototypes for each class $$k$$. In practice all classes have 10 prototypes. 
In total the model learns a total of $$m$$ prototypes defined by 
$$\mathbf{P} = \{ \mathbf{p}_j \}_{j=1}^m$$ with shape $$H_1 \times W_1 \times D$$  such that 
$$H_1 \leq H$$ and $$W_1 \leq W$$ (in the experiments, $$H_1 = W_1 = 1$$). This means that a prototype can represent the
activation of a patch of the convolutional features which corresponds to a real patch in the image. 

Given $$z$$, $$g_{\mathbf{P}}$$ computes the $$L^2$$ distance between each prototype $$\mathbf{p}_j$$ and all patches of 
$$z$$ that have the same shape. This creates an activation map that can be upsampled to show which region of the image
is similar to the prototype. Global pooling is used to reduce the activation map to a single value similarity score.  

![](/collections/images/thislookslikethat/g_equation.jpg)

The $$m$$ similarity scores are passed to the fully-connected $$h$$ which computes output logits which are then normalized
by a softmax to return a standard classification output. 

## Training 

Training consists of 3 parts. 

**1) Stochastic gradient descent (SGD) of layers before last layer.** The goal of this stage is to learn the latent space. 
During this stage both the the convolutional weights $$w_{conv}$$ and prototypes $$\mathbf{P}$$ are learned. The 
fully-connected matrix is fixed and this stage and the weights are manually set to 1 for connections between prototypes and outputs logits
that are of the same class. The other weights are set to -0.5. 
   
Given a dataset $$D=[X, Y] = \{ x_i, y_i \}_{i=1}^n$$ the following loss function is minimized 

![](/collections/images/thislookslikethat/loss.jpg)

The CrsEnt term is a standard cross-entropy loss. The Clst term is a cluster loss that forces each training image to have 
a latent  patch close to at least one prototype of its class. The Sep term is a separation cost that encourages every 
latent path of an image to be projected far away from the prototypes not associated to its class. 

**2) Projection of prototypes** This stage takes the learned prototypes and associates them to a real latent patch that
corresponds to a real training example. Each prototype $$\mathbf{p}_j$$ associated to class $$k$$ is pushed to the nearest 
latent training patch from the same class. 
   
![](/collections/images/thislookslikethat/push.jpg)

The authors provide a theorem that explains theoretically how this has little effect on the prediction of the model. 
   
**3) Convex optimization of last layer** The final stage of training optimizes the weights of the last fully-connected
layer $$w_h$$. This optimization takes place as the convolutional weights and prototypes are frozen. The optimization 
goal is given by 

![](/collections/images/thislookslikethat/loss2.jpg)

The second term encourages sparsity in the weight matrix such that prototypes and logits that do not share a class have 
weights close to zero. 

## Prototype Visualization 

In order to be interpretable, the prototypes must correspond to a real patch in a training image. To find this 
corresponding patch for each prototype obtained from step 2 of the training process, the activation map for prototype's
corresponing image is upsampled to get an activation map. The patch is defined as the smallest rectangle that encloses 
all the pixel values that are as large as the 95th percentile of values in the activation map. 


##  Reasoning process of the network

![](/collections/images/thislookslikethat/fig3.jpg)



# Data


# Results 

![](/collections/images/thislookslikethat/results.jpg)


# Conclusion