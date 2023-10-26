---
layout: review
title: "ProtoASNet: Dynamic Prototypes for Inherently Interpretable and Uncertainty-Aware Aortic Stenosis Classification in Echocardiography"
tags: interpretability classification uncertainty ultrasound
author: "Thierry Judge"
cite:
    authors: "Hooman Vaseli, Ang Nan Gu, S. Neda Ahmadi Amiri, Michael Y. Tsang, Andrea Fung, Nima Kondori, Armin Saadat, Purang Abolmaesumi, Teresa S. M. Tsang"
    title:   "ProtoASNet: Dynamic Prototypes for Inherently Interpretable and Uncertainty-Aware Aortic Stenosis Classification in Echocardiography"
    venue:   "MICCAI 2023"
pdf: "https://arxiv.org/pdf/2307.14433.pdf"
---

# Notes
* Link to the code [here](https://github.com/hooman007/ProtoASNet)

# Highlights
* The authors propose a interpretable classification model to classify aortic stenosis (AS). 
* The model is a prototype-based model which analyzed the similarity between the input and a set of prototypes. 

# Introduction

Most deep-learning approaches to classification of aortic stenosis are black-box models that are not interpretable.



# Method 

## Prototype-Based Models
See "This Looks Like That: Deep Learning for Interpretable Image Recognition" [^1] for a more detail explanation. 
The notation in this paper is slightly different and is redefined here.

A prototype based model is generally the composition of three models. The structure is defined by $$h(g(f(x)))$$ where 

* $$x \in R^{H_o \times W_o \ times 3}$$ is the input 
* $$f(x)$$ is a convolutional backbone. 
* $$g(x)$$ is the prototype layer. 
* $$h(x)$$ is a fully-connected layer.

The prototype layer contains $$P$$ prototypes ($$K$$ for each class or the $$C$$ classes$$): $$p^c_k$$. 

As the prototypes are learned vectors, they must be "pushed" to a real image to be interpretable. This is done by finding 
the closest training example feature in the latent space. 

![](/collections/images/ProtoASNet/push.jpg)

## ProtoASNet

![](/collections/images/ProtoASNet/method.jpg)

In ProtoASNet, the convolutional backbone is a pre-trained R(2+1)D-18 with two region of interest (ROI) modules. Given
an input video $$x \in R^{H_o \times W_o \times T_o \times 3}$$, features $$F(x) \in R^{H \times W \times T \times D}$$
are computed. The second branch computes $$P$$ region of interest (one for each prototype) 
$$M_{p^c_k}(x) \in R^{H \times W \times T}$$. 

The spatial temporal features are pooled before being compared to the prototypes. 

![](/collections/images/ProtoASNet/eq_roi.jpg)

where $$\circ$$ is the Hadamard product. 

The prototype pooling is done by computing the cosine similarity between a prototype $$p^c_k$$ and a feature vector 
$$f_{p^c_k}$$

![](/collections/images/ProtoASNet/eq_cosine.jpg)


The similarity scores are used by the fully-connected layer to predict the class probabilities. The weights of the layer
$$w_h$$ are initialized to be 1 between corresponding class logits and prototypes and 0 otherwise. 


## Prototypes for Aleatoric Uncertainty
To learn the aleatoric uncertainty, another set of prototypes (not related to any class) is defined: $$p^c_k$$. 
The similarity between the features $$f_{p^c_k}$$ and the uncertainty prototypes $$p^c_k$$ are used to predict 
$$\alpha \in [ 0, 1 ]$$. The loss to train this part of the model is taken from [^2]


![](/collections/images/ProtoASNet/abs_loss.jpg)

## Full loss 

The full loss function is given by 

![](/collections/images/ProtoASNet/loss.jpg)

The clustering and separation losses are not applied to the uncertainty prototypes are there are no labels for the 
uncertainty. 

The other terms are 

*  Orthogonality loss (Eq. (8)) to encourage diverse prototypes 
* Transformation loss $$L_{trns}$$ to "regularize the consistency of the predicted occurrence regions under random
affine transformations" (taken from [^3])
* $$L_norm$$ regularizes $$w_h$$ for sparsity with respect to noncoresponding prototypes and class logits. [^1] 


Contrary to [^1] which pushes the prototypes only after the latent representation is learned. The authors "push" the 
prototypes to the closest training feature every 5 epochs. They also do not mention training in stages which implies 
the training is conducted end-to-end. 


# Experiments 

## Data 

The authors used two datasets. The first is a private AS dataset for which the AS severity was determined with a standard 
Dopple echo. The dataset contains 5055 PLAX and 4062 PSAX view cines for a total of 2572 studies. 

The second dataset is the TMED-2. It contains 599 fully labeled echo studies (17270 images). The studies contain 

## Results 

![](/collections/images/ProtoASNet/table1.jpg)


## Examples

![](/collections/images/ProtoASNet/ex1.jpg)

![](/collections/images/ProtoASNet/ex2.jpg)

![](/collections/images/ProtoASNet/ex3.jpg)

## Ablation study 

The authors report an ablation study showing the effects of different components of their model. 


![](/collections/images/ProtoASNet/ablation.jpg)



# References


[^1]: Chen, C., Li, O., Tao, D., Barnett, A., Rudin, C., Su, J.K.: This looks like that:
deep learning for interpretable image recognition. Neurips 2019.

[^2]: DeVries, T., Taylor, G.W.: Learning confidence for out-of-distribution detection in
neural networks. arXiv.  

[^3]: Kim, E., Kim, S., Seo, M., Yoon, S.: Xprotonet: Diagnosis in chest radiography
with global and local explanations. CVPR 2021