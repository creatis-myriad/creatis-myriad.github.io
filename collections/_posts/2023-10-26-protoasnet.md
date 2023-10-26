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

Authors define the background for prototype-based models as generally having a structure defined by $$h(g(f(x)))$$ where 

* $x \in R^{H_o \times W_o \ times 3}$ in the input 
* $f(x)$  
* $g(x)$
* $h(x)$


## ProtoASNet