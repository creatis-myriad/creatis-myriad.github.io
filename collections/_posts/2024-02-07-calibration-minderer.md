---
layout: review
title: "Revisiting the Calibration of Modern Neural Networks"
tags: classification, calibration
author: "Gaspard Dussert"
cite:
    authors: "Matthias Minderer, Josip Djolonga, Rob Romijnders, Frances Hubis, Xiaohua Zhai, Neil Houlsby, Dustin Tran, Mario Lucic"
    title: "Revisiting the Calibration of Modern Neural Networks"
    venue: "Arxiv 2021"
pdf: "https://arxiv.org/abs/2106.07998"
---

# Notes

* Quick recap of the key concepts in calibration using Guo et al. paper
* More interesting results with Minderer et al. paper

# Introduction

For the introduction of the key concepts let's go back to the article of Guo et al., **On Calibration of Modern Neural Networks**, PMLR 2017.

## Calibration 

**A model is well-calibrated if the predicted confidence scores represents a good approximation of the actual probability of correctness.**

For example : if we have 100 images predicted *cancer* with a score of 80%, we expect 20 predictions to be wrong. 

How to measure calibration ? Two simple way, first visually and then quantitatively. 

## Reliability histogram

**Group the predicted scores into bins, and plot the observed accuracy vs the expected expected accuracy**. In more details : 

For a set of $$N$$ images, we define the true class of the $$i$$-th image $$y_i$$ and $$p_i = (p_{i1}, ..., p_{iK})$$ the confidences scores of the $$K$$ classes. The predicted class $$\hat{y}_i$$ is the top-1 classification prediction, that is the class with the greatest confidence score, denoted $$s_i$$:
\begin{equation}
    \hat{y}_i = \underset{k \in [1, K]}{\arg\max} \: p_i  \hspace{0.5cm} \text{and} \hspace{0.5cm}  s_i = \underset{k \in [1, K]}{\max} \: p_i
\end{equation}

For $$M$$ evenly spaced bins, we can define $$b_m$$ the set of indices $$i$$ such as $$s_i \in ]\frac{m-1}{M}, \frac{m}{M}]$$ and compute the average bin accuracy and the average bin confidence score :

\begin{equation}
    \operatorname{acc}\left(b_m\right) = \frac{1}{\left|b_m\right|} \sum_{i \in b_m} \mathbb{1}\left(\hat{y}_i=y_i\right) 
\end{equation}

\begin{equation}
    \operatorname{conf}\left(b_m\right) = \frac{1}{\left|b_m\right|} \sum_{i \in b_m} s_i
\end{equation}

![](/collections/images/calibration/reliability_histogram.jpg)

## Expected Calibration Error

**ECE is defined as the bin-wise calibration error weighted by the size of the bin** :

\begin{equation}
    \mathrm{ECE}=\sum_{m=1}^M \frac{\left|b_m\right|}{N}\left|\operatorname{acc}\left(b_m\right)-\operatorname{conf}\left(b_m\right)\right|
\end{equation}

## Temperature Scaling 

Temperature scaling is a post-processing method to improve the calibration of the model after the training. **The scores predicted by the model are rescaled by a temperature parameter $$T > 0$$** using a generalization of the softmax function :

\begin{equation}\label{eq:1}
    p_{ij} = \frac{\exp^{z_{ij}/T}}{\sum_{k=1}^K \exp^{z_{ik}/T}}
\end{equation}

## Conclusion of Guo et al. on the calibration of "modern" networks of 2017 :

![](/collections/images/calibration/guo_results.jpg)

* Deep Learning models are poorly-calibrated : often very **overconfident**
* Temperature Scaling is very effective to improve the calibration of these models




