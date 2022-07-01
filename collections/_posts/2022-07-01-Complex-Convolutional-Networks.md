---
layout: review
title: "Complex Convolutional Neural Networks for Image Reconstruction from IQ Signal"
tags: complex-CNNs IQ-signal image-reconstruction
author: "Julia Puig"
cite:
    authors: "Jingfeng Lu, Fabien Millioz, Damien Garcia, Sebastien Salles, Dong Ye, Denis Friboulet"
    title:   "Complex Convolutional Neural Networks for Ultrafast Ultrasound Imaging Reconstruction From In-Phase/Quadrature Signal"
    venue:   "IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control"
pdf: "https://arxiv.org/pdf/2009.11536.pdf"
---


# Context

* Ultrafast imaging requires the **compounding** of several acquisitions to improve image contrast and resolution. There is a trade-off between image quality and image rate.
* Previous work: the same authors developed a CNN for the compounding of 3 RF acquisitions and reached the same quality than when using 31 RF acquisitions [[1]](https://arxiv.org/abs/1911.03416).
* Objective: To reduce data size and power requirement, use IQ signals instead of RF signals.
* Problem: **IQ signals are complex**.


# Highlights

* Implementation of a **complex convolution** inspired by the multiplication of complex numbers.
* Definition of a **complex activation function** based on the complex module.
* Experimental demonstration that the complex CNN outperforms the straightforward method of processing the real and imaginary parts separately.


# Methods

The complex CNN has 3 main building blocks:

1.  The **complex convolution** is based on [[2]](https://arxiv.org/abs/1705.09792) and uses complex weights $W = W_r + jW_i$. It is defined as:

$$
\left[\begin{array}{c}
\operatorname{Re}(W * X) \\
\operatorname{Im}(W * X)
\end{array}\right]=\left[\begin{array}{cc}
W_{r} & -W_{i} \\
W_{i} & W_{r}
\end{array}\right] *\left[\begin{array}{c}
X_{r} \\
X_{i}
\end{array}\right]
$$

![](/collections/images/complex_cnns/complex_convolution.jpg)

2. The maxout activation function (MU) used in the previous work is replaced by the **amplitude maxout** (AMU). Given a complex convlutional layer output $Z$ and $Z_a = |Z|$, the amplitude maxout of $Z$ is defined as:

$$
\left[\begin{array}{c}
\operatorname{Re}(\text{AMU}(Z)) \\
\operatorname{Im}(\text{AMU}(Z))
\end{array}\right]=\left[\begin{array}{cc}
\operatorname{Re}(Z)\left[\operatorname{argmax}\left(Z_{a}\right)\right] \\
\operatorname{Im}(Z)\left[\operatorname{argmax}\left(Z_{a}\right)\right]
\end{array}\right]
$$


![](/collections/images/complex_cnns/amplitude_maxout.jpg)




# Results


![](/collections/images/complex_cnns/result_image_quality.jpg)



![](/collections//images/complex_cnns/result_contrast_ratio.jpg)




# Conclusions


