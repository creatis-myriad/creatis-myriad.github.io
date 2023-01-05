---
layout: review
title: "Gaussian distributions are boring : pimp them with Normalizing Flows"
tags: deep-learning normalizing-flow
author: "Robin Trombetta"
cite:
    authors: "George Papamakarios, Eric Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, Balaji Lakshminarayanan"
    title:   "Normalizing Flows for Probabilistic Modeling and Inference"
    venue:   "Journal of Machine Learning Research 2022"
pdf: "https://arxiv.org/pdf/1912.02762.pdf"
---

# Note
The review is not focused on a single article; the idea is to introduce a type of models, Normalizing Flows, by presenting some articles that have proposed the main advances on this field. The references of all these articles are reported at the end of the page.

[This video](https://www.youtube.com/watch?v=i7LjDvsLWCg) may help understanding better this topic.

# Highlights

* Normalizing flow is a method to construct complex distributions by transforming a probability density by applying a sequence of simple invertible transformation functions.
* Flow-based generative models are fully tractable, allowing exact likelihood computation and both easy sample generation and density estimation.
* Normalizing flows have multiple applications including data generation, density estimation for outlier detection, noise modelling, etc.

&nbsp;

# Introduction

## Motivation

Main generative models include Generative Adversarial Network and Variational Auto-Encoder, that have both demonstrated impressive performance results on many tasks. However, these models have several issues limiting their application, one of them being that they do not allow for exact evaluation of the density of new points. Normalizing flow are a family of generative model with tractable distributions where both sampling and density evaluation can be efficient and exact.

## Change of variable

Let $$z \in Z$$  be a random variable with a known density function $$p_{Z}$$ and denote $$f : Z \to X$$ a diffeomorphism. The change of variable operated by $$f$$ can be used to transform $$z \sim p_{Z}(z)$$ into a random variable $$x = f(z)$$ that has the following probability distribution :

$$ p_{X}(x) = p_{Z}(z) |det \frac{\partial f^{-1}(x)}{\partial x}| = p_{Z}(z) |det \frac{\partial f(z)}{\partial z}|^{-1} $$

where $$\frac{\partial f}{\partial z}$$ is the Jacobian matrix of the application $$f$$.

&nbsp;

# Normalizing flow

Normalizing flow is a type of generative models created for better and more powerful distribution approximation. It allows to transform a simple distribution (typically a multivariate normal distribution) into a more complex one though a serie of invertible mappings.
As the choice of transformation functions is restricted, the final transformation is often constructed with a serie of simple functions $$f_1, ..., f_K$$ :

$$ f_\theta = f_K \circ f_{K-1} \circ ... \circ f_1 $$

The following figure illustrates principle of a normalizing flow model.

![](/collections/images/noflow/flow.jpg)
<p style="text-align: center;font-style:italic;">Figure 1. Illustration of a normalizing flow model.</p>

&nbsp;

During the successive modifications, a sample $$z$$ ***flows*** though a sequence of transformations and always kept as a valid distrubution function because it is ***normalized*** at each step.

Given such type of transformation, it is possible to compute directly the likelihood of some data observed $$x = \{x^{(i)}\}_{i=1,...,N}$$ :

$$ \log p(x|\theta) = \sum_{i=1}^{N} \log p_{Z}(f^{-1}(x^{(i)})|\theta) + \log |\text{det} \frac{\partial f^{-1}(x^{(i)})}{\partial x^{(i)}} | $$

$$ \log p(x|\theta) = \sum_{i=1}^{N} \log p_{Z}(f^{-1}(x^{(i)})|\theta) + \sum_{k=1}^{K} \log |\text{det} \frac{\partial f_{k}^{-1}(x_{k}^{(i)})}{\partial x_{k}^{(i)}} | $$

where $$\theta$$ denotes the parameters of the transformation $$f$$.

Compared to other generative models such as GANs or VAEs, the optimization problem is much more straightforward as it does not require an adversarial network or the introduction of a lower bound of the likelihood.

**_NOTE:_**  Since all the functions involved in the transormation of densities are bijective and differentiable, the normalizing flow process can be equivalently seen the other way around. Another common manner to introduce normalizing flow is to present the startings distribution as a complex one ($$x$$ in the figure above, *e.g.* samples from real world data) that is step by step normalized to a simple one. In that case, the optimization problem is generally introduced by saying that we aim to minimize the KL-divergence between the transformed distribution $$f_X(x)$$ and the simpler density function (typically $$\mathcal{N}(0,I)$$).

&nbsp;

# Types of flow

Theoretically, any diffeomorphism could be used to build a normalizing flow model, but in practice it should satisfy two properties to be applicable:
* Be invertible with an easy-to-compute inverse function (depending on the application)
* Computing the determinant of its Jacobian needs to be efficient. Typically, we want the Jacobian be a triangular matrix.

Therefore, designing flows is the core problem adressed by research on this topic. The objective is to find functions such as described above that can still be complex enough to build models that yield good expressive power.

&nbsp;

## Illustration with planar and radial flows

In the change of variable formula, the absolute value of the determinant of the jacobian of $$f$$ is a dilation/retractation factor of the space. In low dimension and with simple transformation function, it is possible to observe how an initial density function can be distorted during the flow process.

&nbsp;

We consider famiily of transformations, called planar flows[^1], described by:

$$ f(\textbf{z}) = \textbf{z} + \textbf{u}h(\textbf{w}^{T}\textbf{z} + b) $$

where $$\lambda = \{\textbf{w} \in \mathbb{R}^{D}, \textbf{u} \in \mathbb{R}^{D}, b \in \mathbb{R} \}$$ are free parameters and $$h(\cdot)$$ is a differentiable element-wise and non-linear function. This particular flow transforms a density by applying a series of contractions and expansions in the direction perpendicular to the hyperplane $$\textbf{w}^T \textbf{z}+b = 0$$. For this mapping, we can compute the determinant of the Jacobian in $$O(D)$$ time :

$$ | \text{det} \frac{\partial f}{\partial z} | = |1+\textbf{u}^Th'(\textbf{w}^{T}\textbf{z} + b)\textbf{w} |$$

&nbsp;

Similarly, the family described by :

$$ f(\textbf{z}) = \textbf{z} + \beta h(\alpha,r)(\textbf{z} - \textbf{z}_0) $$

$$ | \text{det} \frac{\partial f}{\partial z} | = [1+\beta h(\alpha,r)]^{d-1} [1+\beta h(\alpha,r)+\beta h'(\alpha,r)r ]$$


with $$r = |\textbf{z} - \textbf{z}_0|$$, 
$$h(\alpha,r)=1/(\alpha + r)$$, and parameters $$\lambda= \{ \textbf{z}_0 \in \mathbb{R}^D, \alpha \in \mathbb{R}^{+}, \beta \in \mathbb{R} \}$$, applies contractions and expansions around the reference point $$\textbf{z}_0$$ and is called radial flow.

 
The effect of these types of transformations can be seen in Fig. 2 for two examples of 2D distributions.

![](/collections/images/noflow/planar_radial_flow.jpg)
<p style="text-align: center;font-style:italic;color=#1A56A7;">Figure 2. Effect of planar and radial flow on two distributions<SUP>1</SUP>.</p>

&nbsp;

## Coupling layers

A core family of transformation has been introduced by Dinh et al.[^2] in 2015 and is called coupling layers.

An input vector $$z$$ is split into $$z_{1:d}$$ and $$z_{d+1:D}$$. In the forward pass, the ouput vector $$x$$ is obtained as follows (Fig. 3) :

$$
\left\{
    \begin{array}{ll}
        x_{1:d} = z_{1:d} \\
        x_{d+1:D} = g(z_{d+1:D},m(z_{1:d}))
    \end{array}
\right.
$$

where $$m(\cdot)$$ can by any function and $$g(\cdot)$$ is an invertible function.
If $$g$$ is easy enough to invert, the backwards pass is simple as well :

$$
\left\{
    \begin{array}{ll}
        z_{1:d} = x_{1:d} \\
        z_{d+1:D} = g^{-1}(x_{d+1:D},m(x_{1:d}))
    \end{array}
\right.
$$


![](/collections/images/noflow/coupling_layers.jpg)
<p style="text-align: center;font-style:italic;">Figure 3. Illustration of the principle of coupling layers.</p>

&nbsp;

The Jacobian matrix of this transformation is :

$$
\frac{\partial x}{\partial z} = \begin{bmatrix}
\textbf{I}_d & 0\\
\frac{\partial x_{d+1:D}}{\partial z_{1:d}} & \frac{\partial x_{d+1:D}}{\partial z_{d+1:D}}
\end{bmatrix}
$$

The key of this flow is that there is no need to invert the mapping $$m(\cdot)$$ to compute the determinant of the Jacobian, thus it can be anything, such as a dense neural network, a CNN, a Transformer, etc. Only the function $$g(\cdot)$$ needs to remain relatively simple, for instance affine with non-linear scaling factors.

Sine only a fraction of the input vector goes though a complex transformation, coupling layers are stacked and alternated with permutations [^3] to improve the expressivity of the flow model.

# Examples of results

In 2016, Dinh et al. [^3] introduced Real NVP, a flow model using real-valued non-volume preserving transformations. It was one of the first deep learning model using normalizing flow to perform density estimation and image generation. The results with this model, shown in Fig. 4, are far from the standards we have now but is similar to the best generative models at that time. 

![](/collections/images/noflow/realnvp_results.jpg)
<p style="text-align: center;font-style:italic;">Figure 4. Examples of faces generated by Real NVP trained on CelebFaces Attributes dataset.</p>

&nbsp;

Later, in 2018, Glow[^4] used coupling layers and introduced 1x1 invertible convolutions to achieve much better looking results are set a new state-of-the-art flow-based generative model.

![](/collections/images/noflow/glow_results.jpg)
<p style="text-align: center;font-style:italic;">Figure 5. Examples of faces generated by GLOW trained on CelebFaces Attributes dataset.</p>

&nbsp;

For several years now, flow models are used to performed diverse tasks such as unsupervised anomaly detection. With FastFlow[^5] for example, Wu. et al. achieved state-of-the-art performances on the task of unsupervised anomaly detection on the industrial dataset MVTec AD.

![](/collections/images/noflow/fastflow_results.jpg)
<p style="text-align: center;font-style:italic;">Figure 6. Examples of unsupervised anomaly detection on an industrial dataset using normalizing flow.</p>


# Conclusion

Flow models are a type a generative models designed to transform distributions in fully tractable way though a series a invertible mappings. They can be used for any application related to data generation and density estimation. 

# References

[^1]: D. Jimenez Rezende, S. Mohamed. [Variational Inference with Normalizing Flows](https://openreview.net/pdf?id=BywyFQlAW). June 2016.
[^2]: L. Dinh, D. Krueger, Y. Bengio. [NICE: Non-linear Independent Components Estimation](https://arxiv.org/pdf/1410.8516.pdf). In ICLR. April 2015.
[^3]: L. Dinh, J. Sohl-Dickstein and S. Bengio. [Density estimation using Real NVP](https://arxiv.org/pdf/1605.08803.pdf). In ICLR. February 2017.
[^4]: D. P. Kingma, P. Dhariwal. [Glow: Generative Flow with Invertible 1x1 Convolutions](https://arxiv.org/pdf/1807.03039.pdf). July 2018.
[^5]: J. Wu et al. [FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows](https://arxiv.org/pdf/2111.07677.pdf).  November 2018.
