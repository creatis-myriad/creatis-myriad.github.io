---
layout: post
title:  "Introduction to Plug-and-Play methods for image restoration"
author: 'Thibaut Modrzyk'
date:   2024-02-02
categories: optimization, deep learning
---

# Note

This tutorial introduces a relatively recent family of algorithms for solving inverse problems, with a particular focus on image restoration.
While prior knowledge of inverse problems and optimization is not strictly required, it will help in following the discussion more easily.
In the first section I'll briefly go over the key concepts of inverse problems in order for the tutorial to be self-sufficient. For a deeper dive on the topic I strongly recommend Amir Beck's textbook **First-Order Methods in Optimization** [^1].

Note also that this tutorial is heavily influenced by the first part of Samuel Hurault's Thesis **Convergent plug-and-play methods for image inverse problems with explicit and nonconvex deep regularization**.

Finally all of what is discussed here relies on denoising, so having some intuition as to how denoiser and data are related through Tweedie's formula might help. See for instance [my video](https://www.youtube.com/watch?v=0V96wE7lY4w) on the topic.

&nbsp;

# Summary

- [Note](#note)
- [Summary](#summary)
- [Inverse Problems and Optimization](#inverse-problems-and-optimization)
  - [Inverse Problems](#inverse-problems)
  - [Ill-posedness](#ill-posedness)
  - [Variational formulation](#variational-formulation)
  - [Proximal operator](#proximal-operator)
- [Plug-and-Play algorithms](#plug-and-play-algorithms)
  - [The denoising prior](#the-denoising-prior)
  - [Neural Networks as proximal operators](#neural-networks-as-proximal-operators)
  - [What about convergence ?](#what-about-convergence-)
- [Explicit Denoising Regularization](#explicit-denoising-regularization)
  - [Gradient-Step denoiser](#gradient-step-denoiser)
  - [Gradient-Step Proximal Gradient Descent](#gradient-step-proximal-gradient-descent)
  - [Proximal denoiser](#proximal-denoiser)
- [Conclusion and Perspectives](#conclusion-and-perspectives)
- [References](#references)

&nbsp;

# Inverse Problems and Optimization

## Inverse Problems

Inverse problems encompass a very broad family of situations where one aims at recovering a latent "clean" variable from a set of degraded observations.
For instance one can think of the classical example of Computed Tomography (CT), where scanners are rotating around an object and acquiring projection data at several angles, forming a sinogram.
Retrieving a clean image of the object from this sinogram is a typical exemple of inverse problem.

<div style="text-align:center">
<img src="/collections/images/plug_and_play/radon.jpg" width=500></div>
<p style="text-align: center;font-style:italic;margin-top: 15px;">Figure 1. The tomographic recontruction inverse problem. The goal is to retrieve the image of a patient's body from the sinogram on the right. Here the forward model is the Radon transform. </p>

A more formal definition of the problem would be to retrieve an image $$ x \in \mathbb{R}^n $$ from an observation $$ y \in \mathbb{R}^m $$:

$$ y = \mathcal{N}(Ax)$$

with $$ A \in \mathbb{R}^{m \times n} $$ representing the degradation, and $$ \mathcal{N} $$ being a noise model.
The matrix $$ A $$ is often called *forward model*, or *system matrix*, depending on the application.
Although one might think that assuming the degradation to be linear would be restrictive, most inverse problems that we face in imaging fall under these assumptions. Here's a non-exhaustive list of well-known linear inverse problems in imaging:
- Deconvolution (aka deblurring)
- Super-resolution (aka upscaling)
- Impainting
- Denoising
- Hyperspectral unmixing (aka source separation)
- Pansharpening

<div style="text-align:center">
<img src="/collections/images/plug_and_play/inverse_problems.jpg" width=700></div>
<p style="text-align: center;font-style:italic;margin-top: 15px;">Figure 2. The tomographic recontruction inverse problem. The goal is to retrieve the image of a patient's body from the sinogram on the right. Here the forward model is the Radon transform. </p>

&nbsp;

## Ill-posedness

Even with a linear degradation, solving such problems proves fairly challenging.
Indeed most image reconstruction problems are what we call *ill-posed* inverse problems, meaning that many reconstruction candidate $$ x $$ might fit the observation $$ y $$.
[Hadamard](https://en.wikipedia.org/wiki/Jacques_Hadamard) proposed a more formal definition in stating that the problem is ill-posed if $$ x $$ does not change continually with respect to $$ y $$, meaning that small changes in the observation might result in widely different reconstructions $$ x $$. 
This is for instance the case when $$ A $$ is not invertible, or when its condition number $$\kappa$$ is small.
Let us recall that the condition number is defined as:

$$ \kappa(A) = \left| \frac{\lambda_{max}}{\lambda_{min}} \right| $$

where $$ \lambda_{max}, \lambda_{min}$$ are respectively the biggest and smallest eigenvalues of $$A$$.
A small condition number typically causes the noise in $$y$$ to explode when computing the pseudo-inverse $$A^{\dagger}$$.

&nbsp;

## Variational formulation

A typical approach to adress ill-posed inverse problems is to formulate them as optimization problems integrating what is often called a *regularization* term:

$$ \hat{x} = \arg \min_x f(x) + \lambda g(x) $$

where $$ \hat{x} $$ is our reconstruction, $$ f $$ is often called the *data-fidelity* term and $$ g $$ the *regularization*. As the name indicates, the data-fidelity function is there to ensure that our reconstruction $$ \hat{x} $$ matches the observation $$ y $$. In most cases, this function is the $$ L_2 $$ squared distance:

$$ f (x) = \| y - Ax \|^2_2 $$

We call this form a *variational* problem because we study the variations of the objective function to find a minimum. A benefit of this formulation is that it makes the connection with a whole other theoretical toolbox called **Optimization**. Optimization is a well-established field, thus many tools and algorithms have already been developed to solve our problem.
For simplicity, let's assume that the objective function $$f + \lambda g$$ is convex, and that the regularization $$g$$ is smooth. In this case, a well-known algorithm to find a minimum of our objective function is the *gradient descent*.

$$ x^{n+1} = x^{n} - \tau \nabla (f(x^{n}) + \lambda g(x^{n}))$$

where $$\tau$$ is the gradient step size, aka *learning rate*.
We also know that this hyper-parameter should be chosen carefully for the algorithm to converge. Namely, if $$f + \lambda g$$ is $$L$$-smooth, meaning that we have $$L > 0$$ so that:

$$\forall x, y \in \mathbb{R}^n ~ \| \nabla f(x) - \nabla f(y)\| \leq L \|x - y\|$$

then $$\tau$$ should be taken smaller than $$2 / L$$.

&nbsp;

## Proximal operator

This is all well and good but actually the regularizations we want to use are usually non-smooth. Worse than that, they are not even differentiable everywhere. Indeed many regularizations rely on the absolute value, for instance the Total Variation (TV) regularization, and the absolute value is non-differentiable at 0 (it has two different derivatives whether you are left or right of zero).

This is why mathematicians have developed a new tool in order to be able to take a sort of gradient step on functions that are not differentiable everywhere: the *proximal operator*. Funnily enough, this operator is itself defined as an optimization problem:

$$ \text{prox}_{\tau g} (x) = \arg \min_u g(u) + \| x - u\|_2^2$$

This operator has *many* different interpretations, but let's just say that we compute the minimum of a smoothed version of our original pointy objective function $$g$$. Interestingly, if $$g$$ is convex we know that $$\text{prox}_{\tau g} (x)$$ is single-valued, meaning it really is a function.
When we have a closed-form expression to compute the proximal operator of a function in any points, we say this function is *proximable*. These expressions are typically found in textbooks or on the website [Prox Repository](https://proximity-operator.net/).

<div style="text-align:center">
<img src="/collections/images/plug_and_play/abs_prox.jpg" width=600></div>
<p style="text-align: center;font-style:italic;margin-top: 15px;">Figure 3. The famous example of the absolute value and its proximal operator. For each point, we can draw the orange tangent function. The proximal operator is computed by taking the minimum of this function at each point. </p>

This operator enables us to construct many other optimization algorithms with convergence guarantees even when $$g$$ is non-differentiable. The most simple of them is the *Forward-Backward* splitting, or *Proximal Gradient Descent*. This is a so-called *splitting* method, because we alternate between the minimization of $$f$$ using a gradient step, and $$g$$ using a proximal step:

$$x^{n+1} = \text{prox}_{\tau g} \circ (\text{Id} - \tau \nabla f) (x^{n}) $$

This algorithm has good convergence properties in most of the cases that we encounter in image processing, meaning that it converges to a minimum of our objective function, or at least a **local minimum**.

<div style="text-align:center">
<img src="/collections/images/plug_and_play/tv.jpg" width=600></div>
<p style="text-align: center;font-style:italic;margin-top: 15px;">Figure 4. Typical look of a reconstructions made using the TV regularization. Here the algorithm is not the proximal gradient descent but another algorithm using proximal operators called ADMM. </p>

&nbsp;

# Plug-and-Play algorithms

Until recently, image restoration was primarily approached by formulating an optimization problem and designing an appropriate regularization. However, with the rise of deep learning over a decade ago, data-driven methods quickly gained popularity, outperforming hand-crafted regularization techniques.

One major drawback of these deep learning methods is the lack of theoretical guarantees that their outputs truly correspond to valid solutions of the inverse problem. This is particularly concerning because, while they may perform well on a given dataset, they often fail on out-of-domain data. Many deep learning approaches to inverse problems follow an "end-to-end" paradigm, where a neural network is trained to map an observation $$y$$ to the target $$x$$, given a forward model $$A$$. The problem, however, is that if $$A$$ changes, the trained network becomes ineffective.

Plug-and-Play (PnP) methods offer an elegant solution by reinterpreting neural networks as regularizers, decoupling them from the forward model $$A$$, which is typically embedded in the data-fidelity term. More importantly, PnP methods provide convergence guarantees—something exceedingly rare in deep learning-based approaches. Interestingly, these methods were first introduced in 2013, roughly the same period as deep learning’s take off, yet they originally did not involve neural networks at all.

&nbsp;

## The denoising prior

Plug-and-Play methods start with the 2013 paper of Venkatakrishnan *et al.* [^2].
This paper starts with the simple observation that if one has a good Gaussian denoiser at hand, it is a solution of the Gaussian denoising optimization problem. 

The Bayesian formulation of inverse problem is:

$$ D_\sigma (y) = \arg \max_x p(x \mid y) = \arg \min_x - \log p(y \mid x) - \log p(x) $$

where $$-\log p(y \mid x)$$ is the negative log-likelihood, which is basically the data-fidelity term, and $$ \log p(x) $$ is the prior over the clean data distribution $$x$$.

In the special case of Gaussian denoising, we have:

$$D_\sigma(y) = \arg \min_x \frac{1}{2} \| x - y \|^2 - \sigma^2 \log p(x)$$

where $$\sigma$$ is the standard deviation of the Gaussian noise, sometimes called *noise level*.
Does it reminds you of anything ? Well yes, the denoiser can actually be written as a proximal operator over an unknown prior $$- \sigma^2 \log p(x)$$ !

$$ D_\sigma(y) = \text{prox}_{- \sigma^2 \log p(x)} (y)$$

This observation leads Venkatakrishnan to interpret state-of-the-art Gaussian denoisers as explicit proximal operators. At the time, the best denoiser was BM3D [^3],  which is not a neural network at all but rather relies on non-local self-similarity and transform-domain filtering.

&nbsp;

## Neural Networks as proximal operators

As of 2025, deep convolutional neural networks remain the gold standard for Gaussian denoising, offering state-of-the-art performance with fast inference.
Early models like DnCNN [^4] required retraining for each noise level $$\sigma$$, meaning if one wanted to use different noise levels it required a family of trained models tuned for each noise level.
The most widely used denoising model is a convolutional UNet nicknamed DRUNet [^5] developed by Kai Zhang in 2022.
This model takes the noise level $$\sigma$$ as input, meaning we now only need a single model for the vast majority of noise levels we encounter.
Due to its convenience of use and very good performances, it is currently the baseline in the vast majority of Plug-and-Play papers.

As mentionened previously, since these denoisers act as Gaussian denoisers, we can also interpret them as approximating a proximal operator over an implicit prior $$- \sigma ^2 \log p(x)$$ related to our data distribution
(*Shameless plug*: if you want more intuitions regarding this distribution, check out [my video](https://www.youtube.com/watch?v=0V96wE7lY4w) on the topic).
This denoiser can then be "plugged" in many optimization algorithms originally designed with a proximal step over an explicit regularization in mind.
Here are a few examples of such algorithms:

**Plug-and-Play Proximal Gradient Descent (PnP-PGD)**

$$x^{n+1} = D_\sigma \circ \left( \text{Id} - \tau \nabla f \right) (x^{n})$$

**Plug-and-Play Douglas-Rachford Splitting (PnP-DRS)**

$$x^{n+1} = \left( \beta (2 D_{\sigma} - \text{Id}) \circ \alpha \text{prox}_{\tau f} + (1 - \beta) \text{Id} \right) (x^n)$$

**Plug-and-Play Alternating Direction Method of Multipliers (PnP-ADMM)**

$$\left\{
    \begin{array}{ll}
        y^{n+1} = \text{prox}_{\tau \lambda f} (x^n) \\
        z^{n+1} = D_\sigma(y^{n+1} + x^{n}) \\
        x^{n+1} = x^n + (y^{n+1} - z^{n+1})
    \end{array}
    \right.$$

The benefit of the Plug-and-Play framework is that we can use all of what we know about optimization and just "Plug" state-of-the-art denoisers, including neural networks, to achieve excellent perfomances. 
We can also accelerate these algorithms easily thanks to known methods such as Nesterov [^5] or Polyak's [^6] iterations for first-order methods.

## What about convergence ?

<div style="text-align:center">
<img src="/collections/images/plug_and_play/convergent.jpg" width=600></div>
<p style="text-align: center;font-style:italic;margin-top: 15px;">Figure 5. Typical look of a metric / loss function for image restoration algorithms. In most cases, you reach a maximum at a given iterations and then at best you see diminishing returns, at worse you start degrading your image. Convergent algorithms however reach a sort of stationnary point. </p>

We identify Gaussian denoisers to proximal operators thanks to the Bayesian MAP interpretation, but this is a rather hand-wavy explanation.
While this actually works experimentally, meaning that these PnP algorithms show very good performances, one might want to recover convergence guarantees of the original algorithms.

**Why don't we have convergence directly since we say denoisers are proximal operators ?**

- *Ideal* denoisers can be seen as proximal operators, but no denoiser is optimal and we never have a loss of zero when training / testing a model
- Proximal operators have other properties that are required for convergence proofs, and these properties are not easily met by neural networks
- The regularization is *implicit*, meaning we have no idea what its true expression is, nor do we have a mean to compute its value, so we can't experimentally verify the convergence
- We train neural network denoisers using *Denoising Score Matching* [^8], which actually implies that our denoisers approximate the MMSE of the denoising problems, not the MAP

# Explicit Denoising Regularization

Several works have attempted to bridge the theoretical gap preventing convergence guarantees for PnP algorithms. Unfortunately, this often requires imposing strong and unrealistic assumptions on the network.

One such condition is *non-expansiveness*, meaning that for any inputs $$x$$ and $$y$$, the denoiser does not increase the distance between the inputs:

$$\| D_{\sigma} (y) - D_{\sigma} (x) \| < \| y - x\|$$

This is equivalent to impose that the gradient of the denoiser is Lipschitz continuous with a Lipschitz constant smaller than 1:

$$\| \nabla D_\sigma (x) \| < 1$$

Enforcing this property typically requires architectural constraints on the network. Unfortunately, attempts in this direction have led to a noticeable degradation in denoising performance.
An alternative approach is to modify the training objective to encourage non-expansiveness. However, this does not strictly guarantee contractiveness across all inputs, leaving theoretical guarantees uncertain.

## Gradient-Step denoiser

So what we would really like is to have a neural network that does not only *approximates* as proximal operator, but rather that *is* a proximal operator.
A somewhat related intermediate step to achieve this is to create a neural network that is formulated as an explicit gradient-step.
This is exactly what Samuel Hurault *et al.* and Regev Cohen *et al.* simultanuously proposed in 2021 at NeurIPS and ICLR.
The idea is to formulate a **potential** $$g_{\sigma}$$ from a neural network $$N_{\theta}$$ to create a denoiser $$D_\sigma$$ which is an explicit gradient-step over the potential:

$$D_\sigma(x) = x - \tau \nabla g_\sigma(x)$$

The main difference between the works of Cohen and Hurault lies in their scope. In Cohen *et al.* [^9], multiple examples of potentials $$g_\sigma$$​ are proposed, along with a general convergence theorem that applies to all these cases.
However, some assumptions are again imposed on these potentials, that are not easily met.
In contrast, Hurault's NeurIPS paper [^10] focuses on a single instance of such a potential, which only requires realistic assumptions, and offers a particularly elegant Bayesian interpretation.
In the following, we will focus on Hurault's work, which he has further developed and refined throughout his thesis. Also, he's French (cocorico).

So the idea is to focus on the potential:

$$g_\sigma = \| x - N_{\theta} (x) \|^2_2$$

One then formulates the denoiser as an explicit gradient step over this potential, and trains it like a regular Gaussian denoiser, with denoising score matching:

$$\mathbb{E}_{x \sim p(x), \epsilon \sim \mathcal{N(0, 1)}} \| D_\sigma(x + \sigma^2 \epsilon) - x \|^2_2$$

which with this choice of $$g_\sigma$$ is equal to:

$$\mathbb{E}_{x \sim p(x), \epsilon \sim \mathcal{N(0, 1)}} \| \nabla g_\sigma (x + \sigma^2 \epsilon) - \sigma^2 \epsilon \|^2_2$$

You might notice that this effectively imposes a special form to our neural network based denoiser $$D_{\sigma}$$.
However Hurault *et al.* verify experimentally that this does not impact the denoising performances of the model $$N_{\theta}$$, which they choose as the state-of-the-art DRUNet architecture.
If you want to learn more about these results, check the [blog post](https://creatis-myriad.github.io/2025/01/18/GradientStepDenoiser.html) about of this specific paper.
A nice interpretation links $$g_\sigma$$ to our implicit prior over the training data $$p(x)$$.
We know thanks to Tweedie's formula that we have:

$$\begin{align}
  D_\sigma(x) &\approx x + \sigma^2 \nabla \log p_{\sigma}(x) \\
  \nabla g_\sigma (x) &\approx - \sigma^2 \nabla \log p_{\sigma}(x) \\
  g_\sigma (x) &\approx - \sigma^2 \log p_{\sigma}(x) + C
\end{align}
$$

So the potential $$\| x - N_\sigma(x)\|^2_2$$ is somewhat related to the smoothed version of our prior $$p(x)$$.

## Gradient-Step Proximal Gradient Descent

A nice property of $$g_\sigma$$ as defined by Hurault *et al.* is that it is $$L$$-smooth, meaning it has $$L$$-Lipschitz gradient.
This enables them to write a slightly modified version of the Proximal-Gradient-Descent and drectly use the convergence proofs available.
The usual gradient-descent scheme reads as:

$$x^{n+1} = \text{prox}_{\tau g} \circ (\text{Id} - \tau \nabla f) (x^{n}) $$

However here we have our denoiser is a gradient step on $$g$$ and not on $$f$$. So what Hurault did is simply swap the two in the algorithm.
This is unusual but makes everything much easier to prove.
Meaning the iterations read as:

$$ x^{n+1} = \text{Prox}_{\tau f} \circ \left( \text{Id} - \tau \lambda \nabla g_\sigma \right) (x^{n})$$

And if we replace with the expression of the denoiser:

$$ x^{n+1} = \text{Prox}_{\tau f} \circ D_\sigma (x^{n})$$

So to sum up usually we have two differentiable convex functions $$f$$ and $$g$$, with the regularization $$g$$ generally non-smooth (for instance TV).
And here we sort of swap the assumptions: we have $$g_\sigma$$ $$L$$-smooth, possibly non-convex, and $$f$$ convex possibly non-smooth.

**Is it truly Plug-and-Play**? Not exactly—we don’t strictly replace a proximal operator with a denoiser. However, in a broader sense, yes: the denoiser acts as a form of regularization, and we can integrate any network while still preserving convergence guarantees.

I won't get into more details as to the actual proof, but it follows a very classic scheme in optimization that we can use mainly because of the "hack" of swapping $$f$$ and $$g$$.

**Why is this work important ?** Well although it does not exactly match the Plug-and-Play framework, it is the first algorithm to make realistic assumptions on the neural netwwork while both preserving convergence guarantees and offering state-of-the-art performances on several inverse problems.

## Proximal denoiser

The gradient-step denoiser was a stepping stone to a denoiser that exactly matches the usual Plug-and-Play framework. In their ICML 2022 paper [^11], Hurault *et al.* propose a new interpretation to their previous work enabling to see it as a proximal step over an implicit function $$\phi_{\sigma}$$:

$$D_{\sigma} (x) = x - \nabla g_{\sigma} (x) = \text{prox}_{\phi_{\sigma}} (x)$$

I'll skip over these details because it is purely theoretical and does not change anything regarding the actual denoiser $$D_\sigma$$: it is still the same gradient-step denoiser. However, formulating it as a proximal operator better fits the original Plug-and-Play vision, and opens up more naturally to extending the convergence results to well-known optimization algorithms.

# Conclusion and Perspectives

This is time to conclude this tutorial on Plug-and-Play methods. What have we (hopefully) learned here:

- Image restoration tasks can often be formulated as inverse problems, and the resolution of such problems heavily relies on optimization
- Recent developments on Gaussian denoisers parametrized with neural networks can be connected to optimization
- While this link was informal it provided very good experimental results
- However standard convergence guarantees of these algorithms were still out of reach without sacrificing perfomances
- Thanks to Gradient-Step denoisers, Plug-and-Play algorithms are now rigorously proven, while providing state-of-the-art performances on inverse problems

**Why do we even care ?** Theoretical understanding is precisely what much of modern machine learning research lacks. Establishing connections between new methods and well-understood domains is essential for advancing and solidifying our field. More importantly, it allows us to leverage decades of prior work to build better, more principled approaches.

**What's even left to do ?** At first glance, Plug-and-Play might seem *solved*. In a sense, it is—the connection between these methods and classical optimization is now well-established. However, the landscape of deep neural networks has evolved. While convolutionnal Gaussian denoisers were at their peak in 2017, the focus has now shifted towards more powerful generative models like **Diffusion Models**[^12] and **Flow Matching**[^13] as effective regularizers.

We still lack a solid theoretical framework linking these modern generative models to optimization algorithms. The answer likely lies at the intersection of differential equations, optimization, and optimal transport.

&nbsp;

# References

[^1]: First-Order Methods in Optimization, Amir Beck, 2017
[^2]: Plug-and-Play Priors for Model Based Reconstruction, Singanallur V. Venkatakrishnan *et al.*, 2013
[^3]: Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering, Kostadin Dabov *et al.*, 2007
[^4]: Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising, Kai Zhang *et al.*, 2017
[^5]: Plug-and-Play Image Restoration With Deep Denoiser Prior, Kai Zhang *et al.*, 2022
[^6]: A method for solving the convex programming problem with convergence rate O (1/k2), Yuri Nesterov, 1983
[^7]: Some methods of speeding up the convergence of iteration methods, Boris Polyak, 1964
[^8]: A Connection Between Score Matching and Denoising Autoencoders, Pascal Vincent, 2011
[^9]: It Has Potential: Gradient-Driven Denoisers for Convergent Solutions to Inverse Problems, Regev Cohen *et al*, 2021
[^10]: Gradient Step Denoiser for convergent Plug-and-Play, Samuel Hurault *et al.*, 2021
[^11]: Proximal Denoiser for Convergent Plug-and-Play Optimization with Nonconvex Regularization, Samuel Hurault *et al.*, 2022
[^12]: Denoising Diffusion Models for Plug-and-Play Image Restoration, Zhu *et al.*, 2023
[^13]: PnP-Flow: Plug-and-Play Image Restoration with Flow Matching, martin *et al.*, 2024