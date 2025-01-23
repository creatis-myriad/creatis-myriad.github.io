---
layout: review
title: "Provably convergent image restoration with Gradient-Step DRUnet"
tags: Inverse Problems, Plug-and-Play, Non-convex Optimization, Convergence
author: "Thibaut Modrzyk"
cite:
    authors: "Samuel Hurault, Arthur Leclaire, Nicolas Papadakis"
    title:   "Gradient-Step denoiser for convergent Plug-and-Play"
    venue:   "ICLR, 2022"
pdf: "https://arxiv.org/pdf/2110.03220"
---


# Highlights

* Provide an iterative reconstruction method which decreases an explicit cost function at each step, using a neural network regularization
* Provably converges to a local minimum of this cost function
* Can use (almost) any neural network architecture
* Very loose assumptions on the data-fidelity term: $$f$$ convex

# Note

The implementation is part of the [DeepInverse library](https://deepinv.github.io/deepinv/index.html), more specifically the dedicated [GS-DRUnet class](https://deepinv.github.io/deepinv/api/stubs/deepinv.models.GSDRUNet.html).
The original implementation for ICLR 2022 is available here: [https://github.com/samuro95/GSPnP](https://github.com/samuro95/GSPnP).

# Reminders

---


If you're not familiar with optimization and inverse problems, you might want to read this part carefully.
Else, you can probably skip it.

## Inverse problems

Inverse problems are a very broad class of mathematical problems intervening in all sorts of natural sciences. The name "Inverse Problem" comes from the fact that these problems consider degraded observations in order to retrieve hidden parameters.

The general problem can be formulated as "retrieve $$x$$ from $$y$$ where we know that:"

$$y = f(x)$$

with $$f$$ being called the *forward model*, which can be either known, partially known or totally unknown.

Among inverse problems, a very important subclass is that of **linear inverse problems**, because it encompasses many use cases in imaging. These problems specifically assume a *linear* forward model, meaning that the problem now reads as:

$$y = Ax$$

where $$A$$ is a matrix.
An important aspect of inverse problems is **noise**.
Indeed, the observations $$y$$ are very often corrupted by some random noise, which for simplicity we often model as being Gaussian:

$$y = Ax + \epsilon$$

where $$\epsilon \sim \mathcal{N}(0, \sigma^2)$$.

One might think that linear inverse problems are very easy to solve: why don't we just inverse the forward model $$A$$ ?
Well, most of the time this matrix is not invertible, which means that there are several solutions $$x$$ that matches the noisy observation $$y$$.
When this is the case, we say that the inverse problem is *ill-posed*.

In order to reduce the space of admissible solutions, we introduce a-priori knowledge about $$x$$. Basically this means that we have a rough idea of what the solutions should look like. In the Bayesian setting, this is called the *prior*. It is also often called *regularization*.

A very common example of ill-posed linear inverse problem is *Deconvolution*, which is often called Deblurring when dealing with images. This problem aims at finding the solutions $$x$$ of problem:

$$y = h \ast y + \epsilon$$

where $$h$$ is a blur kernel.

Finally, we might want to estimate the forward model $$A$$ as well as the original data $$x$$. In this case the inverse problem is called *Blind*, because the forward model is unknown. This is usually more challenging, as there are many couples $$(H, x)$$ that might produce the observation $$y$$. In these cases, regularization becomes even more important.

## Variational methods

A very common way of solving inverse problems is to formulate them as optimization problems, more specifically minimizing a cost function, often called *energy*:

$$x^* = \arg \min_x f(x, y) + g(x)$$

where $$f$$ is the data-fidelity term, and $$g$$ is the regularization.
You can think of $$f$$ as the part that measures how well the data $$x$$ explains the observation $$y$$, while $$g$$ encodes the prior knowledge.

Let's take a very simple example. We know that our data is corrupted by Gaussian noise, and that our solution is supposed to be sparse. In this case, the correct data-fildelity term is $$f(x, y) = \|x - y\|_2^2$$, and a regularization that promotes sparsity would be $$g(x) = \|x\|_1 $$.

Now we can use all the literature on convex optimization to solve our problem.
However, in many cases our regularization $$g$$ is not differentiable everywhere, meaning that we can't use a regular Gradient Descent algorithm.
This is for instance the case for the $$L_1$$ regularization we just chose, but it is also the case for the *Total Variation* regularization, which is still very common in imaging inverse problems.
Indeed this regularization promotes *piecewise constant* solutions, which describes well-enough natural images with sharp edges.

## Proximal operator

How do we get around the non-differentiability of our objective function ?
Well some very smart mathematicians have developed a special tool in order to minimize non-smooth function.
This tool is called the *proximal map* or *proximal operator* and you can think of it as a sort of smooth projection.

![The moreau enveloppe of the L1 norm](/collections/images/GradientStep/moreau.jpg)

Since applied mathematicians love optimization, this tool is itself formulated as an optimization problem:

$$\text{prox}_{\tau f} (x) = \arg \min_u f(u) + \frac{1}{2 \tau} \| x - u\|^2$$

As you can see when computing the prox of a function, the goal is to minimize the function while staying close to a given point $$x$$.
Long story short, there are closed form solutions for the proximal operators of a lot of usual functions.
And more precisely, most usual non-smooth regularization function that one might use in inverse problems admit closed-form proximal operators.
This is for instance the case of our previous example with the $$L_1$$ norm, but also the famous *Total Variation* regularization and others.

A typical algorithm to solve the previous minimization problem would then be the *Forward-Backward* splitting, or *Proximal Gradient Descent*.

$$x^{n+1} = \text{prox}_{\tau g} \circ \left( \text{Id} - \tau \nabla f \right) (x^{n}) $$

As you can see it's just a regular gradient descent on the smooth data-fidelity term $$f$$ and a proximal step on the non-smooth regularization $$g$$.
There are a lot of other algorithms using the proximal operator, but this is the most basic one.

---

# Plug-and-Play

Let us consider an ill-posed linear inverse problem, for instance super-resolution, deblurring or inpainting. TV regularization was good looking 15 years ago, but now the reconstructions look cartoonish when compared to what end-to-end deep learning approaches can achieve.

![Example with TV regularization](/collections/images/GradientStep/tv.jpg)

However reconstruction methods that use **only** neural networks also have a lot of drawbacks:
- no data-fidelity constraints, meaning that we have no theoretical guarantees on their performances
- very sensitive to domain-shift, so applying them to data way out of their training distribution is risky
- typically require to be re-trained as soon as the degradation operator $$A$$ changes

What we really want is the best of both worlds:
- good reconstructions using neural networks
- some constraints with respect to the observations

Plug-and-Play methods are exactly this compromise. They use the traditionnal variational formulation of inverse problems and replace the hand-crafted regularization $$g$$ by a Gaussian denoiser $$D_\sigma$$.
Let's take the forward-backward splitting: its Plug-and-Play version now simply becomes:

$$x^{n+1} = D_\sigma \circ \left( \text{Id} - \tau \nabla f \right) (x^{n})$$

These Plug-and-Play methods give very good reconstructions, with a natural robustness to domain-shift and less hallucinations than end-to-end methods.

![Example of a PnP reconstruction](/collections/images/GradientStep/pnp.jpg)

However there are still several issues with these methods. With state-of-the-art denoisers, we have no theoretical guarantees that these schemes converge to a fixed-point, let alone the minimum of our original optimization problem. 

In order to have some theoretical guarantees, we need **a lot** of assumptions on $$D_\sigma$$, the most restrictive one is that $$D_\sigma$$ needs to be *contractive*, meaning that:

$$D_\sigma (x) \leq x$$

This is **very hard** to impose during training, and most of the solutions to verify this condition rely on changes in the architectures of the networks. 
This however **dramatically reduces performances**.

# Gradient-Step denoiser

To address these issues and recover theoretical convergence guarantees the authors formulate a special denoizer:

$$D_\sigma (x) = x - \tau \nabla_x \| x - N_\sigma(x)\|^2$$

Which can be seen as an **explicit gradient step** over the regularization $$g_\sigma (x) = \| x - N_\sigma(x)\|^2$$:

$$D_\sigma (x) = x - \tau \nabla_x g_\sigma (x)$$

This denoiser can then be trained as any other, with an $$L2$$ norm over a range of noise levels (now often called denoising score matching):

$$\mathcal{L}(D_\sigma) = \mathbb{E}_{x \sim p(x), \epsilon_\sigma \sim \mathcal N(0, \sigma^2)} \left[ \| D_\sigma(x + \epsilon_\sigma) - x\right \|^2]$$

**REMARK** : with a bit of algebra and Tweedie's formula we can show that: $$\nabla g_\sigma (x) = - \sigma^2 \nabla \log p_\sigma (x)$$. This offers the nice interpretation of this regularization being tightly related to the **score of the noisy prior distribution** $$ p_\sigma(x)$$.

The authors then show in a series of experiments that this special form of denoiser **does not degrade denoising performances**.

| σ (./255)      | 5      | 15     | 25     | 50     | Time (ms) |
|---------------|--------|--------|--------|--------|-----------|
| FFDNet   | 39.95  | 33.53  | 30.84  | 27.54  | 1.9       |
| DnCNN     | 39.80  | 33.55  | 30.87  | 27.52  | 2.3       |
| **DRUNet**    | **40.31** | **33.97** | **31.32** | **28.08** | 69.8      |
|---------------|--------|--------|--------|--------|-----------|
| DRUNet *light*| 40.19  | 33.89  | 31.25  | 28.00  | 6.3       |
| **GS-DRUNet**     | 40.26  | 33.90  | 31.26  | 28.01  | 10.4      |

For context, DRUNet is still considered one of the best denoising model.
The authors achieve roughly similar performances while using a lighter version of the DRUnet (2 residual blocks instead of 4). They even perform slightly better than simply using the exact same architecture, without the gradient-step formulation.

# Plug-and-Play with the Gradient-Step denoiser

The interesting thing about this formulation is that now we have a regularization $$g_\sigma$$ with nice properties. More precisely it is $$L$$-smooth, which is necessary to design convergent first-order algorithms.

I'll skip the details, but the important thing is that now we can devise a **convergent Plug-and-Play** algorithm, with **any neural network** $$N_\sigma$$. The algorithm reads as follows:

$$ x^{n+1} = \text{Prox}_{\tau f} \circ \left( \text{Id} - \tau \lambda \nabla g_\sigma \right) (x^{n})$$

This algorithm is then proven to explicitly decrease the objective function:

$$F(x) = f(x) + g_\sigma (x)$$

which was our initial objective.

**REMARK**: you might notice that it is unusual to take a proximal step on $$f$$ and a gradient step on the regularization $$g$$. This algorithm needs to be formulated as such in order to write the convergence proof, but this is not ideal. In some cases $$f$$ does not admit a closed-form solution for its proximal operator. Subsequent work from the same authors partially fixes this issue.

# Experiments

The authors test their denoiser along with their new algorithm on two tasks:
- deblurring, with 10 different blur kernels (motion blur, gaussian, uniform)
- super-resolution (with gaussians)

They perform on-par with the state-of-the-art other PnP algorithms, while offering convergence guarantees.

![Deblurring experiments](/collections/images/GradientStep/deblurring.jpg)

The explicit objective function $$F$$ can be seen decreasing to a local minimum and reaching a fixed-point in subfigure (g).

![Super-resolution experiments](/collections/images/GradientStep/superresolution.jpg)

The extensive experiments of the authors show that they perform on-par with SOTA methods while using a lighter denoiser.

![Quantitative results on the deblurring task](/collections/images/GradientStep/results.jpg)

# Conclusion

Why does this matter ?

- state-of-the-art performances
- a single network for a wide variety of image restoration problems
- now we can confidently use deep learning models in image restoration
- no need to select the "best" iteration, or use early-stopping: the algorithm is convergent !
- a new framework to easily design other convergent Plug-and-Play algorithms for inverse problems
- opens up a lot of new research directions