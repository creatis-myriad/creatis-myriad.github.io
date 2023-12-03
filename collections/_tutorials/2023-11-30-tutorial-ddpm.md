---
layout: post
title:  "The denoising diffusion probabilistic models (DDPM) paradigm demystified"
author: 'Celia Goujat, Olivier Bernard'
date:   2023-11-16
categories: diffusion, model
---

# Notes

* Here are links to two video and an excellent post that we used to create this tutorial: [video1](https://www.youtube.com/watch?v=HoKDTa5jHvg&ab_channel=Outlier), [video2](https://www.youtube.com/watch?v=TBCRlnwJtZU&ab_channel=Outlier), 
[post](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/).

- [**Introduction**](#introduction)
  - [Overview of diffusion models](#overview-of-diffusion-models)
  - [Diffusion models vs generative models](#diffusion-models-vs-generative models)
- [**Fondamental concepts**](#fondamental-concepts)
  - [Sum of normally distributed variables](#sum-of-normally-distributed-variables)
  - [Bayes theorem](#bayes-theorem)
  - [Reparameterization trick](#reparameterization-trick) 
  - [Cross entropy](#cross-entropy)    

- [**Forward diffusion process**](#forward-diffusion-process)
  - [Principle](#principle) 
  - [How to define the variance scheduler ?](#how-to-define-the-variance-scheduler) 
- [**Backward process**](#backward-process)
  - [General idea](#general-idea) 
  - [Loss function](#loss-function) 
- [**To go further**](#to-go-further)
  - [The first DDPM algorithm](#first-ddpm-algorithm) 
  - [How to improve the log-likelihood ?](#how-to-improve-the-log-likelihood) 
  - [Improve sampling speed](#improve-sampling-speed)

&nbsp;

## **Introduction**

### Overview of diffusion models (DM)

- DM are a class of generative models such as GAN, [normalizing flow](http://127.0.0.1:4000/tutorials/2023-01-05-tutorial_normalizing_flow.html) or [variational auto-encoders](http://127.0.0.1:4000/tutorials/2022-09-12-tutorial-vae.html). 
- DM defines a Markov chain of diffusion steps to slowly add random noise to data.
- The model then learns to reverse the diffusion process to construct data samples from noise.
- The figure below gives an overview of the Markov chain involved in the DM formalism, where the forward (reverse) diffusion process is the key element in generating a sample by slowly adding (removing) noise.

![](/collections/images/ddpm/ddpm_overview.jpg)

&nbsp;

### Diffusion models vs generative models

- DM belongs to the generative models family.
- DM has demonstrated effectiveness in generating high-quality samples.
- Unlike GAN, VAEs and flow-based models, the latent space involved in the DM formalism has high-dimensionality corresponding to the dimensionality of the original data.
- The figure below gives and overview of the different types of generative models:

![](/collections/images/ddpm/generative-model-overview.jpg)

&nbsp;

## **Fondamental concepts**

### Sum of normally distributed variables

- If $$x$$ and $$y$$ be independent random variables that are normally distributed $$ x \sim \mathcal{N}(\mu _X , \sigma ^2_X \, \mathbf{I})$$ and $$ y \sim \mathcal{N}(\mu _Y , \sigma ^2_Y \, \mathbf{I})$$ then $$ x + y \sim \mathcal{N}(\mu _X + \mu _Y, (\sigma ^2_X + \sigma ^2_Y) \, \mathbf{I})$$

> The sum of two independent normally distributed random variables is also normally distributed, with its mean being the sum of the two means, and its variance being the sum of the two variances.

- If $$ x \sim \mathcal{N}(0, \mathbf{I})$$ then $$ \sigma \, x \sim \mathcal{N}(0, \sigma^2 \, \mathbf{I})$$. 

- If x and y be independent standard normal random variables $$x \sim \mathcal{N}(0, \mathbf{I})$$ and $$y \sim \mathcal{N}(0, \mathbf{I})$$, the sum $$\sigma_X x + \sigma_Y y $$ is normally distributed such as $$\sigma_X x + \sigma_Y y \sim \mathcal{N}(0, (\sigma^2_X + \sigma^2_Y) \, \mathbf{I})$$.

&nbsp;

### Bayes Theorem

$$ q(x_{t} \mid x_{t-1}) = \frac{q(x_{t-1} \mid x_{t}) \, q(x_{t})}{q(x_{t-1})} $$

$$ q(x_{t-1} \mid x_t) = \frac{q(x_t \mid x_{t-1}) \, q(x_{t-1})}{q(x_t)} $$

&nbsp;

### Conditional probability theorem

$$ q(x_{t} \mid x_{t-1}) = \frac{q(x_{t-1} \mid x_{t}, x_0) \, q(x_{t} \mid x_0)}{q(x_{t-1}) \mid x_0} $$

&nbsp;

### Marginal theorem

$$q(x_{0},x_{1},\cdots,x_{T}) = q(x_{0:T})$$ 

$$q(x_{0}) = \int q(x_{0},x_{1},\cdots,x_{T}) \,dx_{1}\,\cdots\,dx_{T}$$ 

$$q(x_{0}) = \int q(x_{0:T}) \,dx_{1:T}$$ 

&nbsp;

### Markov chain

$$q(x_{1:T} \mid x_0) = \prod_{t=1}^{T} q(x_t \mid x_{t-1})$$ 

$$q(x_{0:T}) = q(x_{T}) \prod_{t=1}^{T} q(x_{t-1} \mid x_{t})$$

&nbsp;

### Reparameterization trick

- Transform a stochastic node sampled from a parameterized distribution into a deterministic ones. 
- Allows backpropagation through such a stochastic node by turning it into deterministic node. 
- Let's assume that $$x_t$$ is a point sampled from a parameterized gaussian distribution $$q(x_t)$$ with mean $$\mu$$ and variance $$\sigma^2$$. 
- The following reparametrization tricks uses a standard normal distribution $$\mathcal{N}(0,\mathbf{I})$$ that is independent to the model, with $$\epsilon \sim \mathcal{N}(0,\mathbf{I})$$:

$$ x_t = \mu + \sigma \cdot \epsilon$$

- The prediction of $$\mu$$ and $$\sigma$$ is no longer tied to the stochastic sampling operation, which allows a simple backpropagation process as illustrated in the figure below

![](/collections/images/ddpm/reparameterizationTrick.jpg)

&nbsp;

### Cross entropy

- Entropy $$H(p)$$ corresponds to the average information of a process defined by its corresponding distribution

$$ H_{p} = -\int{p(x)\cdot \log\left(p(x)\right)}\,dx$$

> The cross entropy measures the average amount of information you need to add to go from a given distribution $$p$$ to a reference distribution $$q$$

$$H_{pq} = -\int{p(x)\cdot \log\left(q(x)\right)}\,dx = \mathbb{E}_{x \sim p} [\log(q(x))]$$


- It can also been seen as a tool to quantify the extent to which a distribution differs from a reference distribution. It is thus strongly link to the Kullback–Leibler divergence measures as follow:

$$\begin{align}
D_{KL}(p \parallel q) &= H_{pq} - H_{p} \\
& = -\int{p(x)\cdot \log(q(x))}\,dx + \int{p(x)\cdot \log(p(x))}\,dx \\
& = -\int{p(x)\cdot \log\left(\frac{q(x)}{p(x)}\right)}\,dx \\
& = \int{p(x)\cdot \log\left(\frac{p(x)}{q(x)}\right)}\,dx \\
& = \mathbb{E}_{x\sim p} \left[\log\left(\frac{p(x)}{q(x)}\right)\right]
\end{align}$$

- $$H(p)$$, $$H(p,q)$$ and $$D_{KL}(p \parallel q)$$ are always positives.

&nbsp;

> When comparing a distribution $$q$$ against a fixed reference distribution $$p$$, cross-entropy and KL divergence are identical up to an additive constant (since $$p$$ is fixed).


&nbsp;

## **Forward diffusion process**

### Principle

&nbsp;

![](/collections/images/ddpm/ddpm-forward-process.jpg)

&nbsp;

- Let define $$x_0$$ a point sampled from a real data distribution $$x_0 \sim q(X_0)$$. 

- The forward diffusion process is a procedure where a small amount of Gaussian noise is added to the point sample $$x_0$$, producing a sequence of noisy samples $$x_1, \cdots , x_T$$.

> The forward process of a probabilistic diffusion model is a Markov chain, i.e. the prediction at step $$t$$ only depends on the state at step $$t-1$$, that gradually adds gaussian noise to the data $$x_0$$. 

- The full forward process can be modeled as: 

$$q(x_{1:T} \mid x_0) = \prod_{t=1}^{T}{q(x_t \mid x_{t-1})}$$

- Based on the definition of the forward process, the conditional distribution $$q(x_t \mid x_{t-1})$$ can be efficiently represented as:

<div style="text-align:center">
<span style="color:#00478F">
$$q(x_t \mid x_{t-1}) = \mathcal{N}\left((\sqrt{1 - \beta _t}) \, x_{t-1},\beta _t \, \textbf{I}\right)$$
</span>
</div>

&nbsp;

- The step sizes are controlled by a variance schedule $$\{ \beta_t \in (0,1) \}_{t=1}^T$$

- $$\beta_t$$ becomes increasingly larger as the sample becomes noisier, __i.e.__

$$0 < \beta_1 < \beta_2 < \cdots < \beta_T < 1$$

- Based on the above equation, the two extrem cases can be easily derived: 

$$\begin{align}
& \text{if} \quad \beta_t=0 , \quad \text{then} \quad q(x_t \mid x_{t-1})=x_{t-1} \\
& \text{if} \quad \beta_t=1 , \quad \text{then} \quad q(x_t \mid x_{t-1})=\mathcal{N}(0,\textbf{I})
\end{align}$$

&nbsp;

- A nice property of the forward process is that we can sample $$x_t$$ at any arbitrary time step $$t$$ in a closed form using [reparametrization trick](#reparameterization-trick) and the property of the [sum of normally distributed variables](#sum-of-normally-distributed-variables), as shown below:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\left((\sqrt{1 - \beta _t}) \, x_{t-1},\beta _t \, \textbf{I}\right)$$

$$x_t = (\sqrt{1 - \beta_t}) \, x_{t-1} + \sqrt{\beta_t} \, \epsilon_{t-1}$$

$$\quad$$ Let's define $$\alpha_t = 1 - \beta_t$$

$$x_t = \sqrt{\alpha_t} \, x_{t-1} + \sqrt {1-\alpha_t} \, \epsilon_{t-1}$$

$$\quad$$ This expression is true for any $$t$$ so we can write

$$x_{t-1} = \sqrt{\alpha_{t-1}} \, x_{t-2} + \sqrt{1-\alpha_{t-1}} \, \epsilon_{t-2}$$

$$\quad$$ and

$$x_t = \sqrt{\alpha_t \alpha_{t-1}} \, x_{t-2} + \sqrt{\alpha_t- \alpha_t \alpha_{t-1}} \, \epsilon_{t-2} + \sqrt{1-\alpha_t} \, \epsilon_{t-1}$$

$$x_t \sim \mathcal{N}\left(\sqrt{\alpha_t \alpha_{t-1}} \, x_{t-2}, \alpha_t- \alpha_t \alpha_{t-1} \right) + \mathcal{N}\left(0, 1-\alpha_t \right)$$

$$x_t \sim \mathcal{N}\left(\sqrt{\alpha_t \alpha_{t-1}} \, x_{t-2}, \alpha_t- \alpha_t \alpha_{t-1} + 1-\alpha_t \right)$$

$$x_t \sim \mathcal{N}\left(\sqrt{\alpha_t \alpha_{t-1}} \, x_{t-2}, 1 - \alpha_t \alpha_{t-1} \right)$$

$$x_t = \sqrt{\alpha_t \alpha_{t-1}} \, x_{t-2} + \sqrt{1-\alpha_t \alpha_{t-1}} \, \bar{\epsilon}_{t-2}$$

$$\quad$$ One can repeat this process recursively until reaching and expression of $$x_t$$ from $$x_0$$:

$$x_t = \sqrt{\alpha_t \cdots \alpha_1} \, x_0 + \sqrt{1 - \alpha_t \cdots \alpha_1} \, \bar \epsilon _0 $$

$$\quad$$ By defining $$\bar{\alpha}_t = \prod_{k=1}^{t}{\alpha _k}$$, we finally get this final relation:

<div style="text-align:center">
<span style="color:#00478F">
$$x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \bar{\epsilon}_0$$
</span>
</div>


<div style="text-align:center">
<span style="color:#00478F">
$$q(x_t \mid x_{0}) = \mathcal{N}\left( \sqrt{\bar{\alpha}_t} \, x_0, (1 - \bar{\alpha}_t) \, \mathbf{I} \right)$$
</span>
</div>


&nbsp;

> When $$\, T \rightarrow \infty$$, $$\, \beta_t \rightarrow 1$$, $$\, \alpha_t \rightarrow 0$$, thus $$\, \bar{\alpha}_t \rightarrow 0$$ and $$\, x_T \sim \mathcal{N(0,\mathbf{I})}$$. 

> This means that $$x_T$$ is equivalent to a pure noise from a Gaussian distribution. We have therefore defined a forward process called a diffusion probabilistic process that introduce slowly noise into an image. 

&nbsp;

### How to define the variance scheduler?

- The orignial article set the forward variances $$\{\beta_t \in (0,1) \}_{t=1}^T$$ to be a sequence of linearly increasing constants. 

- Forward variances are chosen to be relatively small compared to data scaled to [-1,1]. This ensure that reverse and forward process maintain approximately the same functionnal form.

- More recently, a cosine scheduler has been proposed to improve results. 

$$\alpha _t = \frac{f(t)}{f(0)}, \quad f(t) = cos\left(\frac{\frac{t}{T} + s}{1+s} \cdot \frac{\pi}{2}\right)^2$$

- The variances $$\beta _t$$ can be deducted from this definition as $$\beta_t = 1 - \frac{\bar{\alpha_t}}{\bar{\alpha_{t-1}}}$$ 

- In pactice $$\beta_t $$ is clipped to be no larger than $$0,999$$ to prevent singularities for $$t \rightarrow T$$.

![](/collections/images/ddpm/schedules.jpg)

&nbsp;

## **Backward process**

### General idea

&nbsp;

![](/collections/images/ddpm/ddpm-reverse-process.jpg)

&nbsp;


- If we are able to reverse the diffusion process from $$q(x_{t-1} \mid x_t)$$, we will be able to generate a sample from a Gaussian noise input $$x_T \sim \mathcal{N}(0,\mathbf{I})$$ !

- Unfortunately, using the [Bayes theorem](#bayes-theorem) and keeping in mind that $$q(x_t)$$ is an unknown, one can easy see that $$q(x_{t-1} \mid x_t)$$ is intractable.

$$q(x_{t-1} \mid x_t) = \frac{q(x_{t} \mid x_{t-1}) \, q(x_{t-1})}{q(x_{t})}$$


> It is noteworthy that the reverse conditional probability is tractable when conditioned on $$x_0$$. Indeed, thanks to Bayes theorem $$\, q(x_{t-1} \mid x_t, x_0)$$ = $$\frac{q(x_t \mid x_{t-1}, x_0)q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)}$$, where all distributions are known from the forward process.

&nbsp;

- However, if $$\beta_t$$ is small enough, $$q(x_{t-1} \mid x_t)$$ will also be Gaussian.

- We will learn a model $$p_{\theta}$$ to approximate these conditional probabilities in order to run the reverse diffusion process:

<div style="text-align:center">
<span style="color:#00478F">
$$p_{\theta}(x_{t-1} \mid x_t) = \mathcal{N}(\mu _{\theta}(x_t,t), \Sigma_{\theta}(x_t,t))$$
</span>
</div>


<div style="text-align:center">
<span style="color:#00478F">
$$p_{\theta}(x_{0:T}) = p_{\theta}(x_T) \, \prod_{t=1}^{T} p_{\theta}(x_{t-1} \mid x_t)$$
</span>
</div>

> $$\mu _{\theta}(x_t,t)$$ and $$\, \Sigma_{\theta}(x_t,t))$$ depend not only on $$x_t$$ but also on $$t$$. Those parameters that need to be estimated are thus time-dependent !

<!-- ![](/collections/images/ddpm/reverseProcess.jpg) -->

&nbsp;

### Loss function

&nbsp;

![](/collections/images/ddpm/ddpm_overview_complete.jpg)

&nbsp;

- The loss function designed to learn the reverse process involves minimizing the cross-entropy between the target distribution $$q(X_0)$$ and the approximated distribution $$p_{\theta}(X_0)$$

$$H(q,p_{\theta}) = - \mathbb{E}_{x_0 \sim q}\left[\log( p_{\theta}(x_0))\right]$$

> Minimizing the cross entropy between $$q(X_0)$$ and $$\, p_{\theta}(X_0)$$ results in the two distributions being as close as possible

&nbsp;

- $$p_{\theta}(X_0)$$ depends on $$X_1, X_2, \dots, X_T$$. Thanks to the [mariginal theorem](#marginal-theorem), the above expression can be rewritten as:

$$\begin{align} 
H(q,p_{\theta})  & = - \mathbb{E}_{x_0 \sim q}\left[\log\left(\int p_{\theta}(x_{0:T}) \,d_{x_{1:T}}\right)\right] \\
&  = - \mathbb{E}_{x_0 \sim q}\left[\log\left(\int q(x_{1:T} \mid x_0)\frac{p_{\theta}(x_{0:T})}{q(x_{1:T} \mid x_0)}  \,d_{x_{1:T}} \right)\right] \\
&  = - \mathbb{E}_{x_0 \sim q}\left[\log \left(\mathbb{E}_{x_{1:T} \sim q(x_{1:T} \mid x_0)} \left[\frac{p_{\theta}(x_{0:T})}{q(x_{1:T} \mid x_0)}\right) \right] \right]
\end{align}$$

- Using the Jensen's inequality, the above equation can be rewritten as:


$$\begin{align} 
H(q,p_{\theta})  & \leq - \mathbb{E}_{x_0 \sim q}\,\mathbb{E}_{x_{1:T} \sim q(x_{1:T} \mid x_0)}\left[ \log\left(\frac{p_{\theta}(x_{0:T})}{q(x_{1:T} \mid x_0)}\right) \right] \\
& \leq - \mathbb{E}_{x_{0:T} \sim q(x_{0:T})}\left[ \log\left(\frac{p_{\theta}(x_{0:T})}{q(x_{1:T} \mid x_0)}\right) \right] \\
& \leq \mathbb{E}_{x_{0:T} \sim q(x_{0:T})}\left[ \log\left(\frac{q(x_{1:T} \mid x_0)}{p_{\theta}(x_{0:T})}\right) \right] \\
\end{align}$$

- We define the Variational Lower Bound (VLB) as 

<div style="text-align:center">
<span style="color:#00478F">
$$\mathcal{L}_{VLB} = \mathbb{E}_{x_{0:T} \sim q(x_{0:T})}\left[ \log\left(\frac{q(x_{1:T} \mid x_0)}{p_{\theta}(x_{0:T})}\right) \right]$$
</span>
</div>

&nbsp;

- Since $$H(q,p_{\theta})$$ is positive, minimizing $$\mathcal{L}_{VLB}$$ is equivalent to minimize $$H(q,p_{\theta})$$.

- To convert each term in the equation to be analytically computable, the minimization of $$\mathcal{L}_{VLB}$$ can be further rewritten to be a combination of several KL-divergence and entropy terms, as follows: 

$$ \begin{align} 
\mathcal{L}_{VLB}  & = \mathbb{E}_{x_{0:T} \sim q}\left[\log\left( \frac{q(x_{1:T} \mid x_0)}{p_{\theta}(x_{0:T})}\right)\right] \\ 
& = \mathbb{E}_{x_{0:T} \sim q}\left[\log\left( \frac{ \prod^T_{t=1} q(x_t \mid x_{t-1})}{p_{\theta}(x_{T}) \prod^T_{t=1} p_{\theta}(x_{t-1} \mid x_t)}\right)\right] \\

& = \mathbb{E}_{x_{0:T} \sim q}\left[-\log\left(p_{\theta}(x_{T})\right) + \sum^T_{t=1} \log\left(\frac{ q(x_t \vert x_{t-1})}{p_{\theta}(x_{t-1} \mid x_t)}\right)\right] \\

& = \mathbb{E}_{x_{0:T} \sim q}\left[-\log\left(p_{\theta}(x_{T})\right) + \sum^T_{t=2} \log\left( \frac{ q(x_t \vert x_{t-1})}{p_{\theta}(x_{t-1} \vert x_t)}\right) + \log\left( \frac{ q(x_1 \vert x_0)}{p_{\theta}(x_0 \vert x_1)}\right)\right] 
\end{align} $$

&nbsp;

- Using the [conditional probability theorem](#conditional-probability-theorem), the above expression can reformulated as:

$$ \begin{align} 
\mathcal{L}_{VLB}  & = \mathbb{E}_{x_{0:T} \sim q} \left[-\log \left(p_{\theta}(x_{T}) \right) + \sum^T_{t=2} \log \left( \frac{ q(x_{t-1} \vert x_t,x_0)q(x_t \mid x_0)}{p_{\theta}(x_{t-1} \vert x_t)q(x_{t-1} \mid x_0)} \right) + \log \left( \frac{ q(x_1 \mid x_0)}{p_{\theta}(x_0 \mid x_1)} \right) \right] \\

& = \mathbb{E}_{x_{0:T} \sim q} \left[-\log \left(p_{\theta}(x_{T})\right) + \sum^T_{t=2} \log \left( \frac{ q(x_{t-1} \mid x_t,x_0)}{p_{\theta}(x_{t-1} \mid x_t)} \right) + \sum^T_{t=2} \log \left( \frac{q(x_t \mid x_0)}{q(x_{t-1} \mid x_0)} \right) + \log \left( \frac{ q(x_1 \mid x_0)}{p_{\theta}(x_0 \mid x_1)} \right)\right] \\

& = \mathbb{E}_{x_{0:T} \sim q} \left[-\log \left(p_{\theta}(x_{T}) \right) + \sum^T_{t=2} \log \left( \frac{ q(x_{t-1} \mid x_t,x_0)}{p_{\theta}(x_{t-1} \mid x_t)} \right) + \log \left( \frac{q(x_T \mid x_0)}{q(x_1 \mid x_0)} \right) + \log \left( \frac{ q(x_1 \mid x_0)}{p_{\theta}(x_0 \mid x_1)} \right) \right] \\

& = \mathbb{E}_{x_{0:T} \sim q} \left[\log \left({\frac{q(x_T \mid x_0)}{p_\theta(x_{T})}} \right) + \sum^T_{t=2} \log \left( \frac{ q(x_{t-1} \mid x_t,x_0)}{p_{\theta}(x_{t-1} \mid x_t)} \right) - \log \left( p_{\theta}(x_0 \mid x_1) \right) \right] \\

& =  \underbrace{D_{KL} \left(q(x_T \mid x_0) \parallel p_\theta(x_{T})\right)}_{\mathcal{L}_T} + \sum^{T}_{t=2}  \underbrace{D_{KL}\left(q(x_{t-1} \mid x_t,x_0) \parallel p_{\theta}(x_{t-1} \mid x_t)\right)}_{\mathcal{L}_{t-1}} - \underbrace{\log \left( p_{\theta}(x_0 \mid x_1)\right)}_{\mathcal{L}_0} \\

& =  \underbrace{D_{KL} \left(q(x_T \mid x_0) \parallel p_\theta(x_{T})\right)}_{\mathcal{L}_T} + \sum^{T-1}_{t=1}  \underbrace{D_{KL}\left(q(x_t \mid x_{t+1},x_0) \parallel p_{\theta}(x_t \mid x_{t-1})\right)}_{\mathcal{L}_{t}} - \underbrace{\log \left( p_{\theta}(x_0 \mid x_1)\right)}_{\mathcal{L}_0} \\
\end{align} $$

&nbsp;

- The variational lower bound can thus be rewritten as follow:

<div style="text-align:center">
<span style="color:#00478F">
$$ \mathcal{L}_{VLB} = \mathcal{L}_T +\sum^{T-1}_{t=1}{\mathcal{L}_t} + \mathcal{L}_0$$
</span>
</div>

where

<div style="text-align:center">
<span style="color:#00478F">
$$ \begin{align} 
& \mathcal{L}_T = D_{KL} \left(q(x_T \mid x_0) \parallel p_\theta(x_{T})\right)\\
& \mathcal{L}_t = D_{KL}\left(q(x_t \mid x_{t+1},x_0) \parallel p_{\theta}(x_t \mid x_{t+1})\right)\\
& \mathcal{L}_0 = -\log \left( p_{\theta}(x_0 \mid x_1)\right)\\
\end{align}$$
</span>
</div>

&nbsp;

**$$\mathcal{L}_T:$$ Constant Term**

Then, $$\mathcal{L}_T = D_{KL}(q(x_T \vert x_0) \| p_\theta(x_{T}))$$ can be ignored for the training. Indeed, q has no learnable parameters and p is a gaussian noise probability so the term $$\mathcal{L}_T$$ is a constant.

&nbsp;

**$$\mathcal{L}_0:$$ Reconstruction term**

This is the reconstruction loss of the last denoising step. To obtain discrete log likelihoods, we set it to an independent discrete decoder derived from the Gaussian $$ \mathcal{N}(x_0,\mu _{\theta} (x_1,1), \Sigma _{\theta} (x_1,1)) $$ :

$$ p _{\theta} (x_0 | x_1) = \prod^{D}_{i=1} \int_{\delta - (x_0^i)}^{\delta + (x_0^i)}{\mathcal{N}(x_0,\mu _{\theta} (x_1,1), \Sigma _{\theta} (x_1,1)) dx} $$


$$\delta _{+} (x) = \begin{cases}
\infty & \text{if $x = 1$} \\
x + \frac{1}{255} & \text{if $x < 1$}
\end{cases}
$$
$$\delta _{-} (x) = \begin{cases}
-\infty & \text{if $x = -1$} \\
x - \frac{1}{255} & \text{if $x > -1$}
\end{cases}
$$

where $$D$$ is the data dimensionality and $$i$$ indicate the extraction of one coordinate.

&nbsp;

**$$\mathcal{L}_t:$$ Stepwise denoising terms**

This term compares, at each step of the process, the target $$q$$ and the approximation $$p_{\theta}$$. We recall that $$q(x_{t-1} \vert x_t,x_0) \sim \mathcal{N}(x_{t-1}, \tilde \mu _t(x_t,x_0), \tilde \beta _t \cdot \textbf{I})$$.

Moreover, $$\tilde \mu _t(x_t,x_0) = \frac{1}{\sqrt{\bar \alpha_t}} (x_t - \frac{1-\alpha _t}{1- \bar \alpha _t} \epsilon _t)$$ and $$\tilde \beta _t = \frac{1 - \bar \alpha _{t-1}}{1 - \bar \alpha _t} \cdot \beta _t$$.

Indeed, using [Bayes theorem](#bayes-theorem) : 

$$q(x_{t-1} \vert x_t,x_0) = q(x_t \vert x_{t-1},x_0) \frac {q(x_{t-1}\vert x_0) }{q(x_t \vert x_0) }$$

Given that the form of the probability density function of a normal distribution $$ \mathcal{N}(\mu , \sigma ^2)$$ is $$f(x) =  \frac{1}{\sigma \sqrt{2 \pi }} e^{-\frac{1}{2}(\frac{x- \mu}{\sigma})^2}$$, we obtain :

$$\begin{align} 
q(x_{t-1} \vert x_t,x_0)  & \propto exp[-\frac{1}{2} ( \frac{(x_t - \sqrt{\alpha _t} x_{t-1})^2}{\beta _t} + \frac{(x_{t-1} - \sqrt{\bar \alpha _{t-1}} x_0)^2}{1 - \bar \alpha _{t-1}} + \frac{(x_t - \sqrt{\bar \alpha _t} x_0)^2}{1 - \bar \alpha _t})] \\

& =  exp[-\frac{1}{2} ( \frac{x_t^2 - 2 \sqrt{\alpha _t} x_t x_{t-1} + \alpha _t x_{t-1}^2}{\beta _t} + \frac{x_{t-1}^2 - 2 \sqrt{\bar \alpha _{t-1}} x_0 x_{t-1} + \bar \alpha _{t-1} x_0^2}{1 - \bar \alpha _{t-1}} + \frac{(x_t - \sqrt{\bar \alpha _t} x_0)^2}{1 - \bar \alpha _t})]  \\

& = exp[-\frac{1}{2} ( (\frac{\alpha _t}{\beta _t}+ \frac{1}{1 - \bar \alpha _{t-1}})x_{t-1}^2 - (\frac{2 \sqrt{\alpha _t}} {\beta _t} x_t + \frac{2 \sqrt{\bar \alpha _{t-1}}}{1 - \bar \alpha _{t-1}}x_0)x_{t-1} + C(x_t, x_0)) ]  \\
\end{align} $$

By identification we find :

$$\tilde \beta _t = \frac{1}{\frac{\alpha _t}{\beta _t} + \frac{1}{1 - \bar \alpha _{t-1}}} = \frac{1}{\frac{\alpha _t- \bar \alpha _t + \beta _t}{\beta _t (1 - \bar \alpha _{t-1})}} = \frac{1 - \bar \alpha _{t-1}}{1 - \bar \alpha _t} \cdot \beta _t$$

$$\begin{align} 
\tilde \mu _t(x_t,x_0) & =  (\frac{ \sqrt{\alpha _t}} {\beta _t} x_t + \frac{ \sqrt{\bar \alpha _{t-1}}}{1 - \bar \alpha _{t-1}}x_0 )  \cdot \tilde \beta _t\\
& =  (\frac{ \sqrt{\alpha _t}} {\beta _t} x_t + \frac{ \sqrt{\bar \alpha _{t-1}}}{1 - \bar \alpha _{t-1}}x_0 )  \frac{1 - \bar \alpha _{t-1}}{1 - \bar \alpha _t} \cdot \beta _t\\
&= \frac{\sqrt{ \alpha _t }(1 - \bar \alpha _{t-1})}{ 1 - \bar \alpha _t} x_t + \frac{\sqrt{\bar \alpha _{t-1}} \beta _t}{1 - \bar \alpha _t}x_0 \\

\end{align} $$

Finally , using $$x_t = \sqrt{\bar \alpha _t} x_0 + \sqrt{1 - \bar \alpha _t}  \epsilon  $$, we have $$x_0 = \frac{1}{\sqrt{\bar \alpha _t}} ( x_t - \sqrt{1 - \bar \alpha _t} \epsilon  )$$. We rewritte :

$$\begin{align} 
\tilde \mu _t(x_t,x_0) & = \frac{\sqrt{ \alpha _t }(1 - \bar \alpha _{t-1})}{ 1 - \bar \alpha _t} x_t + \frac{\sqrt{\bar \alpha _{t-1}} \beta _t}{1 - \bar \alpha _t} \frac{1}{\sqrt{\bar \alpha _t}}(x_t - \sqrt{1 - \bar \alpha _t} \epsilon  )\\
&= \frac{1}{\sqrt{\bar \alpha_t}} (x_t - \frac{1-\alpha _t}{1- \bar \alpha _t} \epsilon )
\end{align}$$

Then , we would like to train $$\mu _{\theta}$$ to predict $$\tilde \mu _t$$ and as $$x_t$$ is available as input at training time, we can reparameterize it to make the neural network to predict $$\epsilon$$ from $$x_t$$ at time step t : 

$$\mu _{\theta} (x_t , t) = \frac{1}{\sqrt{\bar \alpha_t}} (x_t - \frac{1-\alpha _t}{1- \bar \alpha _t} \epsilon _{\theta} (x_t ,t))$$

The loss term $$\mathcal{L}_t$$ is defined to minimize the difference from $$\tilde \mu $$:

$$ \begin{align}
\mathcal{L}_t &= \mathbb{E}_{x_0, \epsilon} [ \frac{1}{ 2 \|\Sigma _{\theta} (x_t , t) \|^2_2} \| \tilde \mu _t (x_t , x_0) - \mu _{\theta}(x_t , t) \|^2]\\
&= \mathbb{E}_{x_0, \epsilon} [ \frac{1}{ 2 \|\Sigma _{\theta} (x_t , t) \|^2_2} \| \frac{1}{\sqrt{\bar \alpha_t}} (x_t - \frac{1-\alpha _t}{1- \bar \alpha _t} \epsilon ) -  \frac{1}{\sqrt{\alpha_t}} (x_t - \frac{1-\alpha _t}{1- \bar \alpha _t} \epsilon _{\theta}(x_t , t) ) \|^2]\\
&= \mathbb{E}_{x_0, \epsilon} [ \frac{(1-\alpha _t)^2}{ 2 \alpha _t (1- \bar \alpha_t)\|\Sigma _{\theta} (x_t , t) \|^2_2} \| \epsilon  - \epsilon _{\theta}(x_t , t)\|^2]\\
&= \mathbb{E}_{x_0, \epsilon} [ \frac{(1-\alpha _t)^2}{ 2 \alpha _t (1- \bar \alpha_t)\|\Sigma _{\theta} (x_t , t) \|^2_2} \| \epsilon  - \epsilon _{\theta}(\sqrt{\bar \alpha _t} x_0 + \sqrt{1 - \bar \alpha _t}\epsilon, t)\|^2]
\end{align}$$

&nbsp;


## **To go further**

### The first DDPM algorithm

Empirically , a first article on DDPM proposed to fix $$\beta _t$$ as constants because they found that learning a diagonal variance $$\Sigma _{\theta}$$ leads to unstable training and poorer sample quality. Thus, instead of making $$\beta _t$$ learnable, they set $$\Sigma _{\theta} (x_t ,t) = \sigma_t^2 I$$. Then, they found experimentally that both $$\sigma _t^2 = \tilde \beta _t = \frac{1- \bar \alpha _{t-1}}{1 - \bar \alpha _t} \beta _t$$ and $$\sigma _t^2 = \beta _t$$ had similar results. Indeed the first is optimal for $$ x_0 \sim \mathcal{N}(0,I)$$ and the second one is optimal for $$x_0$$ set to one point deterministically. These are in facts the two extrem choices and correspond to the upper and the lower bounds on reverse process entropy. With this simplifacation, the loss at step $$t$$ become the following :

$$ \mathcal{L}_t = \mathbb{E}_{x_0, \epsilon} [ \frac{(1-\alpha _t)^2}{ 2 \alpha _t (1- \bar \alpha_t)\sigma _{t}^2} \| \epsilon  - \epsilon _{\theta}(\sqrt{\bar \alpha _t} x_0 + \sqrt{1 - \bar \alpha _t}\epsilon, t)\|^2]$$

They also found that a simplification in the loss by ignoring the weithing term would improve the training of the model, making it less noisy:

$$ \mathcal{L}_t^{simple} = \mathbb{E}_{x_0, \epsilon} [ \| \epsilon  - \epsilon _{\theta}(\sqrt{\bar \alpha _t} x_0 + \sqrt{1 - \bar \alpha _t }\epsilon, t)\|^2]$$

$$\mathcal{L}_{simple} = \mathcal{L}_t^{simple} + C$$
where C is a constant that does not depends on $$\theta$$.

Finally, to sample $$x_{t-1} \sim p_{\theta} (x_{t-1} \vert x_t)$$, we compute $$x_{t-1} = \frac{1}{\sqrt{\bar \alpha_t}} (x_t - \frac{1-\alpha _t}{1- \bar \alpha _t} \epsilon _{\theta} (x_t ,t)) + \sigma _t z $$ where $$ z \sim \mathcal{N}(0,1)$$.

![](/collections/images/ddpm/algorithms.jpg)

### How to improve the log-likelihood ?

However, if DDPM have been shown to produce excellent samples,  the log-likelihood did not reflect this quality. The second article suggests that fixing $$\sigma _t$$ is a reasonable choice for the sake of sample quality but as the first few steps of the diffusion process contribute the most to the variational lower bound, using a better choice of $$\Sigma _{\theta}$$ would improve  the log-likelihood. Thus, they proposed to learn $$\Sigma _{\theta} (x_t, t)$$ as an interpolation of $$\beta _t$$ and $$\tilde \beta _t$$ by model predicting a vector $$\mathbf v$$ :

$$ \Sigma _{\theta} (x_t, t) = exp (\mathbf  v log(\beta _t)+ (1- \mathbf v) log(\tilde \beta _t))$$

However, $$\mathcal{L}_{simple}$$ does not depend  on $$\Sigma _{\theta} (x_t,t)$$. They defined a new hybrid objective :

$$\mathcal{L}_{hybrid} = \mathcal{L}_{simple} + \lambda \mathcal{L}_{_vlb}$$

They set $$\lambda = 0{,}001 $$ to prevent $$\mathcal{L}_{vlb}$$ from overwhelming $$\mathcal{L}_{simple}$$ and applied a stop gradient to the $$ \mu _{\theta} (x_t,t) $$ output of the $$\mathcal{L}_{vlb}$$ term. The objective was to make $$\mathcal{L}_{vlb}$$ guide $$\Sigma _{\theta} (x_t,t)$$ and keeping $$\mathcal{L}_{simple}$$ as the main source of influence over $$ \mu _{\theta} (x_t,t) $$.


### Improve sampling speed

The principal weakness of diffusion problem is the time-consuming process of sampling. Indeed, it is very slow to generate a sample from DDPM by following the Markov chain of the reverse process and producing a single sample can takes several minutes on a modern GPU. Thus, it is hard to produce high-resolution samples.

One simple way to improve this is to run a strided sampling scheduler. For a model trained with T diffusion steps, instead of sample using the same sequence of t values $$(1, 2, \dots , T )$$ as used during training, the sample uses an arbitrary subsequence S of t values. Thus, given the training noise schedule $$\bar \alpha _t$$, for a given sequence S  the sampling noise schedule $$\bar \alpha _{S_t}$$ is used to obtain corresponding sampling variances:

$$ \beta _{S_t} = 1 - \frac{\bar \alpha _{S_t}}{\bar \alpha _{S_{t-1}}} , \tilde \beta _{S_t} = \frac{1 - \bar \alpha _{S_{t-1}}}{1- \bar \alpha _{S_t}}.$$

Another algorithm, DDIM, modified the forward diffusion process making it non-Marovian to speed up the generation.

### Conditional image generation : Guided diffusion

The conditioning of the sampling is a crucial point of image generation. The aim of the conditioning, or guidance, is to incorporate image embeddings into the diffusion to "guide" the generation. It refers to conditioning a prior data distribution with a condition y : the class label or an image/text embedding. To turnour diffusion model into a conditional model, we conditione each diffusion step wih the information $$y$$: 

$$ p_{\theta}(x_{0:T} \vert y) =  p_{\theta}(x_T) \prod_{i=1}^{T}{ p_{\theta}(x_{t-1} \vert x_t , y)}$$

A first idea for conditionning is to use a second model, called a classifier, to guide diffusion during the trainning. Therefore, we train a classifier $$ p_{\phi}(y \vert x_t,t) $$ that approximates the label distribution for a noised sampple $$x_t$$.

Indeed, let's define a Markovian process $$\hat q$$ such as $$\hat q(y \vert x_0) is known and :

$$\hat q(x_0) := q(x_0)$$
$$\hat q(x_t \vert x_{t-1},y) := q(x_t \vert x_{t-1})$$
$$\hat q(x_{1:T} \vert x_0,y) := \prod^T_{t=1}\hat q(x_t \vert x_{t-1},y)$$

Then, we can easily find that $$\hat q$$ behaves exacly like $$q$$ when not conditioned on $$y$$ :

$$\hat q(x_t \vert x_{t-1}) = \int_y \hat q(x_t,y \vert x_{t-1})dy$$

Thanks to conditional probability formula : 

$$\hat q(x_t \vert x_{t-1}) = \int_y \hat q(x_t\vert x_{t-1},y)q(y \vert x_{t-1})dy$$

And by definition:

$$\hat q(x_t \vert x_{t-1}) = \int_y q(x_t\vert x_{t-1}) \hat q(y \vert x_{t-1})dy =  q(x_t\vert x_{t-1}) \int_y \hat q(y \vert x_{t-1}) dy= q(x_t \vert x_{t-1}) = \hat q(x_t\vert x_{t-1},y)$$

With a similar method, we can find $$\hat q(x_{1:T} \vert x_0) = q(x_{1:T} \vert x_0)$$ and $$\hat q(x_t) = q(x_t)$$. Thus using [Bayes theorem](#bayes-theorem) we also have $$\hat q(x_{t-1} \vert x_t) = q(x_{t-1} \vert x_t)$$.

$$\begin{align}
\hat q(x_{t-1}\vert x_t,y) &= \frac{\hat q(x_t,x_{t-1},y)}{\hat q(x_t,y)}\\
&= \frac{\hat q(x_t,x_{t-1},y)}{\hat q(y \vert x_t)\hat q(x_t)}\\
&= \frac{\hat q(x_t,x_{t-1},y)}{\hat q(y \vert x_t)\hat q(x_t)}\\
&= \frac{\hat q(x_{t-1} \vert x_t) \hat q(y \vert x_{t-1} , x_t)\hat q(x_t)}{\hat q(y \vert x_t)q(x_t)}\\
&= \frac{q(x_{t-1} \vert x_t) \hat q(y \vert x_{t-1})}{\hat q(y \vert x_t)}\\
\end{align}$$

Thus, we can treat $$\hat q(y \vert x_t)$$ as a constant, as it does not depend on $$x_{t-1}$$. Our problem is thus to sample from the distribution $$Z\hat q(x_{t-1} \vert x_t) \hat q(y \vert x_{t-1})$$. We already have a neural network $$p_{\theta}(x_{t-1} \vert x_t)$$ that approximate $$q(x_{t-1} \vert x_t)$$. The approximation of $$\hat q(y \vert x_{t-1})$$ is then done by training a classifier $$ p_{\phi}(y \vert x_t,t) $$.

![](/collections/images/ddpm/classifierAlgorithms.jpg)

Note that is also possible to run conditional difusion steps by incorporating the scores from a conditional and an unconditional diffusion model even if the method is not explained here.