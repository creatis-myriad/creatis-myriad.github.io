---
layout: post
title:  "The denoising diffusion probabilistic models (DDPM) paradigm demystified"
author: 'Celia Goujat, Olivier Bernard'
date:   2023-11-16
categories: diffusion, model
---

# Notes

* Here are links to two video that we used to create this tutorial: [video1](https://www.youtube.com/watch?v=HoKDTa5jHvg&ab_channel=Outlier), [video2](https://www.youtube.com/watch?v=TBCRlnwJtZU&ab_channel=Outlier). 
* We was also strongly inspired by this excellent [post](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/).

&nbsp;
- [**Introduction**](#introduction)
- [**Fondamental concepts**](#fondamental-concepts)
  - [Sum of normally distributed variables](#sum-of-normally-distributed-variables)
  - [Bayes theorem](#bayes-theorem)
  - [Reparameterization trick](#reparameterization-trick) 
  - [Cross Entropy](#cross-entropy)    

- [**Forward diffusion process**](#forward-diffusion-process)
  - [Principle](#principle) 
  - [How to define the scheduler ?](#how-to-define-the-scheduler) 
- [**Backward process**](#backward-process)
  - [General idea](#general-idea) 
  - [Loss function](#loss-function) 
- [**To go further**](#to-go-further)
  - [The first DDPM algorithm](#first-ddpm-algorithm) 
  - [How to improve the log-likelihood ?](#how-to-improve-the-log-likelihood) 
  - [Improve sampling speed](#improve-sampling-speed)
&nbsp;

## **Introduction**

Diffusion models are a class of generative models such as GAN, [normalizing flow](http://127.0.0.1:4000/tutorials/2023-01-05-tutorial_normalizing_flow.html) or [variational auto-encoders](http://127.0.0.1:4000/tutorials/2022-09-12-tutorial-vae.html). A diffusion probabilistic model involves a forward stage where the original input image undergoes gradual perturbation by adding random noise. The model learns to reverse this process, constructing data samples from noise. These models have demonstrated effectiveness in generating high-quality samples that encompass a wide range of modes.  However, they come with significant computational burdens. Unlike VAEs or flow models, diffusion models have high-dimensional latent variables equivalent to the input.

&nbsp;

<div style="text-align:center">
<img src="/collections/images/ddpm/diffusionModels.jpg" width=600></div>

&nbsp;

Diffusion model can be classified into two main perspective : **the variational perspective** and the **score perspective**. [Score models](http://127.0.0.1:4000/tutorials/2023-05-09-tutorial-score-based-models.html) are based on a maximum likelihood-based estimation approach. They use the score function of the loglikelihood of the data to estimate parameters within the diffusion process. On the other hand, the variational perspective involves models that approximate the target distribution using variational inference. Typically, these models minimize the Kullback-Liebler (KL) divergence between the target and the approximate distribution. An example of a diffusion model following the variational perspective is the Denoising Diffusion Probabilistic Model (DDPM).

![](/collections/images/ddpm/diffusionProcess.jpg)


DDPMs are inspired by non-equilibrum thermodynamics. The fundamental concept is to gradually dismantling the structure present in the data distribution and subsequently learning a reverse process that reconstructs this structure within the data. In the forward diffusion process, we start with an original image. A step-by-step addition of noise, with a sufficient number of iteration, make the image become pur noise. In existing DDPMs, this noise is sampled from a normal distribution. The quantity of noise added at each step increases and is specified by a predefined schedule that determines the variance of the noise. This schedule is designed to prevent explosive growth of the noise.

The reverse process, also know as denoising, should recover an image from pur noise. This process involves a neural network that has learned to remove noise from an image step by step. The objective is to input noise sampled from a normal distribution into the model, so it could gradually remove the noise until generating a clear image as output. This stepwise process is preferred over direct transformation from noise to image, as the latter would not be tractable and would give inferior results. The neural networks predicts, at each step, the noise of the image. Since the noise follows a normal distribution, predicting the noise is equals to predict both its mean and its variance. Some DDPMs fix the variance, so the neural network only needs to predict the mean of the noise.

&nbsp;

## **Fondamental concepts**

### Sum of normally distributed variables

Let X and Y be independent random variables that are normally distributed $$ X \sim \mathcal{N}(\mu _X , \sigma ^2_X)$$ and $$ Y \sim \mathcal{N}(\mu _Y , \sigma ^2_Y)$$. Then, their sum is defined as follow : 

$$ X + Y \sim \mathcal{N}(\mu _X + \mu _Y, \sigma ^2_X + \sigma ^2_Y)$$

This means that the sum of two independent normally distributed random variables is also normally distributed, with its mean being the sum of the two means, and its variance being the sum of the two variances.

It is noteworthy that if $$ X \sim \mathcal{N}(0, 1)$$ then $$ \sigma X \sim \mathcal{N}(0, \sigma ^2)$$. Thus if X and Y be independent standard normal random variables $$ X \sim \mathcal{N}(0, 1)$$ and $$ Y \sim \mathcal{N}(0, 1)$$, the sum $$ \sigma _X X +  \sigma _Y Y $$ is normally distributed such as $$\sigma _X X +  \sigma _Y Y \sim \mathcal{N}(0, \sigma ^2_X + \sigma ^2_Y)$$.

### Bayes Theorem

Bayes theorem describes the probability of occurrence of an event related to any condition. We assume that a data $$x_t$$ is generated by our model from a probability distribution depending on the data $$ x _{t-1}$$. We also assume that we have a prior knowledge about the data $$ x _{t-1} $$ that can be expressed as a probability distribution $$q(x _{t-1})$$. Thus, when the data $$x_t$$ is observed, we can update the prior knowledge using Bayes theorem as follows:

$$ q(x_{t-1} \vert x_t) = \frac{q(x_t \vert x_{t-1})q(x_{t-1})}{q(x_t)} $$


### Reparameterization trick

The idea of the reparametirization trick is to transform a stochastic node sampled from a parameterized distribution into a deterministic ones. Indeed, in a stochastic operation, you cannot perform backpropagation. Reparameterization allows a gradient path through such a stochastic node by turning him into  deterministic node. 

Let's assume that $$x_t$$ is a point sampled from a parameterized gaussian distribution $$q(x_t)$$ with mean $$\mu$$ and variance $$\sigma^2$$. Our model should learn the distribution to subsequently reverse the process. Thus, we want to treat the noise term as a standard normal distribution $$\mathcal{N}(0,1)$$ that is independent of our model (not parameterized by it).  The reparameterization trick make it possible: the sample is drawn from a fixed Gaussian distribution, which we add our $$\mu$$ to and multiply by our standard deviation $$\sigma$$.

$$ x_t = \mu + \sigma \cdot \epsilon$$

The epsilon term introduces the stochastic part ($$\epsilon \sim \mathcal{N}(0,1)$$) and is not involved in the training process. The prediction of the mean and the variance is no longer tied to the stochastic sampling operation. Therefore, we can now compute the gradient and run backpropagation of our loss function with respect to the parameters. 


![](/collections/images/ddpm/reparameterizationTrick.jpg)

### Information theory reminder

In information theory, Entropy $$H(p)$$ corresponds to the average information of a process. Its expression can be written as, for continuous distributions:

$$ H(p) = -\int{p(x)\cdot log\left(p(x)\right)}\,dx$$

From this, we can also define the cross entropy between two probability distributions $$p$$ and $$q$$ concerning the same events. 

$$H_{pq} = -\int{p(x)\cdot log\left(q(x)\right)}\,dx = \mathbb{E}_{p(x)} [log(q(x))]$$

The cross entropy quantifies the average bit requirement to identify an event selected from a set when using a coding scheme optimized for the estimated probability distribution q, instead of the actual distribution p. We can also define the Kullback–Leibler divergence from this two definitions : 

$$\begin{align}
D_{KL}(p \| q) &= H(p,q) - H(p) \\
& = -\int{p(x)\cdot log(q(x))}\,dx + \int{p(x)\cdot log(p(x))}\,dx \\
& = -\int{p(x)\cdot log(\frac{q(x)}{p(x)})}\,dx \\
& = \int{p(x)\cdot log(\frac{p(x)}{q(x)})}\,dx \\
& = \mathbb{E}_{p(x)} [log(\frac{p(x)}{q(x)})]
\end{align}$$

Cross-entropy minimization is frequently used in optimization and rare-event probability estimation. Note that when comparing a distribution $$q$$ against a fixed reference distribution $$p$$, cross-entropy and KL divergence are identical up to an additive constant (since $$p$$ is fixed). Note also that $$H(p)$$, $$H(p,q)$$ and $$D_{KL}(p \| q)$$ are always positives.


## **Forward diffusion process**

### Principle

![](/collections/images/ddpm/forwardProcess.jpg)

Let define $$x_0$$ a point sampled from a real data distribution $$x_0 \sim q(x)$$. The forward process of a probabilistic diffusion model is a Markov chain \(the prediction at the step $$t+1$$ only depends on the state attained at the step $$t$$\) that gradually adds gaussian noise to the data $$x_0$$. We define a forward process with T steps that produces more and more noisy samples $$x_1, x_2,\ldots, x_T$$ : 

$$q(x_{1:T}|x_0) = \prod_{t=1}^{T}{q(x_t|x_{t-1})}$$

Note in particular that the Markov formulation asserts that a given reverse diffusion transition distribution depends only on the previous timestep. The amount of noise added at each step is controlled by a variance schedule $$\{\beta_t \in (0,1)\}_{t=1}^{T}$$ : 

$$q(x_t|x_{t-1}) = \mathcal{N}(x_t,\sqrt{1 - \beta _t},\beta _t \cdot \textbf{I})$$

Usually, the scheduler is defined such as $$ \beta _1 <\beta _2 < \ldots < \beta _T $$. A key point in the quality of the results is the construction of the schedule. However, this sampling process is stochastic and so a backprogation for the training of the reverse process will not be possible. We use a [reparameterization trick](#reparameterization-trick) to make the sampling deterministic:

$$x_t = \sqrt{1 - \beta _t} x_{t-1} + \sqrt {\beta _t} \epsilon _{t-1} $$

Let define $$\alpha _t = 1 - \beta _t $$ and $$\bar \alpha _t = \prod_{k=1}^{t}{\alpha _k}$$. As the previous formula is true for all $$t \in [1,T]$$ we can rewrite :

$$x_t = \sqrt{1 - \beta _t} (\sqrt{1 - \beta _{t-1}} x_{t-2} + \sqrt {\beta _{t-2}} \epsilon _{t-2} )+ \sqrt {\beta _t} \epsilon _{t-1} $$

$$x_t = \sqrt{\alpha _t \alpha _{t-1}} x_{t-2} + \sqrt{\alpha _t - \alpha _t \alpha _{t-1}} \epsilon _{t-2} + \sqrt {1- \alpha_t} \epsilon _{t-1} $$

Using the formula for the [sum of normally distributed variables](#sum-of-normally-distributed-variables) we obtain:

$$x_t = \sqrt{\alpha _t \alpha _{t-1}} x_{t-2} + \sqrt{1 - \alpha _t \alpha _{t-1}} \bar \epsilon _{t-2} $$

Then we can extend this to earlier timesteps recursively and define directly $$x_t$$ from $$x_0$$: 

$$x_t = \sqrt{\bar \alpha _t} x_0 + \sqrt{1 - \bar \alpha _t} \bar \epsilon _0 $$

Notice that when $$T \rightarrow  \infty $$, $$ x_T $$ is equivalent to a pure noise (Gaussian distribution). We have therefore defined a forward process called a diffusion probabilistic process that introduce slowly noise into an image. 

### How to define the scheduler ?

The first article presenting DDPM set the forward variances to be a sequence of linearly increasing constants. This sequence, determining the variance values, is referred to as a scheduler. They are chosen to be relatively small compared to data scaled to [-1,1]. This ensure that reverse and forward process maintain approximately the same functionnal form.

A second article improved the results given by the ddpm algorithm. A first suggestion for this improvment was in the definition of the scheduler : the linear noise schedule work well for high resolution images but proved to be sub-optimal for images of lower resolution (64x64 and 32x32). Indeed, the end of the forward noising process is very noisy with most of the steps being useless. A cosine scheduler was provided :

$$ \alpha _t = \frac{f(t)}{f(0)}, f(t) = cos (\frac{\frac{t}{T} + s}{1+s} \cdot \frac{\pi}{2})^2$$

The variances $$\beta _t$$ can be deducted from this definition as $$ \beta _t = 1 - \frac{\bar \alpha _t}{\bar \alpha _{t-1}}$$ with in pactice $$\beta _t $$ being cliped to be no larger than $$0{,}999$$ to prevent singularities for $$t \rightarrow T$$.

![](/collections/images/ddpm/schedules.jpg)


## **Backward process**

### General Idea

Now, if we are able to reverse the above diffusion process, we will be able to generate a sample from a Gaussian noise input $$x_T \sim \mathcal{N}(0,1)$$. However, unlike the forward process, we can not use $$q(x_{t-1} \vert x_t)$$ to reverse the noise because it is intractable: it needs the entire dataset to estimate. 

![](/collections/images/ddpm/markovProcess.jpg)

It is noteworthy that the reverse conditional probability is tractable when conditioned on $$x_0$$. Indeed, thanks to [Bayes theorem](#bayes-theorem) $$q(x_{t-1} \vert x_t, x_0)$$ = $$\frac{q(x_t \vert x_{t-1}, x_0)q(x_{t-1}\vert  x_0)}{q(x_t \vert  x_0)}$$ where all distribution are known from the forward process.

Then note that if $$\beta _t$$ is small enough, $$q(x_{t-1} \vert x_t,x_0)$$ will also be a Gaussian, $$q(x_{t-1} \vert x_t,x_0) \sim \mathcal{N}(x_{t-1}, \tilde \mu _t(x_t,x_0), \tilde \beta _t \cdot \textbf{I})$$.

Thus, we will train a neural network $$p_{\theta}(x_{t-1} \vert x_t) \sim \mathcal{N}(x_{t-1},\mu _{\theta}(x_t,t), \Sigma _{\theta}(x_t,t) \cdot \textbf{I})$$ to approximate $$q(x_{t-1} \vert x_t,x_0)$$. What is important here is that the learned parameters are time-dependent. 

![](/collections/images/ddpm/reverseProcess.jpg)

### Loss function

The final objective of our reverse process is to have the more similarity between the generated images and the original images. The cross entropy between $$q(x_0)$$ and $$p_{\theta}(x_0)$$ is suitable as loss function. By minimizing the cross entropy between $$q(x_0)$$ and $$p_{\theta}(x_0)$$, we are minimizing the divergence between the two distributions.

$$H(p_{\theta}(x_0),q(x_0)) = - \mathbb{E}_{q(x_0)}[log( p_{\theta}(x_0))]$$

However, $$ p_{\theta}(x_0) $$ depends on $$x_1, x_2, \dots, x_T$$ and so it is intractable (uncomputable), let's rewrite it to find a computable loss: 

$$ \begin{align} 
H(p_{\theta}(x_0),q(x_0))  & = - \mathbb{E}_{q(x_0)}[log(\int p_{\theta}(x_{0:T}) d_{x_{1:T}})] \\
&  = - \mathbb{E}_{q(x_0)}[log( \int q(x_{1:T} \vert x_0)\frac{p_{\theta}(x_{0:T})}{q(x_{1:T} \vert x_0)}  d_{x_{1:T}})] \\
&  = - \mathbb{E}_{q(x_0)}[log( \mathbb{E}_{q( x_{1:T} \vert x_0)}\frac{p_{\theta}(x_{0:T})}{q(x_{1:T} \vert x_0)})]
\end{align}$$

Now, we can use Jensen's inequality :

$$ \begin{align} 
H(p_{\theta}(x_0),q(x_0))  &\leq - \mathbb{E}_{q(x_0)}\mathbb{E}_{q( x_{1:T} \vert x_0)}[log(\frac{p_{\theta}(x_{0:T})}{q(x_{1:T} \vert x_0)})]= - \mathbb{E}_{q( x_{0:T} \vert x_0)}[log(\frac{p_{\theta}(x_{0:T})}{q(x_{1:T} \vert x_0)})] = \mathbb{E}_{q( x_{0:T} \vert x_0)}[log(\frac{q(x_{1:T} \vert x_0)}{p_{\theta}(x_{0:T})})]\\
\end{align}$$


Let $$ \mathcal{L}_{VLB} = \mathbb{E}_q[log( \frac{q(x_{1:T} \vert x_0)}{p_{\theta}(x_{0:T})})]$$. Now, instead of minimizing H(p_{\theta}(x_0),q(x_0)), we can minimize its Variational Lower Bound (VLB) $$\mathcal{L}_{VLB}$$. Note that the Variational Lower Bound can be found simply if we optimize the negative log-likelihood of $$p_{\theta}(x_0)$$, assuming the process is the same than the one of VAE, as the setups are very similar.

$$ \begin{align} 
-log( p_{\theta}(x_0))  & \leq -log( p_{\theta}(x_0)) + D_{KL}(q(x_{1:T} \vert x_0) \| p_{\theta}(x_{1:T} \vert x_0)) \\ 
& = -log( p_{\theta}(x_0)) + \mathbb{E}_q[log( \frac{q(x_{1:T} \vert x_0)}{p_{\theta}(x_{1:T} \vert x_0)})] \\
& = -log( p_{\theta}(x_0)) + \mathbb{E}_q[log( \frac{q(x_{1:T} \vert x_0)}{(\frac{p_{\theta}(x_{0:T})}{p_{\theta}(x_0)})})] \\
& = -log( p_{\theta}(x_0)) + \mathbb{E}_q[log( \frac{q(x_{1:T} \vert x_0)}{p_{\theta}(x_{0:T})}) + log(p_{\theta}(x_0))]\\
& = \mathbb{E}_q[log( \frac{q(x_{1:T} \vert x_0)}{p_{\theta}(x_{0:T})})]
\end{align} $$

We want to express each term of the loss to be anatycally computable, so using [Bayes theorem](#bayes-theorem), we can rewritte :

$$ \begin{align} 
\mathcal{L}_{VLB}  & = \mathbb{E}_q[log( \frac{q(x_{1:T} \vert x_0)}{p_{\theta}(x_{0:T})})] \\ 
& = \mathbb{E}_q(log( \frac{ \prod^T_{t=1} q(x_t \vert x_{t-1})}{p_{\theta}(x_{T}) \prod^T_{t=1} p_{\theta}(x_t \vert x_{t-1})})] \\

& = \mathbb{E}_q[-log(p_{\theta}(x_{T}) ) + \sum^T_{t=1} log( \frac{ q(x_t \vert x_{t-1})}{p_{\theta}(x_t \vert x_{t-1})})] \\

& = \mathbb{E}_q[-log(p_{\theta}(x_{T}) ) + \sum^T_{t=2} log( \frac{ q(x_t \vert x_{t-1})}{p_{\theta}(x_t \vert x_{t-1})}) + \log( \frac{ q(x_1 \vert x_0)}{p_{\theta}(x_1 \vert x_0)})] 
\end{align} $$

Then,using the conditionnal probability formula : 

$$ \begin{align} 
\mathcal{L}_{VLB}  & = \mathbb{E}_q[-log(p_{\theta}(x_{T}) ) + \sum^T_{t=2} log( \frac{ q(x_t \vert x_{t-1},x_0)q(x_t \vert x_0)}{p_{\theta}(x_t \vert x_{t-1})q(x_{t-1} \vert x_0)}) + \log( \frac{ q(x_1 \vert x_0)}{p_{\theta}(x_1 \vert x_0)})] \\

& = \mathbb{E}_q[-log(p_{\theta}(x_{T}) ) + \sum^T_{t=2} log( \frac{ q(x_t \vert x_{t-1},x_0)}{p_{\theta}(x_t \vert x_{t-1})}) + \sum^T_{t=2} log( \frac{q(x_t \vert x_0)}{q(x_{t-1} \vert x_0)}) + \log( \frac{ q(x_1 \vert x_0)}{p_{\theta}(x_1 \vert x_0)})] \\

& = \mathbb{E}_q[-log(p_{\theta}(x_{T}) ) + \sum^T_{t=2} log( \frac{ q(x_t \vert x_{t-1},x_0)}{p_{\theta}(x_t \vert x_{t-1})}) + log( \frac{q(x_T \vert x_0)}{q(x_1 \vert x_0)}) + \log( \frac{ q(x_1 \vert x_0)}{p_{\theta}(x_1 \vert x_0)})] \\

& = \mathbb{E}_q[log({\frac{q(x_T \vert x_0)}{p_\theta(x_{T})}} ) + \sum^T_{t=2} log( \frac{ q(x_t \vert x_{t-1},x_0)}{p_{\theta}(x_t \vert x_{t-1})}) + \underbrace{log(\frac{q(x_1 \vert x_0)}{q(x_1 \vert x_0)})}_{= 0} - log( p_{\theta}(x_1 \vert x_0))] \\

& = \mathbb{E}_q[log({\frac{q(x_T \vert x_0)}{p_\theta(x_{T})}} ) + \sum^T_{t=2} log( \frac{ q(x_t \vert x_{t-1},x_0)}{p_{\theta}(x_t \vert x_{t-1})}) - log( p_{\theta}(x_1 \vert x_0))] \\

& =  \underbrace{D_{KL}(q(x_T \vert x_0) \| p_\theta(x_{T}))}_{L_T} + \sum^T_{t=2}  \underbrace{D_{KL}(q(x_t \vert x_{t-1},x_0) \| p_{\theta}(x_t \vert x_{t-1}))}_{L_{t-1}} -  \underbrace{log( p_{\theta}(x_1 \vert x_0))}_{L_0} \\
\end{align} $$


Now that we expressed the variational lower bound as a sum of T+1 components, $$ \mathcal{L}_{VLB} = \mathcal{L}_T +\sum^T_{t=2}{\mathcal{L}_{t-1}} + \mathcal{L}_0$$, let take a look on each one.

**$$\mathcal{L}_T:$$ Constant Term**

Then, $$\mathcal{L}_T = D_{KL}(q(x_T \vert x_0) \| p_\theta(x_{T}))$$ can be ignored for the training. Indeed, q has no learnable parameters and p is a gaussian noise probability so the term $$\mathcal{L}_T$$ is a constant.

**$$\mathcal{L}_t:$$ Stepwise denoising terms**

This term compares, at each step of the process, the target $$q$$ and the approximation $$p_{\theta}$$. We recall that $$q(x_{t-1} \vert x_t,x_0) \sim \mathcal{N}(x_{t-1}, \tilde \mu _t(x_t,x_0), \tilde \beta _t \cdot \textbf{I})$$.

Moreover, $$\tilde \mu _t(x_t,x_0) = \frac{1}{\sqrt{\bar \alpha_t}} (x_t - \frac{1-\alpha _t}{1- \bar \alpha _t} \epsilon _t)$$ and $$\tilde \beta _t = \frac{1 - \bar \alpha _{t-1}}{1 - \bar \alpha _t} \cdot \beta _t$$.

Indeed, using [Bayes theorem](#bayes-theorem) : 

$$q(x_{t-1} \vert x_t,x_0) = q(x_t \vert x_{t-1},x_0) \frac {q(x_{t-1}\vert x_0) }{q(x_t \vert x_0) }$$

Given that the form of the probability density function of a normal distribution $$ \mathcal{N}(\mu , \sigma ^2)$$ is $$f(x) =  \frac{1}{\sigma \sqrt{2 \pi }} e^{-\frac{1}{2}(\frac{x- \mu}{\sigma})^2}$$, we obtain :

$$\begin{align} 
q(x_{t-1} \vert x_t,x_0)  & \propto exp[-\frac{1}{2} ( \frac{(x_t - \sqrt{\alpha _t} x_{t-1})^2}{\beta _t} + \frac{(x_{t-1} - \sqrt{\bar \alpha _{t-1}} x_0)^2}{1 - \bar \alpha _{t-1}} + \frac{(x_t - \sqrt{\bar \alpha _t} x_0)^2}{1 - \bar \alpha _t})] \\

& =  exp[-\frac{1}{2} ( \frac{x_t^2 - 2 \sqrt{\alpha _t} x_t x_{t-1} + \alpha _t x_{t-1}^2}{\beta _t} + \frac{x_{t-1}^2 - 2 \sqrt{\bar \alpha _{t-1}} x_0 x_t + \bar \alpha _{t-1} x_0^2}{1 - \bar \alpha _{t-1}} + \frac{(x_t - \sqrt{\bar \alpha _t} x_0)^2}{1 - \bar \alpha _t})]  \\

& = exp[-\frac{1}{2} ( (\frac{\alpha _t}{\beta _t}+ \frac{1}{1 - \bar \alpha _{t-1}})x_{t-1}^2 - (\frac{2 \sqrt{\alpha _t}} {\beta _t} x_t + \frac{2 \sqrt{\bar \alpha _{t-1}}}{1 - \bar \alpha _{t-1}}x_0)x_t + C(x_t, x_0)) ]  \\
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
\tilde \mu _t(x_t,x_0) & = \frac{\sqrt{ \alpha _t }(1 - \bar \alpha _{t-1})}{ 1 - \bar \alpha _t} x_t + \frac{\sqrt{\bar \alpha _{t-1}} \beta _t}{1 - \bar \alpha _t} \\
&= \frac{1}{\sqrt{\bar \alpha_t}} (x_t - \frac{1-\alpha _t}{1- \bar \alpha _t} \epsilon )
\end{align}$$

Then , we would like to train $$\mu _{\theta}$$ to predict $$\tilde \mu _t$$ and as $$x_t$$ is available as input at training time, we can reparameterize it to make the neural network to predict $$\epsilon$$ from $$x_t$$ at time step t : 

$$\mu _{\theta} (x_t , t) = \frac{1}{\sqrt{\bar \alpha_t}} (x_t - \frac{1-\alpha _t}{1- \bar \alpha _t} \epsilon _{\theta} (x_t ,t))$$

The loss term $$\mathcal{L}_t$$ is defined to minimize the difference from $$\tilde \mu $$:

$$ \begin{align}
\mathcal{L}_t &= \mathbb{E}_{x_0, \epsilon} [ \frac{1}{ 2 \|\Sigma _{\theta} (x_t , t) \|^2_2} \| \tilde \mu _t (x_t , x_0) - \mu _{\theta}(x_t , t) \|^2]\\
&= \mathbb{E}_{x_0, \epsilon} [ \frac{1}{ 2 \|\Sigma _{\theta} (x_t , t) \|^2_2} \| \frac{1}{\sqrt{\bar \alpha_t}} (x_t - \frac{1-\alpha _t}{1- \bar \alpha _t} \epsilon ) -  \frac{1}{\sqrt{\bar \alpha_t}} (x_t - \frac{1-\alpha _t}{1- \bar \alpha _t} \epsilon _{\theta}(x_t , t) ) \|^2]\\
&= \mathbb{E}_{x_0, \epsilon} [ \frac{(1-\alpha _t)^2}{ 2 \alpha _t (1- \bar \alpha_t)\|\Sigma _{\theta} (x_t , t) \|^2_2} \| \epsilon  - \epsilon _{\theta}(x_t , t)\|^2]\\
&= \mathbb{E}_{x_0, \epsilon} [ \frac{(1-\alpha _t)^2}{ 2 \alpha _t (1- \bar \alpha_t)\|\Sigma _{\theta} (x_t , t) \|^2_2} \| \epsilon  - \epsilon _{\theta}(\sqrt{\bar \alpha _t} x_0 + \sqrt{1 - \bar \alpha _t}\epsilon, t)\|^2]
\end{align}$$


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

**Classifier guidance**

A first idea for conditionning is to use a second model, called a classifier, to guide diffusion during the trainning. Therefore, we train a classifier $$ f_{\phi}(y \vert x_t,t) $$ on the image $$x_t$$ to predict its class $$y$$.

**Classifier free guidance**