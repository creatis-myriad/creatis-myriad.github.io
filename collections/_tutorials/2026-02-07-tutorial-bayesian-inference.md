---
layout: post
title:  "Introduction to Bayesian inference"
author: 'Olivier Bernard'
date:   2026-02-08
categories: Bayesian, posterior, likelihood, prior, distribution
---
<style>
  div.post-content p {
    text-align: justify; /* helps the reading flow */
  }
</style>

# Summary

- [**Introduction**](#introduction)
  - [What is a Bayesian method?](#what-is-a-bayesian-method)
  - [What is a Bayesian inference?](#what-is-a-bayesian-inference)
- [**Variational inference**](#variational-inference)
  - [Problem formulation](#problem-formulation)
  - [Modeling](#vi-modeling)
  - [Applications](#vi-applications)
- [**Amortized simulation-based inference**](#amortized-simulation-based-inference)
  - [Definition](#definition)
  - [Problem formulation](#problem-formulation)
  - [Modeling](#modeling)
  - [tabPFN application](#tabpfn-application)
- [**References**](#references)

&nbsp;

## **Introduction**

### What is a Bayesian method?

A Bayesian method is a statistical approach that relies on Bayes’ theorem to reason under uncertainty.

#### Main idea

We are not just trying to estimate an unknown value. Rather, we describe our uncertainty about this value using probabilities, and we update it as we observe data.

The three key building blocks are:
- <spam style="color:green">The prior probability (prior)</spam>: what we believe before seeing the data (prior knowledge, assumptions, domain expertise)
- <spam style="color:blue">The likelihood</spam>: how compatible the observed data are with a given hypothesis
- <spam style="color:red">The posterior probability (posterior)</spam>: what we believe after seeing the data

<div style="text-align:center">
<img src="/collections/images/bayesian-inference/bayes_theorem.jpg" width=600></div>

&nbsp;

> A Bayesian method updates probabilistic beliefs based on observed data

What this changes compared to “classical” methods:
- We obtain a full distribution over the parameters (not just a point estimate)
- Uncertainty can be quantified in a natural way
- Prior knowledge can be incorporated
- The results are often more interpretable (credible intervals, probability of an event, etc.)

&nbsp;

#### Example: diagnosing a disease based on a test

##### 1. Variables (clear notation)

We define the latent variable $$z \in \{0,1\}$$ as the subject status:
$$z = 1$$: subject is ill
$$z = 0$$: subject is healthy

We define the observed data as $$x$$. 
It can be the result of a medical test (e.g. biomarker, imaging, AI score).

We want to compute $$p(z=1 \mid x)$$, i.e. the posterior probability that the patient is ill, given the observed data.

##### 2. The prior

Before observing any data, we have baseline information on the disease prevalence. This will serve as a prior $$p(z)$$ on the targeted pathology. For example:
$$p(z=1) =0.01$$ => $$1\%$$ of prevalence, i.e. on average, $$1\%$$ of the population under consideration is affected by the disease.
$$p(z=0) =0.99$$

This is the medical prior (population-level, clinical context).

##### 3. The likelihood

The likelihood $$p(x \mid z$$ models the behavior of the test. Let's assume a sensitivity of $$95\%$$ and a specificity of $$90\%$$, thus
$$p(x=+ \mid z=1) = 0.95$$
$$p(x=+ \mid z=0) = 0.10$$

This is the test model, not yet the diagnosis.

##### 4. Bayes’ formula

$$p(z \mid x) = \frac{p(x \mid z) \, p(z)}{p(x)}$$

with $$p(x) = \sum_{z \in \{0,1\}}p(x \mid z) \, p(z)$$

##### 5. Explicit computation for a positive test

We want to know the probability of being effectively ill when a test is positive: $$p(z=1 \mid x=+)$$:

$$p(z=1 \mid x=+) = \frac{p(x=+ \mid z=1) \, p(z=1)}{p(x=+)}$$

Numerator: $$p(x=+ \mid z=1) \, p(z=1)$$ = $$0.95 \times 0.01 = 0.0095$$

Denumerator: $$p(x=+)$$ = $$0.95 \times 0.01 + 0.10 \times 0.99 = 0.1085$$

Posterior: $$p(z=1 \mid x=+) = \frac{0.0095}{0.1085} \approx 8.8 \%$$

##### 6. Clinical interpretation

Even with:
- a good test
- a positive result

the probability of actually being ill remains below $$10\%$$, due to the low prevalence. This is exactly what Bayesian inference captures and what human intuition often misses.

&nbsp;

### What is a Bayesian inference?

#### Definition

Bayesian inference is the process of:
- inferring unknown quantities from data
- by modeling them as probability distributions
- updating them using Bayes’ theorem

> Bayesian inference consists in computing the posterior distribution of parameters or latent variables given the observed data


#### What we actually infer?

Depending on the problem, we may infer:
- a parameter (e.g., disease probability, model weights)
- a latent variable (e.g., a hidden pathological state)
- a future prediction (e.g., event risk)

The estimation is always in the form of a distribution, not just a single number.

#### The mental model

Bayesian inference is always:

$$\text{prior} + \text{data} \rightarrow \text{posterior}$$

#### Inference ≠ learning

This is an important distinction:

- Bayesian learning
  $$\rightarrow$$ learning the parameters of a model
- Bayesian inference
  $$\rightarrow$$ reasoning about unknown quantities given the data

In practice, the two are often intertwined

#### What we get in the end

With Bayesian inference, you can:
- compute a mean estimate
- construct credible intervals
- compute event probabilities
- propagate uncertainty to predictions

#### And in practice (AI / imaging)

In modern models:
- the posterior is not analytically tractable
- it is approximated using:
  - Markov Chain Monte Carlo
  - variational inference
  - amortized simulation-based inference

## **Variational inference**

### Problem formulation

### Modeling

### Applications

## **Amortized simulation-based inference**

### Definition

The idea behind the amortized simulation-based inference is to model the ouput $$y$$ from a new input $$x$$ <spam style="color:blue">based on a supervised dataset</spam> $$D=(X_{\text{train}},y_{\text{train}})$$ of arbitrary size $$n$$. 

The goal is therefore to model the posterior predictive distribution $$p(y | x, D)$$. Since we explicitly use a support dataset $$D$$ to predict $$y$$ from $$x$$, this model falls under <spam style="color:blue">in-context learning</spam>.

Moreover, amortized simulation-based inference is based on the hypothesis that there exists a relationship between the inputs $$X$$ and the output labels $$y$$. This relationship can be modeled through a prior that can be used to generate synthetic datasets.

### Modeling

#### Modeling relationships in the data

The prior defines a space of hypotheses $$\Phi$$ on the relationship of a set of inputs $$X$$ to the output labels $$y$$. Each hypothesis $$\phi \in \Phi$$ can be seen as a mechanism that generates a data
distribution from which we can draw samples forming a dataset.

#### Problem formulation

Using the law of total probability, the posterior predictive distribution can be rewritten as:

$$p(y | x, D) = \int_{\phi}p(y | x, \phi) \, p(\phi | D) \, d\phi$$

$$p(y | x, D) \propto \int_{\phi}p(y | x, \phi) \, p(D | \phi) \, p(\phi) \, d\phi$$ 

#### Prior sampling scheme

Based on the hypothesis $$\Phi$$, one can implement an efficient prior sampling scheme of the form: 

$$p(D) = \int_{\phi} p(D | \phi) \, p(\phi) \, d\phi$$ 

The generative mechanism is first sampled as $$\phi \sim p(\phi)$$, and then the synthetic dataset is sampled as $$D\sim p(D | \phi)$$. 

#### Learning process

The posterior predictive distribution $$p(y | x, D)$$ is approximated through a parametrized function $$q_{\theta}(y | x, D)$$

The model $$q_{\theta}(\cdot)$$ is trained by minimiing the cross-entropy over samples drawn from the prior:

$$l_{\theta} = \mathbb{E}_{D \cup \{x,y\} \sim p(D)}\left[ - \log q_{\theta}(y |x,D) \right]$$

where $$D \cup \{x,y\}$$ simply is a synthetic dataset of size $$|D|+1$$ sampled from $$p(D)$$.

> The proposed objective $$l_{\theta}$$ is equal to the expectation of the cross-entropy between the posterior predictive distribution $$p(y | x, D)$$ and its approximation $$q_{\theta}(y | x, D)$$ : $$l_{\theta} = \mathbb{E}_{x,D \sim p(D)}\left[ H\left(p(\cdot |x,D) , q_{\theta}(\cdot |x,D) \right) \right]$$

&nbsp;

<em><b>Proof.</b></em> The above can be shown with the following derivation.

$$l_{\theta} = \textcolor{blue}{-\int_{D,x,y}p(x,y,D)} \, \log q_{\theta}(y|x,D) = \textcolor{blue}{-\int_{D,x}p(x,D) \, \int_{y} p(y|x,D)} \, \log q_{\theta}(y|x,D)$$

$$\quad = -\int_{D,x} p(x,D) \, \textcolor{blue}{H \left( p(\cdot|x,D) , q_{\theta}(\cdot|x,D) \right)} = \textcolor{blue}{\mathbb{E}_{x,D\sim p(D)}} \left[ H \left( p(\cdot|x,D) , q_{\theta}(\cdot|x,D) \right) \right]$$

&nbsp;

<em><b>Corollary.</b></em> The loss $$l_{\theta}$$ equals the expected KL-Divergence $$\mathbb{E}_{D,x}\left[ KL\left( p(\cdot|x,D) , q_{\theta}(\cdot|x,D) \right) \right]$$ between $$p(\cdot|x,D)$$ and $$q_{\theta}(\cdot|x,D)$$ over prior data $$x, D$$, up to an additive constant

### tabPFN application

