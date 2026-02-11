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
  - [Results](#results)
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

The likelihood $$p(x \mid z$$) models the behavior of the test. Let's assume a sensitivity of $$95\%$$ and a specificity of $$90\%$$, thus
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

Denominator: $$p(x=+)$$ = $$0.95 \times 0.01 + 0.10 \times 0.99 = 0.1085$$

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

<div style="text-align:center">
<img src="/collections/images/bayesian-inference/illustration-in-context-learning.jpg" width=500></div>

### Modeling

#### Modeling relationships in the data

The prior defines a space of hypotheses $$\Phi$$ on the relationship of a set of inputs $$X$$ to the output labels $$y$$. Each hypothesis $$\phi \in \Phi$$ can be seen as a mechanism that generates a data distribution from which we can draw samples forming a dataset.

> A prior is not just a law governing parameters: it is a distribution that defines the fundamental relationships between input and output data, allowing inductive bias to be introduced.

<br>

> Inductive bias is the set of structural assumptions that a model imposes on the space of plausible solutions before observing any data. In Bayesian inference, it is formally encoded by the prior, and it determines how the model generalizes from a finite number of observations 

<!--
#### Problem formulation

Using the law of total probability, the posterior predictive distribution can be rewritten as:

$$p(y | x, D) = \int_{\phi}p(y | x, \phi) \, p(\phi | D) \, d\phi$$

$$p(y | x, D) \propto \int_{\phi}p(y | x, \phi) \, p(D | \phi) \, p(\phi) \, d\phi$$ 

-->

#### Prior sampling scheme

Based on the hypothesis $$\Phi$$, one can implement an efficient prior sampling scheme of the form: 

$$p(D) = \int_{\phi} p(D | \phi) \, p(\phi) \, d\phi$$ 

The generative mechanism is first sampled as $$\phi \sim p(\phi)$$ which encodes the relationships between $$X$$ and $$y$$. The synthetic dataset is then sampled as $$D\sim p(D | \phi)$$. 

#### Learning process

The posterior predictive distribution $$p(y | x, D)$$ is approximated through a parametrized function $$q_{\theta}(y | x, D)$$

The model $$q_{\theta}(\cdot)$$ is trained by minimiing the cross-entropy over samples drawn from the prior:

$$l_{\theta} = \mathbb{E}_{D \cup \{x,y\} \sim p(D)}\left[ - \log q_{\theta}(y |x,D) \right]$$

where $$D \cup \{x,y\}$$ simply is a synthetic dataset of size $$|D|+1$$ sampled from $$p(D)$$.

> The proposed objective $$l_{\theta}$$ is equal to the expectation of the cross-entropy between the posterior predictive distribution $$p(y | x, D)$$ and its approximation $$q_{\theta}(y | x, D)$$ : $$l_{\theta} = \mathbb{E}_{x,D \sim p(D)}\left[ H\left(p(\cdot |x,D) , q_{\theta}(\cdot |x,D) \right) \right]$$

<br>

<em><b>Proof.</b></em> The above can be shown with the following derivation.

$$l_{\theta} = \textcolor{blue}{-\int_{D,x,y}p(x,y,D)} \, \log q_{\theta}(y|x,D) = \textcolor{blue}{-\int_{D,x}p(x,D) \, \int_{y} p(y|x,D)} \, \log q_{\theta}(y|x,D)$$

$$\quad = -\int_{D,x} p(x,D) \, \textcolor{blue}{H \left( p(\cdot|x,D) , q_{\theta}(\cdot|x,D) \right)} = \textcolor{blue}{\mathbb{E}_{x,D\sim p(D)}} \left[ H \left( p(\cdot|x,D) , q_{\theta}(\cdot|x,D) \right) \right]$$

<br>

<em><b>Corollary.</b></em> The loss $$l_{\theta}$$ equals the expected KL-Divergence $$\mathbb{E}_{D,x}\left[ KL\left( p(\cdot|x,D) , q_{\theta}(\cdot|x,D) \right) \right]$$ between $$p(\cdot|x,D)$$ and $$q_{\theta}(\cdot|x,D)$$ over prior data $$x, D$$, up to an additive constant.

$$\begin{aligned}
& \mathbb{E}_{x,D}\left[ KL \left( p(. | x,D), q_{\theta}(. | x,D) \right) \right] \\
&= - \mathbb{E}_{x,D}\left[ \int_y p(y | x,D) \, \log \frac{q_{\theta}(y | x,D)}{p(y | x,D)} \right] \\
&= - \mathbb{E}_{x,D}\left[ \int_y p(y | x,D) \, \log q_{\theta}(y | x,D) \right] + \mathbb{E}_{x,D}\left[ \int_y p(y | x,D) \, \log p(y | x,D) \right] \\
&= \mathbb{E}_{x,D}\left[ H \left( p(. | x,D), q_{\theta}(. | x,D) \right) \right] - \mathbb{E}_{x,D}\left[ H \left( p(. | x,D) \right) \right] \\
&= l_{\theta} + C
\end{aligned}$$

where $$C$$ is a constant that does not depend on $$\theta$$.

### tabPFN application

#### Prior modeling through Structural Causal Models (SCMs)

Tabular data can be seen as the result of several simple mechanisms interacting with each other.
- a table row corresponds to a real-world entity (patient, customer, transaction, etc.)
- each column corresponds to a measurement, decision, or attribute produced by a real process
- the label represents a consequence (diagnosis, defect, class, etc.)

> Tabular data result from chains of decisions, mechanisms, and constraints. Even if the exact causal structure is unknown, tabular data are almost always causal in essence

Structural Causal Models (SCMs) are thus used as the prior to model the implicit structure of tabular data. 

SCMs impose a “reasonable” structure without being rigid. 

They model:
- nonlinear dependencies
- interactions
- noise
- different graphs (i.e. relationships) across datasets

They allows:
- local, compositional, and parsimonious structures
- plausible dependencies between columns
- preference for simple relationships

#### Synthetic dataset generation

To generate a synthetic dataset, TabPFN essentially follows the following pipeline:
- Sample a causal structure (a DAG)
- Sample causal functions along each edge
- Sample noise terms
- Generate the features
- Generate the label
- Apply realistic transformations
- Sample a small dataset (few-shot regime)

Each dataset corresponds to a task for which TabPFN learns to perform Bayesian inference.

##### 1- Sampling the causal structure (DAG)

- Number of variables: randomly sampled within a range (e.g., 5 to 100)
- Graph structure
  - sparsity is encouraged
  - a small number of parents per node
  - a random topological ordering
  
> The intuition beind this sampling scheme is that real-world tabular variables rarely exhibit global dependencies across all columns

##### 2- Sampling the causal mechanisms (structural Functions)

The following relation is defined for each variable $$X_i$$ with parents $$Pa(X_i)$$:
$$X_i = f_i \left( Pa(X_i) \right) + \epsilon_i$$

$$f_i$$ is randomly chosen from a mixture of function families:
- linear functions
- simple nonlinear functions
- small neural networks
- sometimes tree- or threshold-based function

But with:
- low depth
- low complexity
- simple activations

##### 3- Sampling the noise

Each variable has its own noise term $$\epsilon_i \sim N(0,\sigma_i^2)$$. 
The variance $$\sigma$$ is sampled randomly.

##### 4- Feature generation (propagation through the DAG)
Once we have:
- the graph
- the structural functions
- the noise terms,

data are generated according to the causal ordering:
- variables without parents $$\rightarrow$$ sampled directly
- intermediate variables $$\rightarrow$$ computed via $$𝑓_i$$
- deeper variables $$\rightarrow$$ accumulate dependencies and noise

A set of features are then randomly selected from the graph

##### 5- Label generation $$y$$

The label is treated as a final causal variable:
$$y = g \left( Pa(y) \right) + \epsilon_y$$

where:
- $$g$$ is sampled as a simple function
- sometimes depends on few variables
- sometimes depends indirectly on many through the DAG

For classification:
- $$g$$ produces a latent score
- passed through a sigmoid or softmax
- then the class is sampled

Here again, y is randomly selected from the graph. The figure below shows an example of SCMs sampled from the prior. The grey nodes correspond to the sampled inputs $$X$$ and output $$y$$.

<div style="text-align:center">
<img src="/collections/images/bayesian-inference/scms.jpg" width=600></div>

##### 6- "Realistic” Transformations

Before feeding the dataset to the model, TabPFN applies:
- random normalization
- column permutation
- different scalings per feature
- sometimes monotonic transformations
- introduction of class imbalance

These steps prevent the model from “cheating” by recognizing the generator.

##### 7- Sampling in the Few-Shot Regime
Finally:
- a small number of samples $$𝑛$$ is drawn (often $$<100$$)
- train/test split is created
- everything is provided in-context to the transformer

#### Data preprocessing

Both the synthetic and the real datasets are represented as follows:
<div style="text-align:center">
<img src="/collections/images/bayesian-inference/data-preparation-0.jpg" width=400></div>

<br>

The categorical data are encoded as integers
<div style="text-align:center">
<img src="/collections/images/bayesian-inference/data-preparation-1-2.jpg" width=500></div>

<br>

A z-normalization across feature/column dimension is applied 
<div style="text-align:center">
<img src="/collections/images/bayesian-inference/data-preparation-3.jpg" width=500></div>


#### Tokenization procedure

After data preprocessing, each tabular feature is embedded as a token using a shared linear projection

<div style="text-align:center">
<img src="/collections/images/bayesian-inference/tokenization-1.jpg" width=700></div>

<br>

This yields the following representation at the input of the transformer

<div style="text-align:center">
<img src="/collections/images/bayesian-inference/data-preparation-1.jpg" width=700></div>

#### Transformer architecture

The following transformer architecture is proposed. It consists of 12 layers that sequentially apply attention over features and attention over samples. It should be noted that attention over samples is applied between all support samples and one query sample at a time. Query samples do not interact with one another. 

<div style="text-align:center">
<img src="/collections/images/bayesian-inference/transformer-architecture-v1.jpg" width=800></div>

#### Training procedure

During training, each batch is populated with a dataset sampled from the SCM distribution described above. The following scheme is then applied

<div style="text-align:center">
<img src="/collections/images/bayesian-inference/tabpfn-training.jpg" width=700></div>

#### Results
