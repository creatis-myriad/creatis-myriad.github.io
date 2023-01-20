---
layout: review
title: SHAP values
tags: explainability machine-learning black-box post-hoc
author: Pierre-Elliott Thiboud
cite:
    authors: "Scott M. Lundberg, Su-In Lee"
    title:   "A Unified Approach to Interpreting Model Predictions"
    venue:   "Advances in Neural Information Processing Systems 30 (NeurIPS 2017)"
pdf: "http://papers.nips.cc/paper/by-source-2017-2493"
---

# Highlights

- 
- Unification of feature attribution methods under same paradigm
- Optimisation of Shapley values computations

# Introduction

Currently used Machine-learning and Deep-learning models are increasingly powerful and complex. However, for legal, ethical or technical reasons, there is a growing need to understand how their predictions are made. To this end, feature attribution models try to estimate how inputs and outputs are correlated. The authors propose a framework unifying existing explanation models with improved computational performances and better theoretical foundations.

> Need more than a simple introduction  
> Maybe: recap of the needs, quick overview of explanation models (just their name and the idea behind them)

# Additive feature attribution models

## Defining explanations

Viewing explanations of a model's prediction as a model itself allow oneself to design this _explanation model_ to exhibit whatever properties are wanted. In this case, where we want explanations to be accurate and easily interpretable, additive feature attribution models have many advantages.

The goal of those models is to _attribute_ an effect to all considered _features_ of the input while following the additive property: summing the effects for all features approximates the output of the original model. Thus, explanations of a model output can be reformulated as a linear function of binary variables:

$$ g(z') = \phi_0 + \sum_{i=1}^{M} \phi_i z'_i $$

where $$z'$$ is a simplified input, $$z' \in \{0, 1\}^M$$, _M_ is the number of simplified input features, and $$\phi_i \in \mathbb{R}$$.

> Need to define $$x=h_x(x')$$  
> Also need to make clear distinction between $$\,x$$, $$x'$$ and $$z'$$ (they switch from one to the other in the paper)

## Desirable properties

No matter the chosen class of models, some properties are desirable for our explanations:

**Property 1 - Local Accuracy**  
The explanation should accurately represent the prediction:

$$f(x)=g(x')=\phi_0 + \sum_{i=1}^{M} \phi_i x'_i$$

where $$\phi_0 = f(h_x(\mathbf{0}))$$ represents the model with all simplified inputs missing.

**Property 2 - Missingness**  
If a feature is missing, no effect should be attributed to it:

$$x'_i=0 \implies \phi_i=0$$

**Property 3 - Consistency**  
Feature's effect should be correlated to "real" effect:

Let $$f_x(z') = f(h_x(z'))$$ and $$z' \setminus i$$ denotes setting $$z'_i=0$$. For any two models $$f$$ and $$f'$$, if (for all inputs $$z' \in \{0, 1\}^M$$)

$$f'_x(z') - f'_x(z'\setminus i) \geq f_x(z') - f_x(z'\setminus i)$$

Then,

$$\phi_i(f', x) \geq \phi_i(f, x)$$

## Defined model

With those properties and the desired additive feature attribution class of models, there exists only one possible explanation model defined as follow:

$$\phi_i(f,x) = \sum_{z' \subseteq x'} \frac{|z'|! (M - |z'| - 1)!}{M!} \left(f_x(z') - f_x(z' \setminus i)\right)$$

where \|$$z'$$\| is the number of non-zero entries in $$z'$$, and $$z' \subseteq x'$$ represents all $$z'$$ vectors where the non-zero entries are a subset of the non-zero entries in $$x'$$.

[Young (1985)](https://link.springer.com/article/10.1007/BF01769885) demonstrated that Shapley values are the only set of values that satisfy three axioms similar to properties 1 and 3 (and another property proved by the authors to be redundant in this setting). So if a method is not based on Shapley values, it violates local accuracy and/or consistency.

> Add details about the proof?

# Shapley values

## Properties

> Start from linear model example to go to cooperative game theory?

To explain how features gave a specific prediction, we can consider each feature as a player in a cooperative game. Shapley (1952)[^1] formalized a method to fairly distribute the "payout" accross each player whom contributed to the game. In our case, it allows us to estimate how each feature (player) contributed to the current prediction (game).

A feature value's Shapley value is its contribution to the payout, weighted and summed over all possible feature value coalition:

$$\phi_i(val, x) = \sum_{S \subseteq \{1,...,p\} \setminus \{i\}} \frac{|S|! (p - |S| - 1)!}{p!} (val_x(S \cup \{i\}) - val_x(S))$$

> It means[^4]:
> $$\frac{|S|!(p-|S|-1)!}{p!} = \left( |S| \binom{p}{|S|} \right)^{-1}$$
> But why? How?

where $$S$$ is a subset of the features used in the model, $$x$$ is the input to be explained, $$p$$ the number of considered features and $$val_x(S)$$ is the prediction for feature values in set $$S$$ that are marginalized over features that are not included in set $$S$$.

> Probably missing explanation or don't go into this much details

Shapley values satisfy 4 properties:

**Efficiency** Feature contributions sum equals the difference between prediction for $$x$$ and the average prediction:

$$\sum_{i=1}^p \phi_i = f(x) - E_X(f(X))$$

**Symmetry** If two feature values contribute equally to all possible coalitions, their contributions should be the same:

$$\forall S \subseteq \{1,...,p\} \setminus \{i,j\}, \qquad val(S \cup \{i\}) = val(S \cup \{j\}) \implies \phi_i = \phi_j$$

**Missingness** A feature $$i$$ which doesn't change the predicted value (no matter the coalition) should have a Shapley value of 0:

$$\forall S \subseteq \{1,...,p\} \setminus \{i\}, \qquad val(S \cup \{i\}) = val(S) \implies \phi_i = 0$$

**Additivity** For a game with combined payouts ($$val_x + val_{x'}$$), the Shapley value $$i$$ is the sum of the respective Shapley values:

$$\phi_i(val_x + val_{x'}) = \phi_i(val_x) + \phi_i(val_{x'})$$

## Computation

Still Shapley values have a problem: their computation. It requires to calculate each possible coalition for each feature $$i$$, so $$2^N$$ with $$N$$ features. More than coalitions, how can we define $$z \setminus i$$, an input where feature $$i$$ is missing? How do we account for possible dependencies between features?

To resolve this, [Strumbelj et al. (2014)](https://link.springer.com/article/10.1007/s10115-013-0679-x) proposed the _Shapley sampling_, an approximation with Monte-Carlo sampling:

$$\hat{\phi}_i = \frac{1}{M} \sum_{m=1}^M \left( f(x_{+i}^m) - f(x_{-i}^m) \right)$$

where $$f(x_{+i}^m)$$ is the prediction for $$x$$ but with a random number of feature values replaced by feature values from a random data point $$z$$, except for feature $$i$$. In contrast to $$x_{+i}^m$$, in $$x_{-i}^m$$, the feature $$i$$ is also taken from the sampled $$z$$. This creates some sort of "Frankenstein monster" instance assembled from two instances which, with enough iteration $$M$$, approximates the removal of a single feature value while keeping _some_ feature interactions.

This result in the following algorithm to compute the Shapley value for a single feature $$i$$:

1. For all $$m=1,...,M$$:
   1. Choose a random permutation $$o$$ of the feature values
   2. Draw a random instance $$z$$ from our dataset $$X$$
   3. Order instances with permutation $$o$$:

       $$x_o = (x_{(1)},...,x_{(i)},...,x_{(p)})$$

       $$z_o = (z_{(1)},...,z_{(i)},...,z_{(p)})$$

   4. Construct two new instances:

       $$x_{+i} = (x_{(1)},...,x_{(i-1)},x_{(i)},z_{(i+1)},...,z_{(p)})$$

       $$x_{-i} = (x_{(1)},...,x_{(i-1)},z_{(i)},z_{(i+1)},...,z_{(p)})$$

   5. Compute the marginal contribution: $$\hat{\phi}_i^m = f(x_{+i}) - f(x_{-i})$$
2. Compute the Shapley value as the average of marginal contributions:

    $$\hat{\phi}_i(x) = \frac{1}{M} \sum_{m=1}^M \hat{\phi}_i^m$$

## Note

> Move into limitations, with other disadvantages/risks

Be careful when interpreting Shapley values: they do NOT represent the difference in the prediction if removed from the model, they are the average contribution of a feature value to the prediction in different features coalitions.

# SHAP findings

- KernelSHAP (model agnostic)
- Mix Shapley values & Additive models
  - Rewrite some axioms
- Upgrade existing models to follow Shapley axioms
- Optimizations for specific models (decision trees, deep learning)

> Talk about multiclass classification and images, how is it handled

# References

[^1]: Lloyd Shapley. A value for N-person games (1952). [Link](https://www.rand.org/content/dam/rand/pubs/papers/2021/P295.pdf)  
[^2]: E. Štrumbelj and I. Kononenko. Explaining prediction models and individual predictions with feature contributions (2013). [Link](https://link.springer.com/article/10.1007/s10115-013-0679-x)  
[^3]: christophm.github.io. [Link](https://christophm.github.io/interpretable-ml-book/shapley.html)  
[^4]: Samuel Mazantti. [Link](https://towardsdatascience.com/shap-explained-the-way-i-wish-someone-explained-it-to-me-ab81cc69ef30)
<!-- [^2] H. P. Young. Monotonic solutions of cooperative games (1985). [Link](https://link.springer.com/article/10.1007/BF01769885) Include as ref ? -->
