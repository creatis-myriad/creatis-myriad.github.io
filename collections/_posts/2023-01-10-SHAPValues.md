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

- Local explainability with proven desirable properties
- Unification of feature attribution methods under same paradigm
- Optimisation of Shapley values computations for specific models

# Introduction

Currently used Machine-learning and Deep-learning models are increasingly powerful and complex. However, for legal, ethical or technical reasons, there is a growing need to understand how their predictions are made. To this end, there exists several methods approximating the model "prediction path", be it at the model scale or for individual predictions. _Global explanations_ focus on the model and its underlying decision process. Their goal is to get some sense of understanding of the mechanism by which the model works. On the other hand, _local explanations_ provide insights on why a particular predictions was made instead of another one.

Among the different local explainability methods, which include for example _Saliency maps_, _Prototype based_ or _Counterfactuals explanations_, this article focus on _Feature importance_ through Additive feature attribution models. These models estimate how inputs and outputs are correlated, and "how much" each feature of an input impact the prediction.

The authors propose a framework unifying existing _Feature importance_ based explanation model, proving their solution to be unique. With these theoritical results, the authors also proposed model-specific performance improvements through several assumptions.

# Shapley values

## Games and cooperation

Let's assume we trained a model to predict apartment prices based on few features. This model predicted a price of €300,000 for an apartment and we want to explain the prediction. This apartment is located on the 2nd floor with an area of 50m² and a park nearby while cats are banned.

![](/collections/images/shap/prediction_example.jpg)

For linear regression models, the answer is simple. The coefficients represent directly the impact of each feature and if it's a positive or negative effect.

For more complex scenarios, Shapley values[^1] come to the rescue. They are a concept coming from game theory, describing how to fairly distribute a **payout** between **players** participating in a cooperative **game**. But how does it apply to explainability in machine learning? If we have a predictive model, then we can define:

- A **game**: reproduce the prediction of the model
- The **players**: the features of the input
- The **payout**: the individual impact of each feature to the prediction

To determine the importance of a single feature, the idea is to compute the difference between a prediction with the feature and another prediction without. This is the "marginal contribution" of the feature to the prediction.

![](/collections/images/shap/marginal_contribution.jpg)

But, as a cooperative game, multiple features are involved in the prediction, with possible hidden dependencies between them. So **each possible combination (or coalition) of features should be considered**, hence computing the "average marginal contribution". Thus, to explain the prediction for our apartment, we need to compute the predicted price for all coalitions **without** the `cat banned` feature AND all coalitions **with** the `cat banned` feature.

![](/collections/images/shap/coalitions_example.jpg)

## Properties

The formula to compute the Shapley value $$\phi_i$$ for feature $$i$$ is:

$$\phi_i(f, x) = \sum_{S \subseteq \{1,...,P\} \setminus \{i\}} \frac{(|S|-1)! (P - |S|)!}{P!} (f(S \cup \{i\}) - f(S))$$

Where $$f(x)$$ is the prediction of a model $$f$$ for the input $$x$$ with $$P$$ features, and $$S$$ being a subset of the features used by the model. We can find the "marginal contribution" of the feature $$i$$ weighted by a coefficient dependant on the number of features in the coalition, summed over possible coalitions.

The role of these Shapley values being to fairly distribute a reward, they need to exhibit certain properties:

**Efficiency** Feature contributions sum equals the difference between prediction for $$x$$ and the average prediction:

$$\sum_{i=1}^p \phi_i = f(x) - E_X(f(X))$$

**Symmetry** If two feature values contribute equally to all possible coalitions, their contributions should be the same:

$$\forall S \subseteq \{1,...,p\} \setminus \{i,j\}, \qquad f(S \cup \{i\}) = f(S \cup \{j\}) \implies \phi_i = \phi_j$$

**Missingness** A feature $$i$$ which doesn't change the predicted value (no matter the coalition) should have a Shapley value of 0:

$$\forall S \subseteq \{1,...,p\} \setminus \{i\}, \qquad f(S \cup \{i\}) = f(S) \implies \phi_i = 0$$

**Additivity (or linearity)** The effect of a feature on the sum of 2 models $$f + f'$$ equals the sum of the respective Shapley values:

$$\phi_i(f + f') = \phi_i(f) + \phi_i(f')$$

# Additive feature attribution models

## Defining explanations

Explanation models often use _simplified inputs_ $$x'$$ that map to the original inputs through a mapping function $$x = h_x(x')$$. And local methods' goal is to ensure $$g(z') \approx f(h_x(z'))$$ whenever $$z' \approx x'$$. Those _simplified inputs_ usually represent the presence or absence of each features in the input, as a binary vector with $$z_i' \in \{0, 1\}$$.

So, to summarize, our goal is to create a model $$g$$ which associates an explanation $$\phi$$ to a model's output $$f(x)$$:

$$g(f(x)) = \phi(f,x)$$

The **Efficiency** property of Shapley values allows us to define an additive feature attribution model as a linear function of binary variables:

$$ g(z') = \phi_0 + \sum_{i=1}^{M} \phi_i z'_i $$

where $$z'$$ is a simplified input (thus $$z' \in \{0, 1\}^M$$), _M_ is the number of simplified input features, $$\phi_i \in \mathbb{R}$$, and $$\phi_0$$ is the average prediction ($$E_X[f(X)]$$, by definition).

## Unification of existing methods

[Young (1985)](https://link.springer.com/article/10.1007/BF01769885) demonstrated that Shapley values are the only set of values that satisfy axioms related to their _local accuracy_ (of representing the prediction of the model) and their _consistency_ between models. So if a method is not based on Shapley values, it violates local accuracy and/or consistency.

The authors then described other local explanation models such as LIME, DeepLIFT or Layer-wise Relevance Propagation (Saliency map), and showed that these methods follow the previously defined model.

For a quick overview of these methods, LIME **[REF]** interprets individual model predictions by locally approximating the model around a given prediction. DeepLIFT **[REF]** recursively attributes, to Deep Learning models, effect to features comparing each feature value to a reference value (deemed to be uninformative for this specific feature), using a linear composition rule. And Layer-wise Relevance Propagation **[REF]** is showed (by **[REF]**) to be equivalent to DeepLIFT with each reference values being $$0$$.

# SHAP values

## Problems with existing methods (LIME, ...)

> Show their graphs describing the problem (Fig 4 & 5)

![](/collections/images/shap/shap_fig4.jpg)

![](/collections/images/shap/shap_fig5.jpg)

> It seems their SHAP values are defined "to follow Shapley values", maybe that's the "problematic equation"

## Shapley sampling

Shapley values have a problem: their computation. It requires to calculate each possible coalition for each feature $$i$$, so $$2^N$$ with $$N$$ features. To resolve this, [Strumbelj et al. (2014)](https://link.springer.com/article/10.1007/s10115-013-0679-x) proposed the _Shapley sampling_, an approximation with Monte-Carlo sampling:

$$\hat{\phi}_i = \frac{1}{M} \sum_{m=1}^M \left( f(x_{+i}^m) - f(x_{-i}^m) \right)$$

where $$f(x_{+i}^m)$$ is the prediction for $$x$$ but with a random number of feature values replaced by feature values from a random data point $$z$$, except for feature $$i$$. _A contrario_, in $$x_{-i}^m$$, the feature $$i$$ is also taken from the sampled $$z$$. This creates some sort of "Frankenstein monster" instance assembled from two instances which, with enough iterations $$M$$, approximates the removal of a single feature value while keeping _some_ feature interactions.

This results in the following algorithm to compute the Shapley value for the feature $$i$$:

1. For all $$m=1,...,M$$:
   1. Choose a random permutation $$o$$ of the feature values
   2. Draw a random instance $$z$$ from our dataset $$X$$
   3. Order instances with permutation $$o$$:

       $$x_o = (x_{(1)},...,x_{(i)},...,x_{(p)})$$

       $$z_o = (z_{(1)},...,z_{(i)},...,z_{(p)})$$

   4. Construct two new instances:

       $$x_{+i} = (x_{o(1)},...,x_{o(i-1)},x_{o(i)},z_{o(i+1)},...,z_{o(p)})$$

       $$x_{-i} = (x_{o(1)},...,x_{o(i-1)},z_{o(i)},z_{o(i+1)},...,z_{o(p)})$$

   5. Compute the marginal contribution: $$\hat{\phi}_i^m = f(x_{+i}) - f(x_{-i})$$
2. Compute the Shapley value as the average of marginal contributions:

$$\hat{\phi}_i(x) = \frac{1}{M} \sum_{m=1}^M \hat{\phi}_i^m$$

> Be careful when interpreting Shapley values: they do NOT represent the difference in the prediction if removed from the model, they are the average contribution of a feature value to the prediction in different features coalitions.

## Other approximations

> Just describe rapidly what is their solution

- KernelSHAP (LIME + Shapley values)
  - Choosing adequate parameters for LIME equation allow to match Shapley values
  - [Maybe just show LIME equation to point out which parameters need to be chosen correctly]
- DeppSHAP (DeepLIFT + Shapley values)
  - SHAP values for smaller components of the networks

# Limitations

> Point some limitations?

# References

[^1]: Lloyd Shapley. A value for N-person games (1952). [Link](https://www.rand.org/content/dam/rand/pubs/papers/2021/P295.pdf)  
[^2]: E. Štrumbelj and I. Kononenko. Explaining prediction models and individual predictions with feature contributions (2013). [Link](https://link.springer.com/article/10.1007/s10115-013-0679-x)  
[^3]: christophm.github.io. [Link](https://christophm.github.io/interpretable-ml-book/shapley.html)  
[^4]: Samuel Mazantti. [Link](https://towardsdatascience.com/shap-explained-the-way-i-wish-someone-explained-it-to-me-ab81cc69ef30)
<!-- [^2] H. P. Young. Monotonic solutions of cooperative games (1985). [Link](https://link.springer.com/article/10.1007/BF01769885) Include as ref ? -->
