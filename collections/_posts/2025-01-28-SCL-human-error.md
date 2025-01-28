---
layout: review
title: Understanding and Mitigating Human-Labelling Errors in Supervised Contrastive Learning
tags: deep-learning supervised-learning contrastive-learning label error
cite:
    authors: "Z. Long, L. Zhuang, G. Killick, R. McCreadie, G. Aragon-Camarasa, P. Henderson"
    title:   "Understanding and Mitigating Human-Labelling Errors in Supervised Contrastive Learning"
    venue:   "European Conference on Computer Vision (ECCV). pp. 435-454. Springer, Cham, 2025."
pdf: "https://arxiv.org/pdf/2403.06289"
---

# Highlights
* Insight into human-labelling errors vs. synthetic-labelling errors ;
* Quantification of human-labelling errors in supervised contrastive learning framework ;
* Proposition of SCL-RHE, a refined contrastive objective to improve robustness to human-labelling errors with computational efficiency ;
* Improved performance for pre-training and fine-tuning errors on vision datasets for top-1 classification ;
* Demonstrate some robustness to synthetic-labelling errors.

# Contrastive learning

Contrastive learning is a learning strategy within the **representation learning** framework; it aims to represent data in a lower-dimensional space, known as the latent space. 
It is also part of **metric learning**: in this space, the distances between objects are meaningful in relation to a similarity criterion.
In contrastive learning, this similarity criterion is learned by bringing some samples closer together while pushing others apart. A sampling policy is chosen, which assigns to each sample (the anchor) a set of samples that should be closer (the **positive** set) and a set that should be pulled apart (the **negative** set).


# Gap in the literature
blabla
