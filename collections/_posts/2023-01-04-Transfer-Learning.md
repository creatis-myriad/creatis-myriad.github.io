---
layout: review
title: "What is being transferred in transfer learning?"
tags: pre-training transfer-learning large-scale
author: "Julia Puig"
cite:
    authors: "Behnam Neyshabur, Hanie Sedghi and Chiyuan Zhang"
    title:   "What is being transferred in transfer learning?"
    venue:   "NeurIPS 2020"
pdf: "https://arxiv.org/abs/2111.14248"
---

<br/>

# Context

In cases where we lack of a large amount of data or we want a fast training, we are interested in pre-trained networks that can generalize well from a source domain to a target domain.

**Transfer learning** consists in training a model on source data and then fine-tune it in target data, and has encountered a big success in several applications. Still, it is not clear:

* What is needed for a successful transfer leaning?
* Which parts of the network make transfer learning work?

In this paper, the authors try to answer to these questions. Their code is available [here](https://github.com/google-research/understanding-transfer-learning).

<br/>

# Experimental Setup
* Data:
	* Source data for pre-training: ImageNet.
	* Target data for downstream tasks: CheXpert (X-ray) and DomainNet (real, clipart and quickdraw).


* Models:
	* Architecture: ResNet-50.
	* **P-T**: pre-trained model then fine-tuned on the target domain.
	* **RI-T**: random initialization model trained on the target domain.

<br/>

# Experiments
#### 1. Role of feature reuse in transfer learning
**Intuition**: reusing the pre-trained feature hierarchy explains the success of transfer learning. Therefore, we expect better performances when doing transfer learning on target domains that share similar visual features with the source domain. 

![](/collections/images/transfer/transfer_learning_fig2.jpg)

**However**, even distant target domains present enhanced performances with transfer learning. Are some other factors other than feature reuse being useful for the target domain?

![](/collections/images/transfer/transfer_learning_fig1.jpg)

![](/collections/images/transfer/transfer_learning_fig3.jpg)

**Conclusion**:

* Feature reuse is paramount in transfer learning.
* Low-level statistics also boost the performance of transfer learning and in particular training speed.
* P-T models converge much faster than RI-T ones.

<br/>

#### 2. What are the differences between P-T and RI-T models?
By looking at the mistakes made by each model:
 
* Samples for which P-T is incorrect and RI-T is correct are mostly hard samples.
* Samples for which P-T is correct and RI-T is incorrect are mostly easy samples.
* 2 P-T models make very similar mistakes, whereas 2 RI-T models make more diverse mistakes.

How similar are two P-T models in the feature space?

Center Kernel Alignment is used as the similarity index (Kornblith at al (2019) [[1]](https://arxiv.org/abs/1905.00414)).

![](/collections/images/transfer/transfer_learning_tab1.jpg)

What is the distance of the models in the parameter space?

![](/collections/images/transfer/transfer_learning_tab2.jpg)

**Conclusion**:

* The model initialization has a big impact in feature similarity.
* P-T models are reusing the same features and are close to each other compared to RI-T models.

<br/>

#### 3. Generalization performance of P-T and RI-T models
A criterion for good generalization of a model is whether the basin of the loss landscape near the final solution is flat (Garipov et al (2018) [[2]](https://arxiv.org/abs/1802.10026)). When a basin is narrow, a **performance barrier** occurs in points near the minimizer. 

![](/collections/images/transfer/transfer_learning_fig4.jpg)

**Conclusion**:

* There is no performance barrier between P-T models: they are in the same basin of the loss landscape.
* RI-T models present a performance barrier, even with equal random initializations: they are in different basins of the loss landscape.
* P-T models have better generalization capacities.

<br/>

#### 4. Where is feature reuse happening?

Zhang et al (2019) [[3]](https://arxiv.org/abs/1902.01996) have seen experimentally that some layers of a network are less robust to perturbation than others, and are therefore more **critical**. Module criticality can be measured by rewinding the module values back to their initial value while keeping the other trained values fixed, and test the model performance.

![](/collections/images/transfer/transfer_learning_fig5.jpg)

![](/collections/images/transfer/transfer_learning_fig6.jpg)

**Conclusion**:

* Deeper layers are more critical than lower layers.
* Lower layers are responsible for general features.
* Deeper layers are responsible for more target task-specific features.

<br/>

#### 5. Which pre-trained checkpoint model to use? 

![](/collections/images/transfer/transfer_learning_fig7.jpg)

**Conclusion**:

* We can use earlier checkpoints corresponding to the end of a plateau in the source data instead of the last checkpoint. 
* In Abnar et al (2022) [[4]](https://arxiv.org/abs/2110.02095), the same authors see experimentally that: "as we increase the upstream accuracy, performance of downstream tasks saturate".

<br/>

# Conclusions
- The authors use a wide range of tools for the analysis and understanding of models.
- Both image features and low-level statistics play a role in the good performance of transfer learning on target tasks.
- Pre-trained models are close to each other and have better robustness and generalization capacities than randomly initiliazed models.
- The checkpoint model with best performance on source data may not give the best performance on target data.























