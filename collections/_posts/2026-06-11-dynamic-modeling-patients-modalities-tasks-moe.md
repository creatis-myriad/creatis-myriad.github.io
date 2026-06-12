---
layout: review
title: "Dynamic Modeling of Patients, Modalities and Tasks via Multi-modal Multi-task Mixture of Experts"
tags: deep-learning multimodal-learning multi-task-learning mixture-of-experts medical-imaging
author: "Mingtian Liu"
cite:
    authors: "C. Wu, Z. Shuai, Z. Tang, L. Wang, L. Shen"
    title:   "Dynamic Modeling of Patients, Modalities and Tasks via Multi-modal Multi-task Mixture of Experts"
    venue:   "ICLR 2025"
pdf: "https://openreview.net/pdf?id=NJxCpMt0sf"
---

# Highlights

* **M<sup>4</sup>oE** is a multi-modal, multi-task Mixture-of-Experts framework for medical tasks.
* The paper focuses on two realistic clinical problems that are often simplified away: **patient-dependent modality fusion** and **task-dependent modality selection**.
* The model decomposes information into **modality-specific** and **modality-shared task-dependent** components through two modules: MSoE and MToE.

* Experiments on mammography and retinal datasets show consistent gains over medical baselines, general multi-modal multi-task baselines, and ablated variants.

# Motivation

Multi-modal multi-task learning is a natural fit for medical imaging. A clinician rarely makes a decision from a single image or a single target. In mammography, radiologists may combine full-field digital mammography (FFDM) and synthesized 2D images (2DS), and they may reason about breast density, BI-RADS, and future cancer risk together. In ophthalmology, color fundus photos and OCT provide complementary views for glaucoma grading and optic cup segmentation.

The key point of this paper is that the relationship between modalities and tasks is not static across patients.
<div style="text-align:center">
<img src="/collections/images/m4oe/Background intro.png" width=900></div>
<p style="text-align: center;font-style:italic">Figure 1. Traditional multi-modal multi-task modeling uses a fixed shared representation, while M<sup>4</sup>oE creates dynamic links among patients, modalities, experts, and tasks.</p>

The authors identify two challenges.

**Dynamic modality fusion.** The useful information carried by each modality can vary across patients. For example, 2DS can reduce tissue overlap and reveal structures more clearly in dense breasts, while FFDM can be more informative for small calcifications in older patients.

**Modality-task dependence.** Different tasks may require different modality combinations. Breast density may rely mostly on FFDM, while cancer risk can benefit from a broader interaction between FFDM and 2DS.

This is the main clinical intuition behind the paper: effective multi-modal fusion should be both **sample-dynamic** and **task-dependent**.

# Problem Formulation

The setting contains $$m$$ image modalities $$M_1,\ldots,M_m$$ and $$p$$ related tasks $$t_1,\ldots,t_p$$. For each patient, the model receives paired multi-modal images and predicts labels for one or several tasks.

The paper makes two hypotheses:

* **Dynamic modality-specific and modality-shared information:** each modality contains unique information, but the shared information between modalities changes across patients.
* **Dynamic modality-task dependence:** the shared information that matters should also depend on the target task.

This is where MoE is useful. A Mixture-of-Experts layer creates input-dependent computation paths. Instead of forcing all samples through the same fusion function, the model can route different patient samples, modalities, and tasks through different expert combinations.

# Method: M<sup>4</sup>oE

M<sup>4</sup>oE contains three main stages:

1. Each modality is projected into a token embedding by a modality-specific feature embedder.
2. A **Modality-Specific MoE (MSoE)** extracts modality-specific information.
3. A **Modality-shared Modality-Task MoE (MToE)** models shared multi-modal information and its dependence on tasks.

The output of MSoE and MToE is then fused by a basic Soft-MoE fusion block and passed to task-specific heads.

<div style="text-align:center">
<img src="/collections/images/m4oe/Model.png" width=900></div>
<p style="text-align: center;font-style:italic">Figure 2. M<sup>4</sup>oE consists of modality-specific MoEs, a modality-task MoE, and a final Soft-MoE fusion block before the task heads.</p>

## Soft MoE

The building block is Soft MoE. Unlike sparse MoE, which selects a small number of experts through hard or top-k routing, Soft MoE uses soft assignment. Input tokens are dispatched to experts with a dispatch matrix and then recombined with a combine matrix.

This is well suited for medical imaging because modality relevance is rarely binary. A modality can be weakly useful for one sample and strongly useful for another.

## MSoE: Retaining Modality-Specific Information

The first module, **MSoE**, is applied separately to each modality. Its role is to retain information that might be lost during joint training.

This matters because multi-modal models can suffer from **modality competition**. If one modality is easier to optimize or more predictive on average, the shared representation may become dominated by that modality. Other modalities are then underused, even when they contain useful patient-specific signals.

MSoE addresses this by giving each modality its own MoE path before the final fusion stage. It encourages the model to preserve distinct modality information instead of immediately compressing everything into a single shared latent space.

## MToE: Modeling Modality-Task Dependence

The core contribution is **MToE**, the Modality-shared Modality-Task MoE.

MToE receives tokens from all modalities and routes them into experts with task-specific slots. Each expert processes slots associated with different tasks, and the model uses learnable task embeddings to distinguish the target tasks. In other words, experts are the link between:

* modality tokens,
* patient-dependent routing,
* and task-specific feature extraction.

This module is different from a generic multi-modal MoE because the routing tensor explicitly includes the task dimension. The routing weights can therefore be interpreted as a learned relation among modalities, experts, and tasks.

## Conditional Mutual Information Regularization

The authors define a probability model over modality $$M$$, expert $$E$$, and task $$T$$ from the MToE routing weights. The goal is to make expert selection depend on both modality and task.

They maximize the conditional mutual information:

$$
I(M;E \mid T)
$$

The full loss is:

$$
\mathcal{L} = \sum_{i=1}^{p}\mathcal{L}_{t_i} - \alpha I(M;E \mid T)
$$

This term has two effects:

* It encourages stronger dependence between modalities and experts under each task.
* It discourages the model from repeatedly relying on only a small subset of modalities.

Conceptually, this is important. The objective does not only ask the model to be accurate; it also asks the routing structure to learn diverse task-dependent modality patterns.

# Experiments

The model is evaluated on four public multi-modal medical imaging datasets.

**Mammography screening**

* **EMBED:** four modalities, including two FFDM views and two 2DS views; seven tasks in the full setting, with risk, density, and BI-RADS reported in the main table.
* **RSNA:** two FFDM modalities; density and BI-RADS tasks.
* **VinDR:** two FFDM modalities; density and BI-RADS tasks.

**Ophthalmology screening**

* **GAMMA:** color fundus photos and OCT; glaucoma classification and optic cup segmentation.

The main metrics are accuracy for classification tasks and Dice score for segmentation. The reported M<sup>4</sup>oE configuration uses 128 experts in MToE and 32 experts in each MSoE, trained with Adam for 100 epochs.

<div style="text-align:center">
<img src="/collections/images/m4oe/MO4E performance table.png" width=900></div>
<p style="text-align: center;font-style:italic">Table 1. M<sup>4</sup>oE consistently improves over single-task baselines, multi-task baselines, and ablated variants across mammography and retinal benchmarks.</p>

The main empirical conclusions are:

* M<sup>4</sup>oE outperforms medical single-task baselines such as Mirai, AsymMirai, EyeMost, and EyeStar.
* In the multi-task setting, M<sup>4</sup>oE outperforms general multi-modal multi-task methods such as EVIF, FULLER, AIDE, and MModN.
* Adding MToE to existing medical backbones improves them, suggesting that MToE is not tied to a single architecture.
* The full model performs better than variants without MSoE or without MToE, confirming that both modality-specific preservation and modality-task routing matter.

# Analysis

## Modality Competition

The paper first studies whether standard fusion really uses all modalities. On the EMBED cancer risk task, the authors compare modality-specific encoders trained inside a multi-modal model against unimodal models trained on each modality alone.

The result shows a clear performance drop for several modalities in the joint model. This suggests that the multi-modal model has not optimized all modality paths equally; some modalities are suppressed.

<div style="text-align:center">
<img src="/collections/images/m4oe/Visulization1.png" width=900></div>
<p style="text-align: center;font-style:italic">Figure 3. Plain multi-modal fusion and plain Soft MoE show modality competition, while M<sup>4</sup>oE yields more balanced modality utilization.</p>

The comparison between a plain Soft MoE and M<sup>4</sup>oE is especially informative. A generic MoE already provides dynamic routing, but it can still concentrate too much expert capacity on one modality. M<sup>4</sup>oE, through MSoE and MToE, distributes modality utilization more evenly.

## Modality-Task Dependence

The authors also analyze pairwise modality synergy using partial information decomposition. The purpose is to measure how much useful information emerges only when two modalities are combined.

The baseline model shows similar synergy patterns across tasks, mostly emphasizing the FFDM pair. M<sup>4</sup>oE shows different patterns:

* Density prediction is dominated by FFDM-related synergy.
* Cancer risk prediction uses more diverse interactions, including FFDM-2DS combinations.

This supports the central claim that modality fusion should be task-dependent, not just patient-dependent.

## Interpretability from Routing

Because MToE defines routing probabilities between modalities, experts, and tasks, the model can estimate modality contribution at two levels:

* **Sample level:** for a given patient and a given task, which modalities contributed most?
* **Population level:** averaged across the dataset, which modalities are globally important for each task?

<div style="text-align:center">
<img src="/collections/images/m4oe/visulization 2.png" width=900></div>
<p style="text-align: center;font-style:italic">Figure 5. M<sup>4</sup>oE provides sample-level and population-level modality contribution estimates.</p>


# Ablation Studies

The ablations show that all three components matter:

* Removing **MSoE** weakens performance because modality-specific information is less protected.
* Removing **MToE** weakens performance because the model loses explicit modality-task routing.
* Removing the **conditional mutual information regularizer** also hurts performance, though less severely than removing the structural modules.

The authors also vary the number of experts and the regularization weight $$\alpha$$. Increasing the number of experts from 16 to 128 improves performance, while increasing $$\alpha$$ helps up to a point. Too much regularization can slightly degrade prediction, which is expected because the auxiliary objective can start competing with task losses.

# Why This Paper Is Interesting

The paper is interesting because it does not treat "multi-modal" and "multi-task" as two independent extensions. It argues that they interact: the usefulness of a modality depends on the patient, and the usefulness of a modality combination depends on the task.

This is a more realistic formulation for medical AI than fixed fusion. In clinical workflows, the same set of scans can support multiple decisions, but the evidence used for each decision is not identical.

Another strong point is that the paper connects architecture and analysis. The routing weights are not only a black-box implementation detail; they are used to measure modality contribution and to regularize task-dependent modality-expert structure.

# Limitations

The model is more complex than standard fusion models. Using 128 experts in MToE and 32 experts per MSoE increases memory and computation.

The interpretability is based on routing weights. These weights can indicate how the model uses information, but they should not be overinterpreted as causal clinical importance.


The experiments are broad but still concentrated on mammography and ophthalmology. It remains to be shown how well M<sup>4</sup>oE behaves in domains with stronger missing-modality patterns, longitudinal data, or heterogeneous clinical variables.


# Conclusion

M<sup>4</sup>oE is not just another MoE architecture. Its main contribution is to frame medical multi-modal multi-task learning as a dynamic three-way relation between **patients, modalities, and tasks**.

Instead of learning one shared representation for all modalities and all tasks, M<sup>4</sup>oE learns patient-dependent modality-specific features, task-dependent shared features, and interpretable routing patterns that describe how modalities contribute to each clinical task.
