---
layout: review
title: "Dynamic Modeling of Patients, Modalities and Tasks via Multi-modal Multi-task Mixture of Experts"
tags: deep-learning multimodal-learning multi-task-learning mixture-of-experts medical-imaging
cite:
    authors: "C. Wu, Z. Shuai, Z. Tang, L. Wang, L. Shen"
    title:   "Dynamic Modeling of Patients, Modalities and Tasks via Multi-modal Multi-task Mixture of Experts"
    venue:   "ICLR 2025"
pdf: "https://openreview.net/pdf?id=NJxCpMt0sf"
---

# Dynamic Modeling of Patients, Modalities and Tasks via Multi-modal Multi-task Mixture of Experts

## Introduction

This paper addresses multimodal multi-task learning in medical imaging. In real clinical scenarios, diagnosis is rarely based on a single image modality or a single prediction task. Clinicians often combine heterogeneous information sources and make several related decisions simultaneously.

The key motivation of this work is that the importance of each modality may vary across patients and clinical tasks. Therefore, a fixed fusion strategy may be insufficient for medical multimodal learning.

## Motivation

Traditional multimodal fusion methods often assume that each modality contributes in a relatively fixed way. However, in medical imaging, different patients may benefit from different modality combinations.

For example, in mammography, full-field digital mammography (FFDM) and synthesized 2D images (2DS) may provide complementary information depending on patient characteristics and clinical tasks. Similarly, in stroke prediction, DWI, ADC, FLAIR, perfusion imaging, and clinical variables may contribute differently depending on the patient profile, lesion characteristics, and time window.

The paper focuses on two main challenges:

1. **Dynamic modality fusion**: different patients may require different modality combinations.
2. **Modality-task dependence**: different clinical tasks may rely on different modality interactions.

## Method

The authors propose **M4oE**, a Multi-modal Multi-task Mixture-of-Experts framework for medical imaging.

The model contains two main components:

### Modality-Specific MoE

The Modality-Specific MoE, or MSoE, is designed to preserve modality-specific information. This is important because standard multimodal fusion may suffer from modality competition, where one dominant modality suppresses information from other modalities.

### Modality-Task MoE

The Modality-Task MoE, or MToE, is designed to model task-dependent multimodal fusion. It connects modalities, experts, and tasks through a routing mechanism, allowing the model to learn which modality patterns are useful for each task.

The authors also introduce a conditional mutual information regularization term to encourage experts to learn diverse task-dependent modality patterns.

## Experiments

The method is evaluated on four public medical imaging datasets, covering mammography screening and retinal disease diagnosis.

The tasks include breast cancer risk prediction, breast density classification, BI-RADS assessment, glaucoma classification, and optic cup segmentation.

The results show that M4oE outperforms several multimodal and multi-task baselines across different datasets and tasks.

## Analysis

The paper provides several analyses to support the proposed method.

First, the authors analyze modality competition and show that M4oE can improve modality utilization compared with standard multimodal fusion.

Second, they study modality-task dependence using pairwise modality synergy. This analysis shows that different tasks rely on different modality interactions. For example, density prediction mainly depends on FFDM-related synergy, while cancer risk prediction benefits from more diverse cross-modality interactions.

Third, the authors show that the routing mechanism can provide sample-level and population-level modality contribution analysis.

Finally, they analyze gradient conflicts between tasks and suggest that M4oE can help reduce such conflicts through expert specialization.

## Strengths

The main strength of this paper is that it addresses a realistic problem in medical multimodal learning: different patients and different tasks may require different modality fusion strategies.

The proposed framework is also interpretable at the model level, since the routing weights can be used to analyze modality contributions.

Another strength is that the method is evaluated on multiple medical imaging datasets and tasks, suggesting good generalization potential.

## Limitations

The method is relatively complex and uses many experts, which may increase computational cost.

The interpretability mainly comes from routing weights and should not be interpreted as causal clinical importance.

The pairwise synergy analysis is useful and intuitive, but it may not fully capture higher-order interactions among three or more modalities.

Finally, although the method is tested on mammography and retinal imaging datasets, its applicability to other domains, such as stroke MRI, remains to be further validated.

## Discussion

This paper is relevant to patient-specific multimodal fusion in stroke prediction. In stroke outcome prediction or final lesion prediction, different modalities may contribute differently depending on time window, baseline lesion, perfusion status, and clinical severity.

A similar MoE-based fusion strategy could potentially be used to dynamically weight DWI, ADC, FLAIR, perfusion, and clinical variables for each patient.

## Take-home message

M4oE does not simply fuse all modalities with a fixed strategy. Instead, it learns patient-specific and task-specific modality fusion through a mixture-of-experts framework.
