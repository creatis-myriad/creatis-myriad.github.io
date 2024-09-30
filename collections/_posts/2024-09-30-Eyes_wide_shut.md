---
layout: review
title: "Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs"
tags: VLM, CLIP, MLLM, contrastive-learning
author: "Gaspard Dussert"
cite:
    authors: "Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, Saining Xie"
    title: "Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs"
    venue: "CVPR 2024"
pdf: "https://openaccess.thecvf.com/content/CVPR2024/html/Tong_Eyes_Wide_Shut_Exploring_the_Visual_Shortcomings_of_Multimodal_LLMs_CVPR_2024_paper.html"
---

# Highlights


# Introduction : Is vision good enough for language? 

Multimodal Large Language Models (MLLMs) integrate images into LLMs and show remarkable capabilities in tasks such as image understanding and visual question answering.

Howver they still exhibit visual shortcomings, some of which are surprisingly elementary and evident (Figure 1). 

Where do these problems originate? Is it a deficiency in visual modality, language understanding, or their alignment?

A natural hypothesis is that any limitation in the pretrained vision models can cascade into the downstream MLLMs that use them.

<div style="text-align:center"><img src="/collections/images/EWS/EWS1.jpg" width=1500></div>
<p style="text-align: center;font-style:italic">Figure 1. MLLMs (here GPT-4V) struggle with seemingly simple questions due to inaccurate visual grounding. red is an incorrect response, green is an hallucinated explanations.</p>

# Identifying failure examples

They exploit the *erroneous agreements* in the embedding space. If two visually different images are encoded similarly by CLIP, then at least one of the images is likely ambiguously encoded. 
They call such a pair of images a *CLIP-blind* pair.

They use the corpus datasets, ImageNet and LAIONAesthetics, to collect these CLIP-blind pairs.
For each pair they compute the embeddings using CLIP-ViT-L-14 and DINOv2-ViT-L-14. 
They return pairs such that the cosine similarity exceeds 0.95 for CLIP embeddings and less than 0.6 for DINOv2 embeddings.

Using these CLIP-blind pairs they build the Multimodal Visual Patterns (MMVP) benchmark.

<div style="text-align:center"><img src="/collections/images/EWS/EWS2.jpg" width=1500></div>
<p style="text-align: center;font-style:italic">Figure 2. Constructing MMVP benchmark via CLIP-blind pairs.</p>

# Multimodal Visual Patterns (MMVP) benchmark

For each CLIP-blind pair of images, they manually pinpoint the visual details that the CLIP vision encoder overlooks and craft questions that probe these visual details, for example “Is the dog facing left or right?”.
The benchmark is made of 150 pairs with 300 questions.

Human performance is evaluated through a user study where users are presented with 300 questions in a randomized sequence. 

They consider a pair of images to be correctly answered if both the questions associated with the pair are answered accurately.

<div style="text-align:center"><img src="/collections/images/EWS/EWS3.jpg" width=1500></div>
<p style="text-align: center;font-style:italic">Figure 3. Examples of Questions in the MMVP benchmark.</p>

<div style="text-align:center"><img src="/collections/images/EWS/EWS4.jpg" width=1500></div>
<p style="text-align: center;font-style:italic">Figure 4. Benchmark results of current SOTA MLLM models and humans.</p>

<div style="text-align:center"><img src="/collections/images/EWS/EWS5.jpg" width=1500></div>
<p style="text-align: center;font-style:italic">Figure 5. </p>

<div style="text-align:center"><img src="/collections/images/EWS/EWS6.jpg" width=1500></div>
<p style="text-align: center;font-style:italic">Figure 6. </p>

<div style="text-align:center"><img src="/collections/images/EWS/EWS7.jpg" width=1500></div>
<p style="text-align: center;font-style:italic">Figure 7. </p>

<div style="text-align:center"><img src="/collections/images/EWS/EWS_T1.jpg" width=1500></div>
<p style="text-align: center;font-style:italic">Table 1. </p>

<div style="text-align:center"><img src="/collections/images/EWS/EWS_T2.jpg" width=1500></div>
<p style="text-align: center;font-style:italic">Table 2. </p>

<div style="text-align:center"><img src="/collections/images/EWS/EWS_T3.jpg" width=1500></div>
<p style="text-align: center;font-style:italic">Table 3. </p>


## Mask decoder

The decoder architecture is again quite similar to that of SAM. It takes as input the conditioned frame embeddings (with potential mask encoded embeddings) on one hand, and sparse prompts tokens, concatenated with tokens used for the output, on the other hand. The decoder stacks transformer blocks with self-attention between prompt+ouput tokens and alternated "image to token" and "token to image" cross-attention.

> N.B: X to Y attention means that X is query and Y is key and value in the attention layer.

The model produces multiple from the resulting tokens : 
* The tokens are upsampled with convolutions and summed with stages 1 and 2 frame features from Hiera image encoder. They are multiplied with the output tokens (after passing through 3 MLP layers) to produce **low-resolution masks** (at a resolution 4x lower than input image).
*  The output tokens is also used to produce what are called **object pointers**, which are lightweight vectors to encode high-level semantic information of the objecct to segment.
*  As in SAM, there could be ambiguity on the mask to predict, as an object can have several subparts that are also segmented. For this reason, the model produces multiple output masks (3 in general) and predicts the **IoU scores** between the ground truth and each mask.
*  Finally, a specific constraint of video segmentation is that objects of interest can disappear from the image, momentarily or permanently. Hence, SAM 2 also predicts an **oclusion score** for each object.

From these outputs, few post-processing steps, including resolving mask covering (based on the predicted IoU), mask refinement and a final 4x upsampling, allow to produce the final segmentation map at the original image size.

# Reference

[^1]: C. Ryali, Y.-T. Hu, D. Bolya, C. Wei, H. Fan, P.-Y. Huang, V Aggarwal, A. Chowdhury, O. Poursaeed, J. Hoffman, J. Malik, Y. Li, and C. Feichtenhofer. Hiera: A hierarchical vision transformer without the bells-and-whistles. International Conference on Machine Learning (ICML), 2023.

