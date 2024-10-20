---
layout: review
title: "Multi-Layers attention-based explainability via transformers for tabular data"
tags: transformer, explainability
author: "Olivier Bernard"
cite:
    authors: "Andrea Trevino Gavito, Diego Klabjan, and Jean Utke"
    title:   "Multi-Layers attention-based explainability via transformers for tabular data"
    venue: "arXiv 2024"
pdf: "https://arxiv.org/pdf/2302.14278"
---

# Notes

* No code available :(


# Highlights

* Investigate explainable models based on transformers for tabular data
* Use of knowledge distillation (master/student) to train a single head but multi-layers (blocs) transformer to facilitate explicability analysis
* Propose a graph-oriented explainability method based on the set of single head attention matrices
* Compare this approach to attention-, gradient-, and perturbation-based explainability methods

* TODO: Highlight 1
* TODO: Highlight 2
* TODO: Highlight 3

&nbsp;

# Introduction

* The field of explainable Artificial Intelligence is named XAI and has received increasing interest over the past decade
* XAI algorithms for DL can be organized into three major groups: perturbation-based, gradient-based, and, more recently, attention-based
* Transformers posses a built-in capability to provide explanations for its results via the analysis of attention matrices

![](/collections/images/tabular_explainability/tab_exp_1.jpg)

* There exists a total of $$N \times h$$ attention matrices for a standart transformer encoder composed by $$N$$ blocs and $$h$$ heads

See [the tutorial on transformers](https://creatis-myriad.github.io/tutorials/2022-06-20-tutorial_transformer.html) for more details.

&nbsp;

# Methodology

## Groups of features

* ***Hypothesis 1***: features within tabular data can often be grouped intuitively and naturally based on factors such as their source (e.g. sensors, monitoring systems, surveys) and type (e.g demographic, ordinal, or geospatial data)
* ***Hypothesis 2***: given that TD does not provide sequential information, positional encoding is disabled

![](/collections/images/tabular_explainability/tokenization.jpg)

&nbsp;

## Knowledge distillation

* A full-capacity transformer ($$N$$ blocs, $$h$$ heads) is first trained for a classification task. This transformers is seen as a ***master transformer*** 
* A ***student transformer*** is then trained to reproduce the same prediction as the ones from the master but using single heads ($$h=1$$) with more blocs ($$M>N$$)
* The following student's loss function is used

$$\mathcal{L}= - \sum_{i=1}^{n} y_i \log \left( \hat{y}_i \right) \, + \, \lambda \sum_{l=1}^{M} \sum_{j,k=1}^{m} a^{l}_{j,k} \log \left( a^{l}_{j,k} \right)$$

* The first term forces the student prediction ($$\hat{y}_i$$) to be close to the one of the master ($$y_i$$)
* The second term forces the entropy of each attention matrix to be low => it forces the information contained in each attention matrix to be concentrated on few cells => it forces the attention matrices to be sparse !

&nbsp;

## Multi-layer attention-based explainability

* Maps the attention matrices across encoder layers into a directed acyclic graph (DAG)
* The DAG is defined as $$D=(V,A)$$, where $$V$$ and $$A$$ are the set of vertices and arcs that compose the graph $$D$$
* The vertices $$V= \bigcup_{l=0}^{M}  \{ v^l_c \}$$ correspond to groups of features 
* The arcs $$\left( v^{l-1}_{\hat{c}}, v^{l}_{\tilde{c}}\right) \in A$$ correspond to attention values $$a^l_{\hat{c},\tilde{c}}$$, where $$\hat{c}, \tilde{c} \in {1,\cdots,m}$$

![](/collections/images/tabular_explainability/from_attention_to_graph.jpg)

* The maximum probability path $$p$$ is found using Dijkstra’s algorithm and is of the form $$p=\{ v^{0}_{i_0}, v^{1}_{i_1}, \cdots, v^{M}_{i_M} \}$$ 
* The arc cost is $$- \log\left( a^l_{j,k} \right)$$ for $$a^l_{j,k} > 0$$, yielding path cost $$- \log\left( \prod_{l=1}^{M} a^l_{i_{l-1},i_{l}} \right)$$
* The authors focus on the group corresponding to the most relevant input for the final prediction, i.e. group $$c=i_0$$

> Explanations to the student’s predictions are provided by finding the most relevant group for the classification
task, i.e. the group $$c=i_0$$ corresponding to the first vertex $$v^0_{i_0}$$ of the maximum probability path $$p$$ in graph $$D$$

* A single group does not always provide all the relevant information to make a prediction 
* Additional groups are ranked iteratively, i.e. in each iteration the starting point $$v^0_{i_0}$$ of the previously found highest probability path is eliminated from the graph and then search for the respective next highest probability path in $$D$$
* In the experiments, two best groups were used as most to explain predictions

&nbsp;

# Results

* 6 different kinds of image generation: text-to-Image, Layout-to-Image, Class-Label-to-Image, Super resolution, Inpainting, Semantic-Map-to-Image 
* Latent space with 2 different regularization strategies: *KL-reg* and *VQ-reg*
* Latent space with different degrees of downsampling
* LDM-KL-8 means latent diffusion model with KL-reg and a downsampling of 8 to generate the latent space 
* DDIM is used during inference (with different number of iterations) as an optimal sampling procedure
* FID (Fréchet Inception Distance): captures the similarity of generated images to real ones better than the more conventional Inception Score

&nbsp;

## Perceptual compression tradeoffs

<div style="text-align:center">
<img src="/collections/images/latent-DM/results-perceptual-compression.jpg" width=600></div>
<p style="text-align: center;font-style:italic">Figure 3. Analyzing the training of class-conditional LDMs with
different downsampling factors f over 2M train steps on the ImageNet dataset.</p>

* LDM-1 corresponds to DM without any latent representation
* LDM-4, LDM-8 and LDM-16 appear to be the most efficient
* LDM-32 shows limitations due to high downsampling effects

&nbsp;

## Hyperparameters overview


<div style="text-align:center">
<img src="/collections/images/latent-DM/results-hyperparameters-unconditioned-cases.jpg" width=600></div>
<p style="text-align: center;font-style:italic">Table 1. Hyperparameters for the unconditional LDMs producing the numbers shown in Tab. 3. All models trained on a single NVIDIA A100.</p>

&nbsp;

<div style="text-align:center">
<img src="/collections/images/latent-DM/results-hyperparameters-conditioned-cases.jpg" width=600></div>
<p style="text-align: center;font-style:italic">Table 2. Hyperparameters for the conditional LDMs trained on the ImageNet dataset. All models trained on a single NVIDIA A100.</p>

&nbsp;

## Unconditional image synthesis

<div style="text-align:center">
<img src="/collections/images/latent-DM/results-image-generation-uncondition.jpg" width=600></div>
<p style="text-align: center;font-style:italic">Table 3. Evaluation metrics for unconditional image synthesis. N-s refers to N sampling steps with the DDIM sampler. ∗: trained in KL-regularized latent space</p>

<div style="text-align:center">
<img src="/collections/images/latent-DM/results-image-generation-uncondition-CelebA-HQ.jpg" width=600></div>
<p style="text-align: center;font-style:italic">Figure 4. Random samples of the best performing model LDM-4 on the CelebA-HQ dataset. Sampled with 500 DDIM steps (FID = 5.15)</p>

<div style="text-align:center">
<img src="/collections/images/latent-DM/results-image-generation-uncondition-bedrooms.jpg" width=600></div>
<p style="text-align: center;font-style:italic">Figure 5. Random samples of the best performing model LDM-4 on the LSUN-Bedrooms dataset. Sampled with 200 DDIM steps (FID = 2.95)</p>

&nbsp;

## Class-conditional image synthesis

<div style="text-align:center">
<img src="/collections/images/latent-DM/results-image-generation-condition-ImageNet.jpg" width=600></div>
<p style="text-align: center;font-style:italic">Table 4. Comparison of a class-conditional ImageNet LDM with
recent state-of-the-art methods for class-conditional image generation on ImageNet. c.f.g. denotes classifier-free guidance with a scale s</p>

<div style="text-align:center">
<img src="/collections/images/latent-DM/results-class-conditional-image-synthesis.jpg" width=600></div>
<p style="text-align: center;font-style:italic">Figure 6. Random samples from LDM-4 trained on the ImageNet dataset. Sampled with classifier-free guidance scale s = 5.0 and 200 DDIM steps</p>

&nbsp;

## Text-conditional image synthesis

* a LDM with 1.45B parameters is trained using KL-regularized conditioned on language prompts on LAION-400M
* use of the BERT-tokenizer
* $$\tau_{\theta}$$ is implemented as a transformer to infer a latent code which is mapped into the UNet via (multi-head) cross-attention

<div style="text-align:center">
<img src="/collections/images/latent-DM/results-text-conditional-image-synthesis.jpg" width=600></div>
<p style="text-align: center;font-style:italic">Table 5. Evaluation of text-conditional image synthesis on the
256×256-sized MS-COCO dataset: with 250 DDIM steps</p>

<div style="text-align:center">
<img src="/collections/images/latent-DM/results-text-conditional-image-synthesis-2.jpg" width=600></div>
<p style="text-align: center;font-style:italic">Figure 7. Illustration of the text-conditional image synthesis. Sampled with 250 DDIM steps</p>

&nbsp;

## Semantic-map-to-image synthesis

* Use of images of landscapes paired with semantic maps 
* Downsampled versions of the semantic maps are simply concatenated with the latent image representation of a LDM-4 model with VQ-reg.
* No cross-attention scheme is used here
* The model is trained on an input resolution of 256x256 but the authors find that the model generalizes to larger resolutions and can generate images up to the megapixel regime


<div style="text-align:center">
<img src="/collections/images/latent-DM/results-semantic-synthesis.jpg" width=400></div>
<p style="text-align: center;font-style:italic">Figure 8. When provided a semantic map as conditioning, the LDMs generalize to substantially larger resolutions than those seen during training. Although this model was trained on inputs of size 256x256 it can be used to create high-resolution samples as the ones shown here, which are of resolution 1024×384</p>



&nbsp;

# Conclusions

* Latent diffusion model allows to synthesize high quality images with efficient computational times.
* The key resides in the use of an efficient latent representation of images which is perceptually equivalent but with reduced computational complexity

&nbsp;

# References
\[1\] P. Esser, R. Rombach, B. Ommer, *Taming transformers for high-resolution image synthesis*, CoRR 2022, [\[link to paper\]](https://arxiv.org/pdf/2012.09841.pdf)

\[2\] A. van den Oord, O. Vinyals, and K. Kavukcuoglu, *Neural discrete representation learning*, In NIPS, 2017 [\[link to paper\]](https://arxiv.org/pdf/1711.00937.pdf)



