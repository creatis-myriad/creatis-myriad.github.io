---
layout: review
title: "C2FTrans: Coarse-to-Fine Transformers for Medical Image Segmentation"
tags: Coarse-to-Fine-Transformers Segmentation
author: "Hang Jung Ling"
cite:
    authors: "Xian Lin, Zengqiang Yan, Li Yu, and Kwang-Ting Cheng"
    title:   "C2FTrans: Coarse-to-Fine Transformers for Medical Image Segmentation"
    venue:   "arxiv"
pdf: "https://arxiv.org/pdf/2206.14409v1.pdf"
---

> 📝 This paper has been submitted to IEEE TMI and is currently under review at the time of writing this post.  

# Notes
* Code is available on GitHub: [https://github.com/xianlin7/C2FTrans](https://github.com/xianlin7/C2FTrans).

&nbsp;

# Highlights
* Invention of a new transformer architecture, namely Coarse-to-Fine Transformer (C2FTrans), in medical image segmentation. C2FTrans consists of a cross-scale global transformer (CGT) which addresses local contextual similarity in CNN and a boundary-aware local transformer (BLT) which overcomes boundary uncertainty brought by rigid patch partitioning in transformers. 
* C2FTrans has only **1.2M parameters**.

&nbsp;

# Architecture
![](/collections/images/C2Ftrans/architecture.jpg)
The authors use a full U-Net as backbone for the feature extraction and add transformer modules (CGT and BLT) to the decoder the U-Net. This idea is quite different from [UNETR]({% post_url 2022-07-01-UNETR-TransformerMedicalImageSegmentation %}){:target="_blank"} that replaces directly the encoder of the U-Net with transformers. 

## Cross-scale Global Transformer (CGT)
![](/collections/images/C2Ftrans/CGT.jpg)
CGT is made up of two cross-attention modules and a feed forward network (FFN). CGT aims to mix information coming from the three lowest resolution feature maps, *i.e.*, $$ F_3 \in \mathbb{R}^{4 \text{C} \times \frac{\text{H}}{4} \times \frac{\text{W}}{4}} $$, $$ F_4 \in \mathbb{R}^{8 \text{C} \times \frac{\text{H}}{8} \times \frac{\text{W}}{8}} $$, and $$ F_5 \in \mathbb{R}^{16 \text{C} \times \frac{\text{H}}{16} \times \frac{\text{W}}{16}} $$. 

Projections of these three feature maps to obtain $$ Q $$, $$ K $$, and $$ V $$:
*  $$ F_3 $$ is projected into query, $$ Q_{3,i} \in \mathbb{R}^{\frac{\text{HW}}{16} \times \text{D}_h} $$, $$ i \in \{4,5\} $$, where $$ \text{D}_h = 128 $$ denotes the dimension of the transformer module. A patch size of $$ (1 \times 1) $$ is used for tokenization:
  
> $$ F_3 \in \mathbb{R}^{4 \text{C} \times \frac{\text{H}}{4} \times \frac{\text{W}}{4}} \xrightarrow[\text{reshape}]{} \mathbb{R}^{\frac{\text{HW}}{4 \times 4 \times (1 \times 1)} \times ((1 \times 1) \times 4 \text{C})} = \mathbb{R}^{\frac{\text{HW}}{16} \times 4 \text{C}} \xrightarrow[\text{tokenization}]{\mathbb{R}^{4 \text{C} \times \text{D} (=64)}} F^{'}_3 \in \mathbb{R}^{\frac{\text{HW}}{16} \times \text{D}} \xrightarrow[\text{projection/ self-attention}]{\mathbb{R}^{\text{D} \times \text{D}_h}} Q_{3,i} \in \mathbb{R}^{\frac{\text{HW}}{16} \times \text{D}_h} $$

*  $$ F_4 $$ is projected into $$ \{K_4, V_4\} $$, where $$ K_4, Q_4 \in \mathbb{R}^{\frac{\text{HW}}{64} \times \text{D}_h} $$.

*  Likewise, $$ F_5 $$ is projected into $$ \{K_5, V_5\} $$, where $$ K_5, Q_5 \in \mathbb{R}^{\frac{\text{HW}}{256} \times \text{D}_h} $$.

Cross-scale attention is then obtained by $$ F^i_{ca} (Q_{3,i}, K_i, V_i) = \text{softmax}(\frac{Q_{3,i} K^T_i}{\sqrt{\text{d}}}) V_i $$, where

$$ F^4_{ca} \in (\mathbb{R}^{\frac{\text{HW}}{16} \times \text{D}_h} \times \mathbb{R}^{\text{D}_h \times \frac{\text{HW}}{64}} \times \mathbb{R}^{\frac{\text{HW}}{64} \times \text{D}_h}) = \mathbb{R}^{\frac{\text{HW}}{16} \times \text{D}_h} $$ 

$$ F^5_{ca} \in (\mathbb{R}^{\frac{\text{HW}}{16} \times \text{D}_h} \times \mathbb{R}^{\text{D}_h \times \frac{\text{HW}}{256}} \times \mathbb{R}^{\frac{\text{HW}}{256} \times \text{D}_h}) = \mathbb{R}^{\frac{\text{HW}}{16} \times \text{D}_h} $$

To be noted that during this step, multiscale features are mixed. According to the authors, this is crucial as lower resolution feature maps correspond to larger receptive fields, hence contain richer semantic information. By extracting $$ \text{K} $$ and $$ \text{V} $$ from lower resolution feature maps, computational complexity is reduced by at least a factor of 4 since their sequence length is shorter.  

After that, $$ F^4_{ca} $$ and $$ F^5_{ca} $$ from all the transformer heads are concatenated and connected residually with $$ F^{'}_3 $$. To do so, another linear projection matrix $$ W_{ca} \in \mathbb{R}^{2 \text{gD}_h \times \text{D}} $$ is learned, where $$ \text{g} $$ represents the number of self-attention heads. This step is repeated 4 times (4-layer transformer encoder): 

$$
\begin{align}

    F_{ca,1} &= \underbrace{\text{concat}(F^4_{ca,1}, \cdots, F^4_{ca,g}, F^5_{ca,1}, \cdots, F^5_{ca,g})}_{\mathbb{R}^{\frac{\text{HW}}{16} \times \text{D}_h \times \text{2g}} \xrightarrow[\text{reshape}]{} \mathbb{R}^{\frac{\text{HW}}{16} \times \text{2gD}_h}} \cdot \overbrace{W_{ca}}^{\mathbb{R}^{2 \text{gD}_h \times \text{D}}} + F^{'}_3 \in \mathbb{R}^{\frac{\text{HW}}{16} \times \text{D}} \\

    F_{ca,1} &= \text{FFN}(F_{ca,1}) + F^{'}_3  \\ 

    F_{ca,2} &= \underbrace{\text{concat}(F^{4'}_{ca,1}, \cdots, F^{4'}_{ca,g}, F^{5'}_{ca,1}, \cdots, F^{5'}_{ca,g})}_{\mathbb{R}^{\frac{\text{HW}}{16} \times \text{D}_h \times \text{2g}} \xrightarrow[\text{reshape}]{} \mathbb{R}^{\frac{\text{HW}}{16} \times \text{2gD}_h}} \cdot \overbrace{W^{'}_{ca}}^{\mathbb{R}^{2 \text{gD}_h \times \text{D}}} + F_{ca,1} \in \mathbb{R}^{\frac{\text{HW}}{16} \times \text{D}} \\

    F_{ca,2} &= \text{FFN}(F_{ca,2}) + F_{ca,1}  \\ 

    &\;\;\vdots \notag \\
\end{align}
$$

To obtain the final CGT output, $$ F_{CGT} $$:
> $$ F_{ca,4} \in \mathbb{R}^{\frac{\text{HW}}{16} \times \text{D}} \xrightarrow[\text{reshape}]{} \mathbb{R}^{\text{D} \times \frac{\text{H}}{4} \times \frac{\text{W}}{4}} \xrightarrow[1 \times 1 \text{ conv}]{} F_{CGT} \in \mathbb{R}^{\text{D} \times \frac{\text{H}}{4} \times \frac{\text{W}}{4}} $$

> ⚠️ <span style="color:red">In the paper, it is written that they learn two projection matrices and apply a residual connection to get $$ F_{CGT} $$, which contradicts their GitHub code.</span>

Two functionalities of $$ F_{CGT} $$:
* Generate a low-resolution mask for multiscale loss computation.

$$ F_{CGT} \xrightarrow[1 \times 1 \text{ conv}]{} \mathbb{R}^{\text{num class} \times \frac{\text{H}}{4} \times \frac{\text{W}}{4}} $$

* Upsampled to obtain the probability map of initial image dimension that will be concatenated with the another two probaility maps from U-Net and BLT to form the final probability map.  

$$ F_{CGT} \xrightarrow[\text{expand}]{} \mathbb{R}^{\text{D} \times \frac{\text{H}}{4} \times \frac{\text{W}}{4} \times 1} \xrightarrow[\text{repeat along last dim}]{} \mathbb{R}^{\text{D} \times \frac{\text{H}}{4} \times \frac{\text{W}}{4} \times 16} \xrightarrow[\text{reshape}]{} \mathbb{R}^{\text{D} \times \text{H} \times \text{W}} $$

* Used to compute boundary-aware windows for BLT.

$$ F_{CGT} \xrightarrow[1 \times 1 \text{ conv}]{} \mathbb{R}^{\text{num class} \times \frac{\text{H}}{4} \times \frac{\text{W}}{4}} \xrightarrow[\text{softmax}]{} P_{CGT} $$


## Boundary-aware Local Transformer (BLT)
![](/collections/images/C2Ftrans/BLT.jpg)

According to the authors, the rigid patch partitioning of transformer destroys the vital details around boundaries. Hence, the core contribution of BLT is to perform local self-attention within boundary-aware windows. 

Generation of boundary-aware windows:  
1) Create evenly and densely tiled windows over feature map $$ F_2 \in \mathbb{R}^{2\text{c} \times \frac{\text{H}}{2} \times \frac{\text{W}}{2}} $$ to obtain an initial window set $$ \{w \} $$.  
2) Calculate entropy of each pixels in $$ P_{CGT} $$. These entropies are then used to compute the scores for each 

&nbsp;

# Benchmarking datasets


&nbsp;

# Results

&nbsp;

# Conclusions





