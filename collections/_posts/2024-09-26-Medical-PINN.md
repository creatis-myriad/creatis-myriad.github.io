---
layout: review
title: "Medical PINN: non invasive blood pressure estimation"
tags: Flow, Physics, Neural-Networks
author: "Clara Cousteix"
cite:
    authors: "Sel, K., Mohammadi, A., Pettigrew, R. I., & Jafari, R."
    title: "Physics-informed neural networks for modeling physiological time series for cuffless blood pressure estimation"
    venue: "npj Digital Medicine"
pdf: "https://www.nature.com/articles/s41746-023-00853-4"
---

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>


# Context
## Flow around a expresso cup

* PINNs, abreviation for physics-informed neural networks, were presented by Hang Jung two years ago with his post on "Espresso Cup"
* PINN are neural networks that embed into the loss function the knowledge of any theorical laws, such as physical law (eg Navier-Stockes). It can be seen as a regularization term that limits the space of admissible solutions
* In the espresso cup paper, the purpose is to perform an regression task with the estimation of **pressure**, **temperature** and **velocity** around a espresso cup. The loss in this paper is informed with residuals of Navier-Stockes and heat equation that we try and minimize.

<div style="text-align:center">
<img src="/collections/images/PINN_espresso/tomo_bos_seq.jpg" width=600>
</div>

## PINNs for blood pressure estimation?
* The today paper tackles the translation of PINNs to a medical application : the blood pressure estimation. 
* We can already measure blood pressure with a cuff around the arm but can be inconvenient for continuous measure. We would like to be able to determine blood pressure from a more convenient device such as bioimpedance electrodes:

<div style="text-align:center">
<img src="/collections/images/medical_PINN/supp figure 1 bioZ.jpg" width=600>
<h5 style="font-weight: normal;"><u>Figure 2</u>: Bioimpedance measure with electrodes on wrist or finger</h5>
</div>

* The objectives of this paper is to establish a method for blood pressure estimation with PINN using bioimpedance data and limited amount of ground truth blood pressure measures.

# Methodology

## Input data

<div style="text-align:center">
<img src="/collections/images/medical_PINN/41746_2023_853_Fig1_b.jpg" width=800>
<h5 style="font-weight: normal;"><u>Figure 3</u>: Bioimpedance signal segmented into R cycles</h5>
</div>

<div style="text-align:center">
<img src="/collections/images/medical_PINN/41746_2023_853_Fig2_HTML.jpg" width=400>
<h5 style="font-weight: normal;"><u>Figure 4</u>: Feature extraction from a bioimpedance cycle: u1, u2 and u3 </h5>
</div>

## Outputs and ground truth

We try and estimate blood pressure characteristics such as : 
* Systolic Blood Pressure (SBP)
* Diastolic Blood Pressure (DBP)
* Pulse Pressure (PP)

As a ground truth, we have blood pressure measures associated to bioimpedance cycles: 

<div style="text-align:center">
<img src="/collections/images/medical_PINN/41746_2023_853_Fig1_aa.jpg" width=150>
<h5 style="font-weight: normal;"><u>Figure 5</u>: Association of ground truth and inputs </h5>
</div>


## Architecture and Loss

<div style="text-align:center">
<img src="/collections/images/medical_PINN/41746_2023_853_Fig1_c.jpg" width=600>
<h5 style="font-weight: normal;"><u>Figure 6</u>: Architecture of PINN </h5>
</div>



* Inputs:
    * BioImpendance $BioZ$
    * Bioimpedance extracted features $u_1$, $u_2$ and $u_3$
* Outputs:
    * Blood pressure (diastolic $DBP$, systolic $SBP$, pulpe pressure $PP$)
* Loss
    * $\mathcal{L}_{supervised} = \Sigma_{i=1}^{s} (y_{i, meas}-y_{i, pred})²$ 
    * $\mathcal{L}_{physics} = \frac{1}{R-1}\Sigma_{i=1}^{R-1}\Sigma_{k=1}{3}(y_{i+1, NN}-(y_{i, NN} + \frac{\partial y_{i, NN}}{\partial u_{i}^{k}}(u_{i+1}^{k}-u_{i}^{k})))^2$ 
    inspired by the Taylor approximation
    * $\mathcal{L}_{total} = \alpha \mathcal{L}_{supervised} + \beta \mathcal{L}_{physics}$ with $\alpha = 1$ and $\beta = 10$

# Results and conclusion

<div style="text-align:center">
<img src="/collections/images/medical_PINN/41746_2023_853_Fig3_HTML.jpg" width=600>
<h5 style="font-weight: normal;"><u>Figure 7</u>: Comparison results of PINN vs CNN </h5>
</div>

* Prediction is continuous and more accurate 
* Feature extraction : relevant if chosen features are significant. Here $u_1$, $u_2$ and $u_3$ are descriptors related to cardiovascular characteristics
* Taylor approximation : provides representation of input/ouput relation
* Good results with minimal training data, outperforms state-of-the-art models with minimal training data

<div style="text-align:center">
<img src="/collections/images/medical_PINN/supp figure 8 results.jpg" width=600>
<h5 style="font-weight: normal;"><u>Figure 8</u>: Comparison results PINN vs other SOTA regressor </h5></div>

# Discussion

* The aspect "physics-informed" is in the Taylor Approximation and in the feature extraction, which both required some prior knowledge and inform the network. That is being said, we can regret the absence of a physical equation. The term "physics-informed" seem abusive whereas "Theory-Trained Neural Networks" seem more appropriate. 
* The contribution is more about a Proof of Concept, the experience being conducted on a small amount of participant (15 participants, including 1 woman)