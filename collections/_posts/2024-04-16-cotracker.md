---
layout: review
title: "CoTracker: It is Better to Track Together"
tags: motion estimation, transformer
author: "Olivier Bernard"
cite:
    authors: "Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, Christian Rupprecht"
    title: "CoTracker: It is Better to Track Together"
    venue: "Arxiv 2023"
pdf: "https://arxiv.org/pdf/2307.07635.pdf"
---

# Notes

* The [official webpage](https://co-tracker.github.io/) with paper/code/demo 

&nbsp;

# Highlights

* Transformer-based model that tracks dense points in a frame jointly across a video sequence
* Joint tracking results in a significantly higher tracking accuracy and robustness
* Can track arbitrary points, selcted at any spatial location and time in the video 
* Operates causally on short windows (suitable for online tasks)
* Trained by unrolling the windows across longer video sequences, which improves long-term tracking


* Introduce virtual tracks concepts which allows the tracking of 70k points jointly and simultaneously
* Introduce occlusion concepts which allows the tracking of points for a long time even when they are occluded or leave the field of view

&nbsp;

# Introduction

* Numerous existing AI methods perform motion estimation only between two consecutive frames

&nbsp;

# Methodology

## Context formulation

* The goal is to track 2D points throughout the duration of a video $$V=\left(I_t\right)_{t=1}^{T}$$ composed of a sequence of $$N$$ frame $$I_t \in \mathbb{R}^{3 \times W \times H}$$

* CoTracker predicts $$N$$ **_point tracks_** $$P_t^i$$, with 

$$\left\{
\begin{aligned}
    & P_t^i = \left(x_t^i,y_t^i\right) \in \mathbb{R}^2 \\
    & i \in \{1, \cdots N \} \\
    & t \in \{t_i,\cdots,T \} \\
    & t_i \in \{1, \cdots, N\} \quad \text{is the time when the track starts}
\end{aligned}
\right.$$

![](/collections/images/cotracker/context-formulation.jpg)

&nbsp;

* CoTracker also estimates the visibility flag $$\nu_i \in \{0,1\}$$ which tells if a point is visible or occluded in a given frame

&nbsp;

## Information extaction

**_1) Image features_**

* $$\phi(I_t) \in \mathbb{R}^{d \times \frac{H}{4} \times \frac{W}{4}}$$ are _d_-dimensional appearance features that are extracted from each video frame using a CNN which is trained end-to-end. 

* The resolution is reduced by a factor of $$4$$ for efficiency

* $$\phi_s(I_t) \in \mathbb{R}^{d \times \frac{H}{4 \cdot 2^{s-1}} \times \frac{W}{4 \cdot 2^{s-1}}}$$ are scaled versions of the appearance features used to compute correlation features

* $$s \in \{ 1, \cdots, S \}$$ with $$S=4$$ in their implementation

![](/collections/images/cotracker/image-features.jpg)

&nbsp;

**_2) Track features_**

* $$Q_t^i \in \mathbb{R}^d$$ are appearance features of the **_tracks_** 

* They are time dependent to accomodate changes in the track appearance

* They are initialized by sampling iamge features at the starting locations

* They are updated by the transformer

![](/collections/images/cotracker/track-features.jpg)

&nbsp;

**_3) Correlation features_**

* $$C_t^i$$ are correlation features which are computed to facilitate the matching of the point tracks 

* Each $$C_t^i$$ is obtained by comparing the track features $$Q_t^i$$ to the image features $$\phi_s\left(I_t\right)$$ around the current estimate $$\hat{P}_t^i$$ of the track's location

* The vector $$C_t^i$$ is obtained by stacking the inner products $$\left<Q_t^i,\phi_s\left(I_t\right)\left[ \hat{P}_t^i /(4 \cdot s) + \delta \right] \right>$$ 

* $$s \in \{1,\cdots,S\}$$ are the feature scales

* $$d \in \mathbb{Z}^2$$, $$\|\delta\|_{\infty} \leq \Delta$$ are offsets

* The image features $$\phi_s\left(I_t\right)$$ are sampled at non-integer locations by using bilinear interpolation and border padding

* The dimension of $$C_t^i$$ is $$(2\Delta+1)^2S = 196$$ with $$S=4$$ and $$\Delta=3$$

&nbsp;

## Transformer formulation

**_Overal framework_**

* CoTracker is a transformer $$\Psi : G \rightarrow O$$ whose goal is to improve and initial estimate of tracks

* The input $$G$$ corresponds to a grid of input tokens $$G_t^i$$, one for each point track $$i \in \{1, \cdots N \}$$ and $$t \in \{1,\cdots,T \}$$

* The output $$O$$ is a grid of output tokens $$O_t^i$$ which are used to update the point tracks during iterations

&nbsp;

**_Input tokens_**

* The input tokens $$G(\hat{P},\hat{\nu},\hat{Q})$$ code for position, visibility, appearance, and correlation of the point tracks and is given as:

$$G_t^i = \left( \hat{P}_t^i - \hat{P}_1^i, \, \hat{\nu}_t^i, \, Q_t^i, \, C_t^i, \, \eta \left( \hat{P}_t^i - \hat{P}_1^i \right) \right) + \eta' \left( \hat{P}_t^1 \right) + \eta' \left( t \right) $$

* $$\eta(\cdot)$$ is a positional encoding of the track location with respect to the initial location at time $$t=1$$

* $$\eta'(\cdot)$$ is a positional encoding of the start position $$\hat{P}_1^i$$ and for time $$t$$, with appropriate dimensions

* The estimates $$\hat{P}$$, $$\hat{\nu}$$ and $$Q$$ are initialized by broadcasting the initial values $$P_{t_i}^{i}$$ (the location of query point), $$1$$ (meaning visible) and $$\phi\left(I_t\right)[P_{t_i}^{i}/4]$$ (the appearance of the query point) to all time $$t \in \{1,\cdots,T\}$$

![](/collections/images/cotracker/cotracker-initialization.jpg)

&nbsp;

**_Output tokens_**

* The output tokens $$O\left( \Delta \hat{P}, \Delta Q \right)$$ contains updates for location and appareance, i.e. $$O_t^i = \left( \Delta \hat{P}_t^i, \Delta Q_t^i \right)$$

&nbsp;

**_Iterated transformer applications_**

&nbsp;

# Results

* TODO
* TODO

&nbsp;

## TODO

* TODO
* TODO

## TODO

* TODO
* TODO

# Conclusions

* TODO
* TODO
* TODO
* TODO

&nbsp;

# References
\[1\] TODO



