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

* The official webpage with paper/code/demo is available [here](https://co-tracker.github.io/) 

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

**_Image features_**

* $$\phi(I_t) \in \mathbb{R}^{d \times \frac{H}{4} \times \frac{W}{4}}$$ are _d_-dimensional appearance features that are extracted from each video frame using a CNN which is trained end-to-end. 

* The resolution is reduced by a factor of $$4$$ for efficiency

* $$\phi_s(I_t) \in \mathbb{R}^{d \times \frac{H}{4 \cdot 2^{s-1}} \times \frac{W}{4 \cdot 2^{s-1}}}$$ are scaled versions of the appearance features used to compute correlation features

* $$s \in \{ 1, \cdots, S \}$$ with $$S=4$$ in their implementation

![](/collections/images/cotracker/image-features.jpg)

&nbsp;

**_Track features_**

* $$Q_t^i \in \mathbb{R}^d$$ are appearance features of the **_tracks_** 

* They are time dependent to accomodate changes in the track appearance

* They are initialized by sampling iamge features at the starting locations

* They are updated by the transformer

![](/collections/images/cotracker/track-features.jpg)

## Transformer formulation

* TODO

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



