---
layout: review
title: "A principled representation of elongated structures using heatmaps"
tags: machine-learning neural evolutionary-computing
author: "Morgane des Ligneris"
cite:
    authors: "Kordon, F., Stiglmayr, M., Maier, A. et al. "
    title:   "A principled representation of elongated structures using heatmaps"
    venue:   "Nature, sientific reports"
pdf: https://www.nature.com/articles/s41598-023-41221-2
---

# Highlights
- Propose a mathematical description of 2D and 3D elongated structures 
- Create a target function that CNN can well aproximate
- Convert the curve into a heatmap through a convolution with a filter function  
- Linear time complexity 
- Applicalble to various surgical 2D and 3D data task

# Introduction

**Elongated structures** = line, boundary, ridge or contour that should be detected in an image

Detecting elongated structures in images is valuable in many fields, like geography, expression recognition, medical imaging (e.g., X-ray catheter monitoring).

Traditionally, methods use differences in intensity or color to find these structures. In cases where there's no specific model for the structure, Convolutional Neural Networks (CNN) are effective.However, there hasn't been a standard method for representing these elongated structures.

Two common approaches have limitations : 
- Binary segmentation often results in gaps and fragmentation. 
- Skeletonization is sensitive to minor errors.

A new approach, used for facial boundaries detection, involves heatmaps to localise landmark. In this paper they extend the concept of heatmaps from single points to continuous curves in space, while taking in account their positional uncertainty. 

![](/collections/images/elongated_structures/fig_1.jpg)

# Methods

## Heatmap for elongated structures
 
**Representation of a curve as the image of a parameter function** :  

The position of a curve $$\tilde{c} = c(t)+ z(t)$$
with the parameter variable $$t ∈[\underset{0}{t}, \underset{n}{t}]$$ is the sum of the position of the curve $$c(t)$$ superposed by a noise term $$z(t)$$ with probability density function (pdf)

- traces a noisy image of the curve
- can be used as an optimization target for CNN training 

**Description of a curve as a level set**

$$f(x)$$ is a measure for the distance of a point x to the curve : \\
$$(1) \quad C = \{x \in \mathbb{R}^n : f(x) = 0\}$$ 

$$D$$ is the distribtion of a curve : \\
$$(2) \quad D(x) = \delta(f(x))$$ 
- with $$f(x)$$ the distance function, in the space $$\mathbb{R}^n$$
- with $$\delta(s)$$, $$s \in \mathbb{R}$$ the Dirac function. 
$$\delta(t) = \begin{cases}
    0 & \text{pour } t \neq 0 \\
    \infty & \text{pour } t = 0
\end{cases}$$

Now it is possible to convolve this distribution with a filter function $$w(x)$$ and obtain the heatmap :
$$(3) \quad \begin{align*}
H(x) &= D(x) * w(x) \\
     &= \int_{\mathbb{R}^2} \delta(f(y)) w(x - y) dy \\
     &= \int_C w(x - y) dy
\end{align*}$$

To be faster than while using a numerical integration and to generalize the description, $$H(x)$$ is simlified. They make the assumption that $$w(x)$$ is symetric $$w(x) = w(y)$$ for all $$\vert \vert x \vert \vert \underset{2}{} = \vert \vert y \vert \vert \underset{2}{}$$.

If $$C$$ is an infinite straight line, the evaluation of the integral simplifies to $$h : \mathbb{R} \rightarrow \mathbb{R}$$ which only depends on the distance to the line C : 

$$(4) \quad H(x)= h(f(x))$$

(4) will be used in the general case as an approximation for the integral in (3).

## Numerical approximation and implementation

![](/collections/images/elongated_structures/fig_2.jpg)

Here are the different steps of this approximation of a convolution by a distance dependent function : 

- **data point :** description of the original structure with a finite set of data points
- **curve approximation** : the data points are used to derive a functional curve approximation by performing a smooth parametric B-spline interpolation
- **restricting calculation** to a “compact hull” or close region to the curve because only spatial positions within a limited distance to the curve are relevant
- **sampling the interpolating function** : selecting a set of points from the B-spline curve created
- **form a 2D Polyline** : sequence of connected line segments, approximation of the original curve.
- **bounding box around each segment** (minimum enclosing box)
- **calculate distance value for each spatial position inside** : distance from the spatial position to the original curve
- **heatmap calculation** : resolve multiple distance estimates from overlapping bounding boxes then evaluate the distance values with the distance dependent function.

**Curve parameterization**
Approximate spline to be less sensitive to label or sampling noise. And use the arc-length parametrization method because there is not assumption on the curve type and potential physical constraints.

**Local distance estimation**
Create a local hull by constructing the smallest enclosing box for each segment.
Within each bounding box, the point-to-line distance is calculated for every contained spatial location x. The bounding boxes will overlap for non-zero curvatures $$k>0$$ or if the curve is self-intersecting &rarr; multiple distance estimates.

To map $$\underset{x}{f}$$ to a single distance scalar that can be evaluated with $$h(·)$$, there is two options for a reduction function $$h^∗(\underset{x}{f})$$ :
- Minimum distance : only the estimator with the smallest point-to-line distance is used for evaluating the distance-dependent function. Easily parallelized, it allows the heatmap generation algorithm to be executed at linear time complexity
- Inverse distance weighting (IDW) : allows the contribution of all distance estimates.

**Selection of the distance-dependant function**
- frequently observed signal/noise shapes (Gaussian, Laplace, cosine)
- versatile in their encoding characteristics (triangular, rectangular)
- custom signal shape in the form of a simulated hollow-core catheter profile

**Extension to 3D**
The local hull is approximated by extending the two-dimensional bounding boxes to right rectangular prisms.

## Signal extraction and evaluation metrics

Two post-processing protocols for signal recovery and evaluation, with specific curve type and dataset : 

### Center-line extraction and ASSD comparison

Signal estimation as a set of point + post-processing the estimation : 
- turn it to a binary image (using Otsu’s method, automatic image threshpolding)
- transform into a curve with uniform width using medial axis transformation

ASSD between two curves $$C$$ and $$C'$$, 
- minimum euclidiean distance of point $$x$$ in curve $$C$$ to point $$x'$$ on curve $$C'$$ :
$$d(x, C') = \underset{x′ \in C′}{min} \vert \vert x − x' \vert \vert \underset{2}{}$$ 
- evaluating this distance for every point $$x \in C$$ and $$x' \in C'$$ and averaging the resulting distance values :  

$$ASSD = \frac{1}{|C| + |C'|}*(\underset{x \in C}{\Sigma}d(x, C') +\underset{x' \in C'}{\Sigma}d(x, C')d(x', C) )$$ 

Advantage : invariant to the width and distribution of the simulated signal and estimated heatmap

Limitations :
- noisy and self-intersecting heatmap estimates, the thinning can cause branching artifacts 
- larger signal or heatmap width, longer subsidiary branch
- effect is negligible as most errors are caused by signal fragmentation in ambiguous regions

### Non-maximum suppression of the heatmap, morphological thinning and F-measure

Interpret the heatmap as a noisy curve probability map :
- refine the curve estimation using non-maximum suppression (NMS), keep the stronger indicator of a curve’s presence within a certain radius $$r_{spr}$$
- convert probability in heatmap to binary curve label with a probability threshold
- iterative thinning to get a curve with uniform width

Calculate the minimum-cost correspondence (Euclidiean distance) between point on the two curve. Then calculate Precision and Recall to compute F-measure $$F=\frac{2PR}{P+R}$$. Assess how well the estimated curve matches the real one.Select the best curve threshold across all images (either at optimal dataset scale ODS or optimal image scale OIS &rarr; ensuring different scale consideration)

## Neural network architecture 
network's parameters are initialized using the He initialization strategy
cost function by default Mean Square Error but BCE is used in segmentation experiment 

*2D experiments* : Single Hourglass Module (standard model for heatmap-based landmark detection), specific configurations include a feature root of 128, ReLU activation functions, and instance normalization layers to handle varying contrast levels.

*3D experiment* : 3D U-Net architecture with residual connections in encoding and decoding blocks, feature root is set to 16 and is doubled at each of the five encoder levels. 

# Datasets and training protocols

## Simulated signals of chest X-ray (2D experiment)

Simulation of Realistic Curve Signals : 

- background images from COVID-19 image data collection and ChestX-ray, 1125 images 900/225 (80%/20% ) split (training/validation)
- model to simulate realistic curves on X-ray chest with finite number of data points, distributed on a specific area
- mimic real-life elongated structures such as surgical catheters or trivial skeletons
- calculate the heatmap representing the simulated signal, and the strength of the signal is adjusted and there is an addition of noise
- use 200 interpolation points to create smooth lines

Optimization of the hourglass model for 40 epochs, batch size of 2, a learning rate of 0.00025, used the L2 regularization with a factor of 0.00005 and RMSProp update policy.

Online augmentation (shifting, rotating, calling and cropping) + standardized image resolution.

Evaluate the model with ASSD metric.

## Anatomical structures on knee radiographs (2D experiment)  

223 clinical X-ray images centered on the knee-joint collected from anonymized databases + annotations, 174/49 (80%/20%) split.

Optimization of two hourglass model (for direct and contextual structures) 250 epochs with a batch size of 2, a learning rate of 0.00025, and L2 regularization with a factor of 0.00005 using the RMSProp update policy.

Three distinct output channels for the direct anatomical structures instead of combining them into a single heatmap.

Data augmentation (horizontal flipping) + standardized image resolution.

Compared curve approximation variants with and without smoothing conditions to evaluate potential effects of strong local curvature.

Evaluate the model with ODS F-measure.

## Surgical implants in CBCT volumes (3D experiment)

141 3D CBCT volumes recorded across a variety of different body regions (acetabulum, calcaneus, cervical/thoracal/lumbar spine, humerus, distal tibia, proximal tibia, and wrist), down sample (hardware-constraints) + annotations of screw/wire positions, 101 and 40 volumes split.

Optimization with 50 epoch, batch size of 1, learning rate of 0.005.

Evaluate the model with ASSD metric.

# Results

## Analysis of representation properties

**Approximation error for different curvature, heatmap widths and reduction function** 

Approximation yields exact values for straight lines, but it is not the case for curve. Analyze the error introduced by this approximation with distance-dependent weights

![](/collections/images/elongated_structures/fig_3.jpg)


*The influence of the reduction function $$h^*(·)$$*
- Noise and underestimation of the heatmap especially in both IDW schemes (because of overlaping bounding boxes).
- Sparse sampling of the curve reduces the error difference between the different reduction functions but sacrifice precision.

![](/collections/images/elongated_structures/fig_4.jpg)

*The influence of the curve/heatmap width and curbature*
- the larger the curve/heatmap, the higher the susceptibility to errors (even more if high curvature)
- overlap of the cure with itself (not considered in the reduction funciton or implementation) lead to overestimation of the heatmap (due to high-valued distance estimates)

*The relation of sampling rate and curvature*
Shannon-Nyquist theorem 
- High curvature in a curve = rapid changes in direction = higher frequency components
- maximum curvature (κ) linked to the highest frequency component by fmax = κ / (2π).
- sampling theorem requires a sampling rate at least double the highest frequency component → condition r > 4πκ for the sampling rate r

*Impact of the heatmap width*
- sampling quality strongly depends on the chosen heatmap width
- approximation error tends to increase for larger curvatures across all sampling rates
- small improvements in approximation quality from higher sampling rates but higher computational requirements
- self-overlapping of the curve or its noise spread is the main reason for larger errors in the approximation
- high curvatures can be effectively addressed by choosing a small-to-medium-sized heatmap width

## Comparison to related curve representation

Compare the method’s error to that of several related spatial curve representations

### Mean squared error on curve points
- A spatial representation of the curve is created by filtering the discrete curve points with a Gaussian low-pass filter
- high approximation errors with increasing signal complexity (especially high curvature) → cause a shift in the distribution of the points, which affect the likelihood of points conforming to a Gaussian model

### Gaussian-weighted Euclidean distance transform
- represent a curve as a distance field by discretizing its points and using the Euclidean distance transform (EDT)
- distances evaluated using a Gaussian probability density function
- the observed approximation errors using this method are similar to those obtained with a proposed numerical approximation + smoothness increases as more sampling points are used
- stair-step artifacts introduced due to the curve's discretization → can be problematic
- for extremely large noise spread and very dense curve sampling, this representation might be preferred due to very efficient computation with linear time complexity

### Gaussian-weighted BCE
- BCE is typically used for optimizing binary segmentation problems
- a discrete 2D convolution is performed between the curve points and the impulse response of the BCE term
- each position is weighted using a Gaussian probability density function (pdf)
- approximation error calculated by comparing resulting curve with corresponding 1D
- similar to MSE variant, peak flattening and migration toward the inside of the curve (especially in high-curvature areas with substantial noise)
- sparse sampling can also lead to larger errors in regions with little curvature
- minor offsets between the original curve and the approximation result in significant errors (linked to the B-spline interpolation scheme)


## CNN-based estimation error for different signal and heatmap configurations
Signal simulation model and ASSD metric to assess the accuracy of these approximations (*Simulated signals of chest X-ray, 2D experiment)

![](/collections/images/elongated_structures/fig_5.jpg)

*The relation between a signal distribution and heatmap distance-dependent function*
- weak correlation between the signal distribution and the distance-dependent function of the heatmap
- Using a rectangular distribution result in worse reconstruction of the original curve (ASSD metric error up to 5px)
- Distance-dependent functions like Gaussian and Raised Cosine work well for different signal configurations
- the custom catheter signal is estimated with minimal error across heatmap configuration
- strength of the signal has minimal impact

*The influence of signal and heatmap width*
- perfect match between the signal and heatmap width does not benefit the representation quality during network inference
- smaller heatmap width is generally more beneficial, regardless of the signal configuration → advantageous for real life application (signal width varies or information about signal configuration is limited)
- rectangular distribution shows the largest error with increasing heatmap widths
- high width cause network to overestimate the signal intensity distribution (especially for Laplace, Triangular signal types)
- BCE optimization reduce a little this representation biais but does not change the error tendency

*The limitations by signal strength*
- signal strength significantly impact the quality of predicted heatmaps (except for rectangular signal shape)
- for weaker signals, heatmmaps suffer from increased fragmentation, frayed edges, and underestimated intensity values → especially for the wide and leptokurtic signals
- small deviations of the center-line during network inference can lead to much stronger mass shifts (affect Otsu threshold and squeletonization and so the quality of predictions)

## Application to 2D and 3D representation problems 
---
## Anatomiacal structure detection on knee radiographs (2D experiment)

analyze heatmap representations of four elongated anatomical features :
- (A) Blumensaat line (curved)
- (B) mean contour of the medial and lateral femoral condyles (curved)
- (C) plateau line of the proximal tibia (straight)
- (D) anatomical axis of the femur bone (straight)

A, B, C : directly observed (noticeable contrast) ⇒ direct anatomical structure

D relies on the global orientation of the bone ⇒ contextual anatomical structure

![](/collections/images/elongated_structures/fig_6.jpg)

![](/collections/images/elongated_structures/table_1.jpg)

- overall high spatial precision but F-measure slightly higher for direct perceivable feature
- lower individual F-measure attributed to anatomical deviations, slight misalignement of the femoral condyles, or joint deformities like bone bending
- relation between the optimal heatmap width σHM and the type of structure to be represented :
    - delicate structures : smaller width
    - larger and more pronounced structures : larger width
- close interpolation is preferable
    - especially for intricated structures with detailed curves,
    - negligible for larger structure
    - interestingly, for straight structures, if combined with other features, result in better detection performances
- heatmap representation achieve higher F-measures than segmentation and skeletonization methods (Dice and Dice+BCE loss significantly worst for D, low recall)

## Detection of surgical implants in 3D CBCT volumes

screw and k-wire detection crucial for guiding surgical intervention, ASSD evaluation

![](/collections/images/elongated_structures/fig_7.jpg)

- reliably detected with high spatial precision, even in complex cases (multiple implants close)
- compared with 2D exemles, the prediction quality is slightly lower
- test data with metal artifact reduction (MAR) + Gaussian reconstruction filter : median ASSD = 4.59 voxels CI95% [3.65, 7.43]
- high error cases are attributed to
    - false-positive response at metallic plates
    - a heatmap intersection of close neighboring objects
    - over-/underestimation of object length
    
    → often due to ambiguous ground truth in anatomically complex and cluttered regions
    
- test with no MAR but same filter : median ASSD = 5.36 voxels CI95%[3.79, 6.66]

&rarr; 3D heatmap representation can successfully overcome a reduction in image quality and that meaningful heatmap extrapolation in compromised areas is possible

# Conclusions

*Purpose*
- mathematical representation of curves, to be used as a target in optimization based algorithms
- involve generating heatmap by convoluting the spatial curve distribution with a filter function
- approximate the convolution by evaluating freely selectable distance-dependant function → this simplification allows parallelizable evaluation of the heatmap generation function

*Key findings and applications*
- the proposed heatmap representation can approximate various curves and signals (except extreme curvature close to 180°)
- Calculating the heatmap with a mesokurtic function and small or medium width is preferred for most curve types, especially delicate structures.
- The representation is robust, even in cases of degraded image quality, allowing extrapolation of missing information
- both directly visible information and context dependent structure can be approximated
- 3D application are promising but with slightly more error compared to 2D heatmap

*Advantages*
- heatmaps can incorporate prior knowledge of signal distribution and positional noise
- allow more precision representation of structures with known physical parameters
- allows for instance-level parameterization of multiple individual structures within the same output channel

*Drawback* 

- difficulty in modelling the visual appearance of a signal with varying width (naturally non-uniform signals, perspective changes due to out-of-plane components)
- inefficiency in handling dense accumulation of structure with potential overlap

*Challenges*
- no analysis has been conducted to find a method for reconstructing the original signal from the heatmap representation
- thinning approach gives good results in pixel/voxel space, but it is more difficult if the heatmap is fragmented or self-intersecting → even more difficult if several heatmap are close and approximated in the same output channel

*Perspectives*
- it would be helpful to separate structures into separate output channels (even if they can’t be matched between different images of volumes)
- using a neural network to approximate knots and B-spline curve approximation parameters for signal reconstruction is a promising idea

