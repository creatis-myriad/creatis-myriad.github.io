---
layout: review
title: Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results
tags: machine-learning, neural and evolutionary computing
author: Tarvainen Antti, Valpola Harri
cite:
    authors: "Tarvainen et Valpola"
    title:   "Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results"
    venue:   "Neural Information Processing Systems 30 (NIPS 2017) pre-proceedings"
pdf: https://arxiv.org/pdf/1703.01780.pdf
---

# Highlights

- Method : The mean teachers model **averages model weigths**, instead of label predictions like the Temporal Ensembling method. Allows for more accurate targets and faster feedback between the student and teacher models.
- Performances : demonstrates **enhanced performance on various datasets** (SVHN, CIFAR-10, and ImageNet) when combined with residual networks. Achieves lower error rates and higher accuracy compared to Temporal Ensembling, even when trained with fewer labeled examples.
- Scalability and applicability : Enables working on **larger dataset**, and training with **fewer labeled examples**.

# Introduction
Limitations of deep learning models :
- requires large number of parameters and is prone to overfitting. 
- manual high quality labels for training data is expensive and time consuming.

Interest in semi-supervised learning to use the unlabeled data effectively. 

&rarr; focus on **regularization methods** in **semi-supervised learning**

# Method 

## Apllying noise and Esembling method

### **Noise regularization** : 
- Add noise to the input or intermediate representations of a model
- Helps learning abstract invariances
- **Pushes decision boundaries away from labeled data points**  

Limitations of noise regularization :
- classification cost is **undefined for unlabeled examples** &rarr; not helpful in semi-supervised learning.

&rarr; *The $\Gamma$ model to overcome this*

### **The $\Gamma$ model [1]**
- Evaluates each data point with and without noise, and then applies a consistency cost between the two predictions. 
- Student teacher models where the **teacher generates targets**, which are then **used by itself as a student for learning**

Limitations of $\Gamma$ model :
- the model itself generates targets, which may be incorrect
- the cost of inconsistency outweighs that of misclassification, preventing the learning of new information
- can lead to a **confirmation bias**

&rarr; *Confirmation biais can be mitigated by improving the targets. One way is illustrated with the $\Pi$ model, selecting a different teacher model than the student model*

> **NOTE :** The authors mention another way : "to choose the perturbation
of the representations carefully instead of barely applying additive or multiplicative noise" wich is the subject of another paper of [Miyato et al. 2017](https://doi.org/10.48550/arXiv.1704.03976)[2] that will be refered as VAT+EntMin latter. 

### **The $\Pi$ model [3]**
- **noise** is added to the model **during inference**
- result in a "noisy teacher" that can provide **more accurate targets**. 

&rarr; *The Π model can be further improved by Temporal Ensembling (TE)*

### **Temporal Ensembling (TE)**
- **exponential moving average (EMA)** prediction for each training example is formed by combining the **current version of the model's predictions** with the **predictions made by earlier versions of the model** that evaluated the same example. 
- improves the quality of the prediction 
- better results when those predictions are used as the teacher predictions

Limitation of Temporal ensembling :
- Each **target** (prediction) is **updated only once per training epoch**.
- The learned information is **incorporated** into the training process at a relatively **slow pace**.
- The **larger the dataset**, the **longer it takes** for the updates to span the entire dataset.

&rarr; *The authors propose the Mean Teacher model that **averages model weights**, instead of predictions like TE, to improve performances of semi-supervised model.*

![](/collections/images/mean_teachers/binary_classification_2.png)

A sketch of a binary classification task with two labeled examples (large black dots) and one unlabeled example, demonstrating how the choice of the unlabeled target (blue circle) affects the fitted function (gray curve).

(a) A model with no regularization &rarr; free to fit any function that predicts the labeled training examples well  
(b) A model trained with noisy labeled data (small dots) &rarr; consistent predictions around labeled data points  
(c) The teacher model (gray curve) is first fitted to the labeled examples, and then left unchanged during the training of the student model. &rarr; Consistency to noise around unlabeled examples provides additional smoothing  
(d) Noise on the teacher model reduces the bias of the targets without additional training. &rarr; The expected direction of stochastic gradient descent is towards the mean (large blue circle) of individual noisy targets (small blue circles)  
(e) Ensemble model &rarr; gives an even better expected target

---
## Mean Teacher 

The Mean Teacher is an average of consecutive student models.

![](/collections/images/mean_teachers/mean_teacher_method.png)


- Both the student and the teacher model **evaluate the input applying noise (η, η')**
- The sotmax output is compared with one-hot label using **classification cost** for the student model and **consistency cost** for the teacher 
- The teacher model weight are upadted using the **EMA weights of the student model**.
- Both prediction can be used but the **teacher prediction if more likely to be correct**
- No classification cost is applied when it is a training step with unlabeled exemple. 

### Concistency cost 

With $J$ the consistency cost, as the expected distance between the prediction of the student model (with weights $θ$ and noise $η$) and the prediction of the teacher model (with weights $θ'$ and noise $η'$) :

$$ J(θ) = \underset{x,η',η}{E} [ \| f (x, θ', η') - f (x, θ, η) \|^2 ] $$

- Mean squared error as the consistency cost and ramp up its weight from 0 to its final value during the first 80 epochs.
 &rarr; Initially ignoring it and then gradually introducing more and more consistency. 

### Difference between the $\Pi$ model, Temporal Ensembling, and Mean teacher 

- How the teacher predictions are generated :
    - $\Pi$ model : $θ' = θ$  
    - TE : approximates $f (x, θ', η')$ with a **weighted average of successive predictions**
    - Mean Teacher defines $θ'$<sub>t</sub> at training step as the **EMA of successive $\theta$ weights**, where $\alpha$ is a smoothing coefficient hyperparameter : 

$$ θ'_t = αθ'_{t-1} + (1 - α)θ_t $$

- Weights $\theta'$ :
    - $\Pi$ model : applies training to it
    - TE and Mean teacher : treat it as a constant with regards to **optimization**

### Type of noise 
- Random translations and horizontal flips of the input images
- Gaussian noise on the input layer
- Dropout applied within the network

### Loss
- Student : cross-entropy loss 
- Teacher : consitency loss multiply by an importance weight 

## Data
**Street View House Numbers (SVHN)** : 
- 32x32 pixel RGB images belonging to ten different classes :
- close-up of a house number, and the class represents the identity of the digit at the center. 
- 73257 training samples and 26032 test samples.

![](/collections/images/mean_teachers/SVHN.png)

**CIFAR-10** : 
- 32x32 pixel RGB images belonging to ten different classes :
- natural image belonging to a class such as horses, cats,
cars and airplanes etc
- 50000 training samples and 10000 test samples.

![](/collections/images/mean_teachers/CIFAR-10.png)

# Results

## Comparisons of the state-of-the-art methods : 

![](/collections/images/mean_teachers/table_1.png)

**Mean Teacher improves test accuracy** over the $\Pi$ model and Temporal Ensembling on semi-supervised SVHN tasks. 

![](/collections/images/mean_teachers/table_2.png)

Mean Teacher also improves results on CIFAR-10 over our baseline $\Pi$ model.

> **NOTE :** The Virtual Adversarial Training (VAT) by [Miyato et al. 2017](https://doi.org/10.48550/arXiv.1704.03976)[2] performs even better than Mean Teacher on the **1000-label SVHN** and the **4000-label CIFAR-10**. VAT and Mean Teacher are complimentary approaches.  
&rarr; Their combination may yield better accuracy than either of them alone.

## Error percentage over 10 runs on SVHN with extra unlabeled training data :

![](/collections/images/mean_teachers/table_3.png)

## Effect of using mean teacher : 

![](/collections/images/mean_teachers/figure_3.png)

- the **EMA-weighted models** (blue and dark gray curves in the bottom row) give **more accurate predictions** than the bare student models (orange and light gray) after an initial period. 
 
    &rarr; EMA-weighted model as the teacher **improves results** in the semi-supervised settings

- When using **500 labels** **Mean Teacher** **learns faster**, and continues training after the $\Pi$ model stops improving. 

    &rarr; **Mean Teacher uses unlabeled training data more efficiently than the $\Pi$ model**

- In the **all-labeled case**, Mean Teacher and the $\Pi$ model **behave identically**. With 500k extra unlabeled examples $\Pi$ model keeps improving for longer.

     &rarr; **Mean Teacher learns faster**, and eventually **converges to a better result**, but the sheer amount of data appears to offset $\Pi$ model’s worse predictions.

## Mean Teacher with residual networks on CIFAR-10 and ImageNet 

Checking if the methods scales to more natural images on ImageNet with 10% of the labels : 

![](/collections/images/mean_teachers/table_4.png)

# Conclusions

Advantages over Temporal Esembling :
- **more accurate target labels**, which enables **faster feedback loop** between the student and the teacher model
- scales to **large datasets** and online learning.

To conclude : 
- The success of consistency regularization depends on the quality of teacher-generated targets.
- Mean Teacher and Virtual Adversarial Training represent
two ways of exploiting this principle.

    &rarr; Their combination may yield even better targets. 

# References
[1]: Rasmus, Antti, Berglund, Mathias, Honkala, Mikko, Valpola, Harri, and Raiko, Tapani. Semi-supervised Learning with Ladder Networks. In Cortes, C., Lawrence, N. D., Lee, D. D., Sugiyama, M., and Garnett, R. (eds.), Advances in Neural Information Processing Systems 28, pp. 3546–3554. Curran Associates, Inc., 2015  

[2]: Miyato, Takeru, Maeda, Shin-ichi, Koyama, Masanori, and Ishii, Shin. Virtual Adversarial Training: a Regularization Method for Supervised and Semi-supervised Learning. arXiv:1704.03976 [cs, stat], April 2017. arXiv: 1704.03976.

[3] Laine, Samuli and Aila, Timo. Temporal Ensembling for Semi-Supervised Learning. arXiv:1610.02242 [cs], October 2016. arXiv: 1610.02242.