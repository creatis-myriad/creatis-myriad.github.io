---
layout: review
title: "Gradient Descent: The Ultimate Optimizer"
tags: automatic differentiation, differentiable programming, hyperparameter optimization
author: "Lucas Braz"
cite:
    authors: "K. Chandra, A. Xie, J. Ragan-Kelley, E. Meijer"
    title:   "Gradient Descent: The Ultimate Optimizer"
    venue:   "Proceedings of NeurIPS 2022"
pdf: "https://arxiv.org/pdf/1909.13371.pdf"
---

# Useful links

- The article includes a [PyTorch implementation of the algorithm](people.csail.mit.edu/kach/gradient-descent-the-ultimate-optimizer)

# Highlights

- Key idea: A method to efficiently and automatically tune the *hyperparameters* of gradient-based optimization algorithms used in machine learning using *automatic differentiation* (AD).
- The method involves using backpropagation to compute *hypergradients*, which are the gradients of the optimization algorithm's hyperparameters with respect to the loss function being optimized.
- The method is efficient and can be applied to a wide range of optimization algorithms, including SGD, Adam, RMSProp, and others.
- The article also describes how this method can be applied recursively, allowing for the optimization of higher-level *hyper-hyper parameters* (even *hyper-...-hyper-hyperparameters*).

# Introduction

When we train a Machine Learning model by Gradient Descent, we have to select a learning rate $$\alpha$$ (and often some other hyperparameters) for the optimizer. If $$\alpha$$ is too low, the optimizer will runs too slowly but if $$\alpha$$ is too high the algorithm will never converge.    

A first approach to optimize $$\alpha$$ would be a basic grid search to find the best value. It's a common approach but actually a bit strange to use this basic approach while being optimizing some complex neural networks by Gradient Descent.

That's why the idea to compute the derivative of the loss function with respect to $$\alpha$$ in order to optimize it came, which is called a *hypergradient*.

Until now, to use this approach, it requires to manually do the calculus. There is three limitations:

- Manually differentiation optimizer update rule is error-prone and must be re-done for each optimizer variant
- It only tunes the learning rate hyperparameter, not for example the momentum coefficient.
- By doing a hypergradient descent, you introduced a hyper-learning rate which must also be tuned.

This paper introduce an *automatic differentiation* method which:

- automatically compute the derivative
- naturally generalizes to other hyper-parameters
- Can be applied recursively to hyper-hyperparameters and hyper-...-hyper-hyperparameters.

By using it recursively, the model become way more robust to the choice of the initial hyperparameter.

# Methods

We consider some loss function $$f$$ that we we want to minimize using gradient descent.
Let $$w_i$$ be the weights at beginning of step $$i$$.

We recall that, with a learning rate $$\alpha$$, the update rule of a SGD is:

$$w_{i+1}=w_i-\alpha\frac{\partial f(w_i)}{\partial w_i}$$

We now want to optimize $$\alpha$$ at each step so let $$\alpha_i$$ be the learning rate at the beginning of step $$i$$.

We will then want to update $$\alpha_i$$ to $$\alpha_{i+1}$$ using the right update rule and then use $$\alpha_{i+1}$$ as the learning rate to update from $$w_i$$ to $$w_{i+1}$$.

By analogy to $$w$$, we will adjust $$\alpha$$ with a SGD-like update rule, adding a new hyperparameter for the hyper-learning rate named $$\kappa$$:

$$\alpha_{i+1}=\alpha_i-\kappa\frac{\partial f(w_i)}{\partial \alpha_i}$$

and then update $$w$$ as follows:

$$w_{i+1}=w_i-\alpha_{i+1}\frac{\partial f(w_i)}{\partial w_i}$$

## Manual computation of the hypergradient

In the case of the SGD we can easily manually compute $$\frac{\partial f(w_i)}{\partial \alpha_i}$$.

Using the chain rule:

$$\frac{\partial f(w_i)}{\partial \alpha_i}=\frac{\partial f(w_i)}{\partial w_i}\cdot\frac{\partial w_i}{\partial \alpha_i}=\frac{\partial f(w_i)}{\partial w_i}\cdot\frac{\partial (w_{i-1}-\alpha_i\frac{\partial f(w_{i-1})}{\partial w_{i-1}})}{\partial \alpha_i}=\frac{\partial f(w_i)}{\partial w_i}\cdot(-\frac{\partial f(w_{i-1})}{\partial w_{i-1}})$$

The last equality is possible because $$w_{i-1}$$ and $$f(w_{i-1})$$ (and so $$\frac{\partial f(w_{i-1})}{\partial w_{i-1}}$$) do not depend on $$\alpha_i$$ (detached).

In the case of the Adam Optimizer, I will not write the expression here but it is significantly more complex for the hypergradients of $$\alpha$$,$$\beta_1$$,$$\beta_2$$ and $$\epsilon$$.

This manual method does not scale.

## Reverse-mode Automatic Differentiation

The idea of differentiable programming is to build a computation graph as the function is computed forwardly. Each leaves node are $$w_i$$, internal nodes are intermediate computations and root is the final loss. During backpropagation, it then compute gradient in each internal node starting from the node and going through the graph until the $$w_i$$. We can then update $$w$$.

We then *detach* the weights from the computation graph before the next iteration of the algorithm to not let the backpropagation during iteration i+1 atteign weights of iteration i.

for the SGD:

```py
def SGD.__init__(self, alpha):
    self.alpha = alpha

def SGD.step(w):
    d_w = w.grad.detach()
    w = w.detach() - self.alpha.detach() * d_w
```

Now, we can simply apply the same method to the hyperparameter $$\alpha$$ before to update $$w$$ without detaching $$\alpha$$ as described in the previous section.

```py
def HyperSGD.step(w):
    # update alpha using the alpha update rule for SGD
    d_alpha = self.alpha.grad.detach()
    self.alpha = self.alpha.detach() - kappa.detach() * d_alpha

    # update w using equation (2)
    d_w = w.grad.detach()
    w = w.detach() - self.alpha * d_w # not -self.alpha.detach()*d_w
    # as it is already detached and will stay attached for next iteration
```

This can be rewritten as:

```py
def HyperSGD.__init__(self, alpha, opt):
    self.alpha = alpha
    self.optimizer = opt # for example SGD(kappa)

def HyperSGD.step(w):
    self.optimizer.step(self.alpha)

    d_w = w.grad.detach()
    w = w.detach() - self.alpha * d_w
```

The next figure shows the computation graph for the SGD with fixed $$\alpha$$ on the left and the HyperSGD on the right:

![graph-cut](/collections/images/GradientDescent-TheUltimateOptimizer/figure1.jpg)

Since the computation graph is extended with not so many nodes, the method is not really slower.

## Generalization to other hyperparameters

This method generalize quite easily to tune hyperparameters of other optimizers as Adam. We only need to change the last line with the new update rule. In case of Adam which have 4 hyperparameters, we treat each of them simultaneously with the same method as for $$\alpha$$.

There is just two things to note for Adam:

- for $$\beta_1$$ and $$\beta_2$$, we need to clamp them to be in the range $$[0,1]$$ using a sigmoid function.
- Adam involves a square root which is not differentiable at 0 for a term $$v̂_i$$ wich is normally initialized to 0. We can fix this by initializing it to a small value $$\epsilon$$.

## Stacking hyperoptimizers recursively

Using the method described above, we can stack hyperoptimizers recursively. For example, we can use a HyperSGD to tune the learning rate of a SGD and then use a HyperSGD to tune the learning rate of the HyperSGD. Using code above, this can be written as:

```py
HyperSGD(0.01, HyperSGD(0.01, SGD(0.01)))
```

# Results

They firstly conducted their experiments on MNIST using a fully connected network with one hidden layer of size 128, tanh activations and a batch size of 256 train for 30 epochs. They report statistics over 3 runs.

**Hyperoptimizer SGD outperforms the baseline SGD by a significant margin.** It holds even with another hyperoptimizer to optimize the learning rate of the SGD. Use the learned learning rate is also better than the initial one. Same with Adam.

![table1](/collections/images/GradientDescent-TheUltimateOptimizer/table1.png)

## Hyperoptimization at scale for Object Recognition

Using their method for a Computer vision task (ResNet-20 on the CIFAR-10 dataset). They vary the momentum $$ \mu $$ and the learning rate $$ \alpha $$ from too small to too large values. They compare results between the baseline, their hyperoptimizer to tune both $$ \mu $$ and $$ \alpha $$ and an hand-engineered learning rate decay schedule from He et al. (2015).

![curves1](/collections/images/GradientDescent-TheUltimateOptimizer/figure2.jpg)

They conclude that hyperoptimizers are indeed beneficial for
tuning both step size and momentum in this real-world setting and match the results of the hand-engineered learning rate decay schedule.

## Higher order hyperoptimization

We can observe the test error of the hyperoptimizer depending of the order of the hyperoptimization. It appears that the method become more stable as order increase.
We can also observe the runtime cost of higher order hyperoptimization. **The method is efficient and scale linearly**.

![curves2](/collections/images/GradientDescent-TheUltimateOptimizer/figure3.jpg)

# Conclusions

This paper introduce an innovative and elegant method to automatically tune hyperparameters of an optimizer without manual differentiation, using automatic differentiation. The method seems to be simple to implement and can be applied to any optimizer with sometimes some little fixes and is also easy to extend to other hyperparameters. The method is also efficient and can be stacked recursively.
