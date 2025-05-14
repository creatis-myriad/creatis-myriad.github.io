---
layout: review
title: "Spiking Neural Networks"
tags: deep-learning
author: "Thierry Judge"
---

# Introduction

Artificial neural networks are the cornerstone of modern artificial intelligence algorithms. However, despite their impressive performances, 
they still have their respective shortcoming, notably the enormous power required to train and run inference. This has motivated research into spiking neural networks. 

Many reasons justify the interest in spiking neural networks, two of which are 

1. Biological plausibility
2. Temporal and event-driven processing, which allows for sparse and energy-efficient computation. 

The format for this review is based on the class [GEI723 - Neurosciences computationnelles et applications en traitement de l'information](https://www.usherbrooke.ca/admission/fiches-cours/GEI723/neurosciences-computationnelles-et-applications-en-traitement-de-linformation/) at Sherbrooke University. 
The goal of this review will be to introduce the anatomical and physiological concepts of biological neurons before explaining how neurons can be modeled by a computer. Finally, a paper will be presented showing one of many ways a spiking neural network can be trained. 

# Anatomy and physiology

### Neuron components

To appropriately model neurons in a computer, a certain number of concepts must be understood. This section will quickly introduce these concepts. A neuron, as depicted in the following figure, is composed of many parts such as 

- **Nucleus:**  Main part of the cell body.
- **Dendrites:** Part of the neuron that receives inputs from other neurons.
- **Axon:** Cable-like structure that carries the neuron signal.
- **Axon terminal:**  End of the neuron from which connections to other neurons are made (through neurotransmitter)

<div style="text-align:center">
<img src="/collections/images/spiking_neural_networks/Neuron.png" width=400></div>
<p style="text-align: center;font-style:italic">Figure 1. Illustration of a Neuron [^1] .</p>


### Membrane potential

The neuron is separated from the outside of the cell by the neuron membrane. This separation causes a difference in electrical potential between the inside and outside. The change in this potential is what allows neurons to transmit signals, as will be explained later. 

At rest, the membrane potential is around -70 mV. 

The number of ions inside and outside the cell membrane defines the membrane potential. The main ions that affect the membrane potential are 

- Potassium ($K^+$) : Positive ions, more prevalent on the inside of the cell. 
- Sodium ($Na^+$): Positive ions, more prevalent on the outside of the cell. 
- Calcium ($Cl^-$): Negative ions, more prevalent on the outside of the cell. 

<div style="text-align:center">
<img src="/collections/images/spiking_neural_networks/Basis_of_Membrane_Potential2-en.png" width=300></div>
<p style="text-align: center;font-style:italic">Figure 1. Illustration of a cell membrane [^2] .</p>

The membrane potential can be expressed mathematically with the  Goldman-Hodgkin-Katz (GHK) equation

$$
v_m = \frac{RT}{F} ln \frac{P_K [ K^{+} ]_{out} + P_{Na} [ Na^{+} ]_{out} + P_{Cl} [ Cl^{-} ]_{out}}{P_K [ K^{+} ]_{int} + P_{Na} [ Na^{+} ]_{int} + P_{Cl} [ Cl^{-} ]_{int}}
$$

where 
* $P_{ion}$ is the permiability of the membrane for a given ion.
* $\[ion\]_{out}$ is the concentration of an ion outside the neuron
* $\[ion\]_{in}$ is the concentration of an ion inside the neuron. 
- 

### Action potential

Due to an input stimulus, the membrane potential can change. This change in membrane potential affects the membrane's permeability to different ions. 
If the input stimulus is large enough causing the membrane potential to surpass the threshold (-55 mV) an action potential (spike) is generated. 

The increase in membrane voltage cause the permeability to sodium ions to increase leading to a sodium ions rushing into the cell further increasing the membrane potential.
At the peak of the action potential, the sodium permeability goes bac down and the potassium permeability increases causing an outflux of potassium ions causing the membrane potential to fall, below the resting potential. 


<div style="display: flex; justify-content: center; gap: 2rem;">
  <figure style="margin: 0; text-align: center;">
    <img src="/collections/images/spiking_neural_networks/Action_potential.png" width="250" alt="Neuron">
    <figcaption style="font-style: italic;">Figure 2a. Membrane potential during action potential [^2]</figcaption>
  </figure>
  <figure style="margin: 0; text-align: center;">
    <img src="/collections/images/spiking_neural_networks/neuronal_action_potential_timecourse_of_pna_and_pk.jpg" width="400" alt="Synapse">
    <figcaption style="font-style: italic;">Figure 2b. Membrane permeability during action potential [^3]. </figcaption>
  </figure>
</div>

The action potential is followed by a refractory period. Two types of refractory periods exist:

- The absolute refractory period during which the neuron physically cannot fire.
- The relative refractory period during which the neuron can fire given a large enough stimulus.

## Synapses

Synapses connect neurons. When a synapse connects two neurons we refer to the first neuron as the pre-synaptic neuron and the second neuron as the post-synaptic neuron. 
Synapse can be either excitatory or inhibitory causing an increase or decrease the post-synatic neuron activity. 

There are two main types of synapses: chemical and electric.

- Chemical: Electrical activity in the pre-synaptic releases neurotransmitter that can either excite or inhibit the post-synaptic neuron.
- Electric: Special channels between pre- and post-synaptic neurons allow the changes in voltage of the pre-synaptic neuron to induce a change in the post-synaptic neuron.

## Learning

It is mostly considered that learning in the brain occurs when synapse modified du to pre-synatptic stimulation of post-synatomic cells. This process is know as Hebbian learning.
Spike-timing-dependent plasticity (STDP) is the process that adjust the weights of synapse according to the timing of pre-synaptic and post-synaptic spikes.

Different paradigms of STDP exist but the most common is the following: 
* Synaptic weights increase when the pre-synaptic spike induces post-synaptic spike.
* Synaptic weights decrease when the  post-synaptic neuron spikes before the pre-synaptic neuron. 

The size of the weight change is dependant of the time between the two spikes. 
<div style="text-align:center">
<img src="/collections/images/spiking_neural_networks/STDP.jpg" width=400></div>
<p style="text-align: center;font-style:italic">Example of STDP [^6] .</p>


# Modeling Spiking Neural networks

## Neuron models 
Many mathemcatical models have been proposed to represent the behavior of a neuron. Often, these models have to balance biological plausibility and computational complexity. 

<div style="text-align:center">
<img src="/collections/images/spiking_neural_networks/models.png" width=400></div>
<p style="text-align: center;font-style:italic">Figure 1. Neuron models and their relative complexities [^5] .</p>

The most realistic, but also most complex, model is the Hodgkin–Huxley. The model is made of several differential equations representing the neuron potential according to inputs. 

The simplest way to model a neuron is with the integrate and fire (IF). This neuron accumulates voltage from its input and when the voltage reaches a threshold it fires a spike. The voltage is reset to the resting voltage and the neuron cannot fire during a refractory period. A *leaky* version known as the leaky integrate and fire (LIF) loses voltage according to a differentiable equation. 

## Input encoding 

To interact with mathematical models of neurons, we must transmit data to them. To do so, we must encode the data in a way the neurons can receive it. This is done by input neurons which spike according to different encoding schemes. The two most popular schemes are **time to spike** and **rate coding.**

- Time to spike coding: The neuron spikes once with a delay determined by the input amplitude.
- Rate code: the neuron spikes at a frequency determined by the input amplitude (with optional stochasticity)

<div style="text-align:center">
<img src="/collections/images/spiking_neural_networks/encoding.jpg" width=500></div>
<p style="text-align: center;font-style:italic">Example of input encodings.</p>


## Example of neuron function and interaction:

Here are examples of neuron models using the Python package *Brain2.* 

All neurons have a threshold of 1V and a reset voltage of -0.5V. 

<div style="text-align:center">
<img src="/collections/images/spiking_neural_networks/brain_ex1.png" width=400></div>

We can add a second neuron with a synapse. 

<div style="text-align:center">
<img src="/collections/images/spiking_neural_networks/brain_ex2.png" width=400></div>

We can add a second neuron with a synapse. 

<div style="text-align:center">
<img src="/collections/images/spiking_neural_networks/brain_ex3.png" width=400></div>

```python

import numpy as np
from brian2 import *
import matplotlib.pyplot as plt

# Define the equation for the neuron (LIF)
eqs = '''
	dv/dt = (I-v)/tau : 1
	I : 1
	tau : second
'''
# Define the neurons.
G = NeuronGroup(3, eqs, threshold='v>1', reset='v = -0.1', method='euler')
G.I = [2, 0, 2]
G.tau = [10, 100, 50]*ms

# Comment these two lines out to see what happens without Synapses
S = Synapses(G, G, on_pre='v_post += 0.2')
S.connect(i=0, j=1)

# Comment these three lines to remove the inhibitor synapse. 
S2 = Synapses(G, G, on_pre='v_post -= 0.5')
S2.connect(i=2, j=1)
S2.delay = '2*ms'

M = StateMonitor(G, 'v', record=True)

run(100*ms) # Run the simulation for 100 ms. 

plot(M.t/ms, M.v[0], label='Neuron 0')
plot(M.t/ms, M.v[1], label='Neuron 1')
plot(M.t/ms, M.v[2], label='Neuron 2')
xlabel('Time (ms)')
ylabel('v')
legend();
```

# Learning with Spiking Neural Networks - Article Review 

To better understand how all this can be applied, the rest of this review will be dedicated to the review of a paper. While spiking neural networks can be used in many ways, this paper proposes a simple method for training a spiking neural network. 

> Diehl, P., & Cook, M. (2015). *Unsupervised learning of digit recognition using spike-timing-dependent plasticity*. Frontiers in Computational Neuroscience, 9.


## Introduction 

In this paper, the authors propose a method to do *unsupervised* MNIST classification with a spiking neural network training with STDP. 


## Method 

The method has many components, the authors tried to keep the 

### Neuron model 

$$
\tau \frac{dV}{dt} = (E_{rest} - V) + g_e(E_{exc} - V) + g_i(E_{inh}- V)
$$

where

- $E_{rest}$ is the resting membrane potential
- $E_{exc}$ and $E_{inh}$ are the equilibrium potentials of excitatory and inhibitory synapses,
- $g_e$ and $g_i$ the resting membrane potential

### Synapse model

Instead of directly modifying the 

### Network architecture 

### Learning 

The network consists of 3 neuron layers

1. The input layer contains 784 (28 x 28) neurons, one for each pixel in the input image. These neurons encode the pixel values into spikes using a Frequency encoding. 
2. The second layer contains N excitatory neurons.
3. The third layer contains N inhibitory neurons. 

The synapse in the neuron are shown in the following figure (network with 3 input neurons and 4 excitatory/inhibitory neurons). 

1. All input neurons are connected to each excitatory neuron. 
2. The excitatory neurons are connected to the inhibitory in a one-to-one fashion.
3. The inhibitory neurons are connected to all excitatory except the one with which it already has a connection. 



 **Lateral inhibition**
 

### Input encoding 


## Results 



# References 

[^1] https://commons.wikimedia.org/wiki/File:Neuron.svg
[^2] https://commons.wikimedia.org/wiki/File:Basis_of_Membrane_Potential2-en.svg
[^2] https://commons.wikimedia.org/wiki/File:Action_potential.svg
[^3] https://www.physiologyweb.com/lecture_notes/neuronal_action_potential/figs/neuronal_action_potential_timecourse_of_pna_and_pk_jpg_zgn9dUl70MnJPf2CqaZ5uoL3BBD6r9fW.html

[^4] https://brian2.readthedocs.io/en/stable/

[^5] https://www.izhikevich.org/publications/whichmod.pdf
[^6] https://commons.wikimedia.org/wiki/File:STDP-Fig1.jpg