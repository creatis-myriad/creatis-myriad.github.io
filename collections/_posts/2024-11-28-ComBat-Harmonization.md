---
 layout: review
 title: U-Net Convolutional Networks for Biomedical Image Segmentation
 tags: deep-learning CNN segmentation medical essentials
 cite:
     authors: "O. Ronneberger, P. Fischer, T. Brox"
     title:   "U-Net: Convolutional Networks for Biomedical Image Segmentation"
     venue:   "Proceedings of MICCAI 2015, p.234-241"
 pdf: "https://arxiv.org/pdf/1505.04597.pdf"
 ---

 # Introduction

 Famous 2D image segmentation CNN made of a series of convolutions and
 deconvolutions. The convolution feature maps are connected to the deconv maps of
 the same size. The network was tested on the 2 class 2D ISBI cell segmentation
 [dataset](http://www.codesolorzano.com/Challenges/CTC/Welcome.html).
 Used the crossentropy loss and a lot of data augmentation.

 The network architecture:

 A U-Net is based on Fully Convolutional Networks (FCNNs)[^1].

 The loss used is a cross-entropy:
 $$ E = \sum_{x \in \Omega} w(\bold{x}) \log (p_{l(\bold{x})}(\bold{x})) $$

 The U-Net architecture is used by many authors, and has been re-visited in
 many reviews, such as in [this one](https://vitalab.github.io/article/2019/05/02/MRIPulseSeqGANSynthesis.html).

 # References

 [^1]: Jonathan Long, Evan Shelhamer, and Trevor Darrell. Fully convolutional
       networks for semantic segmentation (2014). arXiv:1411.4038.
