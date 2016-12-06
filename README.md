# EEGLearn
A set of functions for supervised feature learning/classification of mental states from EEG based on "EEG images" idea.
This code can be used to construct sequence of images from ongoing EEG activities and to classify between different cognitive states through recurrent-convolutional neural
nets.

# Dependencies
In order to run this code you need to install the following modules:

Numpy and Scipy (http://www.scipy.org/install.html)

Scikit-Learn (http://scikit-learn.org/stable/install.html)

Theano (http://deeplearning.net/software/theano/install.html)

Lasagne (http://lasagne.readthedocs.org/en/latest/user/installation.html)

# Some Notes:
1. When using the images to train a neural network, in most cases it is helpful to scale the values in the images (0-255) in a symmetric range like `[-0.5, 0.5]`.
2. Images generated with `gen_images` function appear in the center of the field with unused space around them. This causes edges to appear around the images. To get around this, an edgeless option was added to gen_images function but I never systematically tried it to evaluate potential gains in performance.  

#Reference

If you are using this code please cite our paper.

Bashivan, et al. "Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks." International conference on learning representations (2016).

http://arxiv.org/abs/1511.06448
