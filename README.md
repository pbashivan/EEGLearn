# EEGLearn
A set of functions for supervised feature learning/classification of mental states from EEG based on "EEG images".
This code can be used to construct sequence of images (EEG movie snippets) from ongoing EEG activities and to classify between different cognitive states through recurrent-convolutional neural
nets. More generally it could be used to discover patterns in multi-channel timeseries recordings with known spatial relationship between sensors. Each color channel in the output image can contains values for a specific features computed for all sensors within a time window.

![alt text](diagram.png "Converting EEG recordings to movie snippets")

# Installation and Dependencies
In order to run this code you need to install the following modules:

Numpy and Scipy (http://www.scipy.org/install.html)

Scikit-Learn (http://scikit-learn.org/stable/install.html)

Theano (http://deeplearning.net/software/theano/install.html)

Lasagne (http://lasagne.readthedocs.org/en/latest/user/installation.html)

`pip install -r requirements.txt`

`pip install [path_to_EEGLearn]`

You can use the package in your code:
```
import eeglearn
import eeglearn.eeg_cnn_lib as eeglib

images = eeglib.gen_images(locs, features, nGridPoints)
eeglib.train(images, labels, train_test_fold, model_type)
```

# Some Notes:
1. When using the images to train a neural network, in many cases it is helpful to scale the values in the images to a symmetric range like `[-0.5, 0.5]`.
2. Images generated with `gen_images` function appear in the center of the field with unused space around them. This causes edges to appear around the images. To get around this, an edgeless option was added to gen_images function but I never systematically tried it to evaluate potential gains in performance.

Tensorflow implementation can be found here (thanks to @YangWangsky):

https://github.com/YangWangsky/tf_EEGLearn

PyTorch implementation (thanks to @VDelv):

https://github.com/VDelv/EEGLearn-Pytorch

# Reference
If you are using this code please cite our paper.

Bashivan, et al. "Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks." International conference on learning representations (2016).

http://arxiv.org/abs/1511.06448
