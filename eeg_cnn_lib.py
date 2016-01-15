# from __future__ import print_function
import time

import numpy as np
np.random.seed(1234)

import math as m

import scipy.io
import theano
import theano.tensor as T

from scipy.interpolate import griddata
from scipy.misc import bytescale
from sklearn.preprocessing import scale
from utils import augment_EEG, cart2sph, pol2cart

import lasagne
# from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Conv2DLayer, MaxPool2DLayer, InputLayer
from lasagne.layers import DenseLayer, ElemwiseMergeLayer, FlattenLayer
from lasagne.layers import ConcatLayer, ReshapeLayer, get_output_shape
from lasagne.layers import Conv1DLayer, DimshuffleLayer, LSTMLayer, SliceLayer


def azim_proj(pos):
    """
    Computes the Azimuthal Equidistant Projection of input point in 3D Cartesian Coordinates.
    :param pos: position in 3D Cartesian coordinates
    :return: projected coordinates using Azimuthal Equidistant Projection
    """
    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
    return pol2cart(az, m.pi / 2 - elev)


# Imagine a plane being placed against (tangent to) a globe. If
# a light source inside the globe projects the graticule onto
# the plane the result would be a planar, or azimuthal, map
# projection.

def gen_images(loc_filename, features_filename, nGridPoints,
               augment=False, pca=False, stdMult=0.1, n_components=2):
    """
    Generates EEG images given electrode locations in 2D space and
    :param loc_filename: Address of the MAT file containing electrode coordinates.
                        An array with shape [n_electrodes, 2] containing X, Y
                        coordinates for each electrode.
    :param features_filename: Address of the MAT file containing features as columns and
                                last column as labels.
    :param nGridPoints: Number of pixels in the output images
    :param augment:     Flag for generating augmented images
    :param pca:         Flag for PCA based data augmentation
    :param stdMult:     Standard deviation of noise for augmentation
    :param n_components:Number of components in PCA to retain for augmentation
    :return:            Tensor of size [samples, colors, W, H] containing generated
                        images.
    """
    # Load electrode locations projected on a 2D surface
    mat = scipy.io.loadmat(loc_filename)
    locs = mat['proj'] * [1, -1]    # reverse the Y axis to have the front on the top
    nElectrodes = locs.shape[0]     # Number of electrodes

    # Load feature values
    mat = scipy.io.loadmat(features_filename)
    data = mat['features']
    feat_array_temp = []

    # Test whether the feature vector length is divisible by number of electrodes (last column is the labels)
    assert data.shape[1] % nElectrodes == 1
    n_colors = data.shape[1] / nElectrodes
    for c in range(n_colors):
        feat_array_temp.append(data[:, c * nElectrodes : nElectrodes * (c+1)])


    if augment:
        if pca:
            for c in range(n_colors):
                feat_array_temp[c] = augment_EEG(feat_array_temp[c], stdMult, pca=True, n_components=n_components)
        else:
            for c in range(n_colors):
                feat_array_temp[c] = augment_EEG(feat_array_temp[c], stdMult, pca=False, n_components=n_components)

    labels = data[:, -1]
    nSamples = data.shape[0]
    # Interpolate the values
    grid_x, grid_y = np.mgrid[
                     min(locs[:, 0]):max(locs[:, 0]):nGridPoints*1j,
                     min(locs[:, 1]):max(locs[:, 1]):nGridPoints*1j
                     ]
    temp_interp = []
    for c in range(n_colors):
        temp_interp.append(np.zeros([nSamples, nGridPoints, nGridPoints]))

    for i in xrange(nSamples):
        for c in range(n_colors):
            temp_interp[c][i, :, :] = griddata(locs, feat_array_temp[c][i, :], (grid_x, grid_y),
                                    method='cubic', fill_value=np.nan)
        print 'Interpolating {0}/{1}\r'.format(i+1, nSamples),

    for c in range(n_colors):
        temp_interp[c][~np.isnan(temp_interp[c])] = \
            scale(temp_interp[c][~np.isnan(temp_interp[c])])
        temp_interp[c] = np.nan_to_num(temp_interp[c])

    return np.swapaxes(np.asarray(temp_interp), 0, 1)     # swap axes to have [samples, colors, W, H]


def build_cnn(input_var=None, W_init=None, imSize=32):
    """
    Builds a VGG style CNN network followed by a fully-connected layer and a softmax layer.
    input_var: Theano variable for input to the network
    outputs: pointer to the output of the last layer of network (softmax)
    :param input_var: theano variable as input to the network
    :param W_init: Initial weight values
    :param imSize: Size of the image
    :return: a pointer to the output of last layer
    """

    weights = []        # Keeps the weights for all layers
    count = 0
    # If no initial weight is given, initialize with GlorotUniform
    if W_init is None:
        W_init = [lasagne.init.GlorotUniform()] * 7

    # Input layer
    network = InputLayer(shape=(None, 3, imSize, imSize),
                                        input_var=input_var)

    # CNN Stack 1
    network = Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
                          W=W_init[count], pad='same')
    count += 1
    weights.append(network.W)
    network = Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
                          W=W_init[count], pad='same')
    count += 1
    weights.append(network.W)
    network = Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
                          W=W_init[count], pad='same')
    count += 1
    weights.append(network.W)
    network = Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
                          W=W_init[count], pad='same')
    count += 1
    weights.append(network.W)
    network = MaxPool2DLayer(network, pool_size=(2, 2))

    # CNN Stack 2
    network = Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                          W=W_init[count], pad='same')
    count += 1
    weights.append(network.W)
    network = Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                          W=W_init[count], pad='same')
    count += 1
    weights.append(network.W)
    network = MaxPool2DLayer(network, pool_size=(2, 2))

    # CNN Stack 3
    network = Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                          W=W_init[count], pad='same')
    count += 1
    weights.append(network.W)
    network = MaxPool2DLayer(network, pool_size=(2, 2))

    return network, weights


def build_convpool_max(input_vars, numTimeWin, nb_classes):
    """
    Builds the complete network with maxpooling layer in time.
    :param input_vars: list of EEG images (one image per time window)
    :param numTimeWin: number of time windows
    :param nb_classes: number of classes
    :return: a pointer to the output of last layer
    """
    convnets = []
    W_init = None

    # Build 7 parallel CNNs with shared weights
    for i in range(numTimeWin):
        if i == 0:
            convnet, W_init = build_cnn(input_vars[i])
        else:
            convnet, _ = build_cnn(input_vars[i], W_init)
        convnets.append(convnet)
    # convpooling using Max pooling over frames
    convpool = ElemwiseMergeLayer(convnets, theano.tensor.maximum)
    # A fully-connected layer of 512 units with 50% dropout on its inputs:
    convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5),
            num_units=512, nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the output layer with 50% dropout on its inputs:
    convpool = lasagne.layers.DenseLayer(lasagne.layers.dropout(convpool, p=.5),
            num_units=nb_classes, nonlinearity=lasagne.nonlinearities.softmax)
    return convpool

def build_convpool_conv1d(input_vars, numTimeWin, nb_classes):
    """
    Builds the complete network with 1D-conv layer to integrate time from sequences of EEG images.
    :param input_vars: list of EEG images (one image per time window)
    :param numTimeWin: number of time windows
    :param nb_classes: number of classes
    :return: a pointer to the output of last layer
    """
    convnets = []
    W_init = None
    # Build 7 parallel CNNs with shared weights
    for i in range(numTimeWin):
        if i == 0:
            convnet, W_init = build_cnn(input_vars[i])
        else:
            convnet, _ = build_cnn(input_vars[i], W_init)
        convnets.append(FlattenLayer(convnet))
    # at this point convnets shape is [numTimeWin][n_samples, features]
    # we want the shape to be [n_samples, features, numTimeWin]
    convpool = ConcatLayer(convnets)

    convpool = ReshapeLayer(convpool, ([0], numTimeWin, get_output_shape(convnets[0])[1]))
    convpool = DimshuffleLayer(convpool, (0, 2, 1))
    # convpool = ReshapeLayer(convpool, (-1, numTimeWin))

    # input to 1D convlayer should be in (batch_size, num_input_channels, input_length)
    convpool = Conv1DLayer(convpool, 64, 3)

    # A fully-connected layer of 512 units with 50% dropout on its inputs:
    convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5),
            num_units=512, nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the output layer with 50% dropout on its inputs:
    convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5),
            num_units=nb_classes, nonlinearity=lasagne.nonlinearities.softmax)
    return convpool


def build_convpool_lstm(input_vars, numTimeWin, nb_classes, GRAD_CLIP=100):
    """
    Builds the complete network with LSTM layer to integrate time from sequences of EEG images.
    :param input_vars: list of EEG images (one image per time window)
    :param numTimeWin: number of time windows
    :param nb_classes: number of classes
    :param GRAD_CLIP:  the gradient messages are clipped to the given value during
                        the backward pass.
    :return: a pointer to the output of last layer
    """
    convnets = []
    W_init = None
    # Build 7 parallel CNNs with shared weights
    for i in range(numTimeWin):
        if i == 0:
            convnet, W_init = build_cnn(input_vars[i])
        else:
            convnet, _ = build_cnn(input_vars[i], W_init)
        convnets.append(FlattenLayer(convnet))
    # at this point convnets shape is [numTimeWin][n_samples, features]
    # we want the shape to be [n_samples, features, numTimeWin]
    convpool = ConcatLayer(convnets)
    # convpool = ReshapeLayer(convpool, ([0], -1, numTimeWin))

    convpool = ReshapeLayer(convpool, ([0], numTimeWin, get_output_shape(convnets[0])[1]))
    # Input to LSTM should have the shape as (batch size, SEQ_LENGTH, num_features)
    convpool = LSTMLayer(convpool, num_units=128, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh)
    # After LSTM layer you either need to reshape or slice it (depending on whether you
    # want to keep all predictions or just the last prediction.
    # http://lasagne.readthedocs.org/en/latest/modules/layers/recurrent.html
    # https://github.com/Lasagne/Recipes/blob/master/examples/lstm_text_generation.py
    convpool = SliceLayer(convpool, -1, 1)      # Selecting the last prediction
    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5),
            num_units=256, nonlinearity=lasagne.nonlinearities.rectify)
    # We only need the final prediction, we isolate that quantity and feed it
    # to the next layer.

    # And, finally, the output layer with 50% dropout on its inputs:
    convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5),
            num_units=nb_classes, nonlinearity=lasagne.nonlinearities.softmax)
    return convpool

def build_convpool_mix(input_vars, numTimeWin, nb_classes, GRAD_CLIP=100):
    """
    Builds the complete network with LSTM and 1D-conv layers combined
    to integrate time from sequences of EEG images.
    :param input_vars: list of EEG images (one image per time window)
    :param numTimeWin: number of time windows
    :param nb_classes: number of classes
    :param GRAD_CLIP:  the gradient messages are clipped to the given value during
                        the backward pass.
    :return: a pointer to the output of last layer
    """
    convnets = []
    W_init = None
    # Build 7 parallel CNNs with shared weights
    for i in range(numTimeWin):
        if i == 0:
            convnet, W_init = build_cnn(input_vars[i])
        else:
            convnet, _ = build_cnn(input_vars[i], W_init)
        convnets.append(FlattenLayer(convnet))
    # at this point convnets shape is [numTimeWin][n_samples, features]
    # we want the shape to be [n_samples, features, numTimeWin]
    convpool = ConcatLayer(convnets)
    # convpool = ReshapeLayer(convpool, ([0], -1, numTimeWin))

    convpool = ReshapeLayer(convpool, ([0], numTimeWin, get_output_shape(convnets[0])[1]))
    reformConvpool = DimshuffleLayer(convpool, (0, 2, 1))

    # input to 1D convlayer should be in (batch_size, num_input_channels, input_length)
    conv_out = Conv1DLayer(reformConvpool, 64, 3)
    conv_out = FlattenLayer(conv_out)
    # Input to LSTM should have the shape as (batch size, SEQ_LENGTH, num_features)
    lstm = LSTMLayer(convpool, num_units=128, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh)
    # After LSTM layer you either need to reshape or slice it (depending on whether you
    # want to keep all predictions or just the last prediction.
    # http://lasagne.readthedocs.org/en/latest/modules/layers/recurrent.html
    # https://github.com/Lasagne/Recipes/blob/master/examples/lstm_text_generation.py
    # lstm_out = SliceLayer(convpool, -1, 1)        # bypassing LSTM
    lstm_out = SliceLayer(lstm, -1, 1)

    # Merge 1D-Conv and LSTM outputs
    dense_input = ConcatLayer([conv_out, lstm_out])
    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    convpool = DenseLayer(lasagne.layers.dropout(dense_input, p=.5),
            num_units=512, nonlinearity=lasagne.nonlinearities.rectify)
    # We only need the final prediction, we isolate that quantity and feed it
    # to the next layer.

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    convpool = DenseLayer(convpool,
            num_units=nb_classes, nonlinearity=lasagne.nonlinearities.softmax)
    return convpool


if __name__ == '__main__':
    input_var = T.TensorType('floatX', ((False,) * 5))()        # Notice the () at the end
    target_var = T.ivector('targets')
    images = gen_images('Neuroscan_sample_locs.mat',
                        'WM_features_MSP_sample.mat',
                        16, augment=True, pca=True, n_components=3)
    network = build_convpool_max(input_var, 5, 3)
    network = build_convpool_conv1d(input_var, 5, 3)
    network = build_convpool_lstm(input_var, 5, 3, 90)
    network = build_convpool_mix(input_var, 5, 3, 90)

