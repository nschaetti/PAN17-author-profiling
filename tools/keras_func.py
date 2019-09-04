#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
import torch
import keras
from keras.utils import to_categorical
import numpy as np


# Transform pyTorch tensor to a list of numpy array
def dataset_to_numpy(loader):
    """
    Transform pyTorch tensor to a list of numpy array
    :param loader:
    :return:
    """
    # Lists
    inputs_list = list()
    labels_list = list()
    time_labels_list = list()

    # Get training data for this fold
    for i, data in enumerate(loader):
        # Inputs and labels
        inputs, labels, _ = data

        # Input length
        Tx = inputs.size(1)

        # Time labels
        time_labels = torch.LongTensor(1, Tx).fill_(labels[0])

        # Inputs to list
        input_to_list = inputs[0].numpy()

        # Time labels to list
        time_labels_to_list = time_labels[0].numpy()

        # Append
        inputs_list.append(input_to_list)
        labels_list.append(labels[0].numpy())
        time_labels_list.append(time_labels_to_list)
    # end for

    return inputs_list, labels_list, time_labels_list
# end dataset_to_numpy


# Transform pyTorch tensors to lists
def dataset_to_list(loader, voc_size):
    """
    Transform pyTorch tensors to lists.
    :param loader:
    :return:
    """
    # Lists
    inputs_list = list()
    labels_list = list()
    time_labels_list = list()

    # Get training data for this fold
    for i, data in enumerate(loader):
        # Inputs and labels
        inputs, labels, _ = data

        # Input length
        Tx = inputs.size(1)

        # Time labels
        time_labels = torch.LongTensor(1, Tx).fill_(labels[0])

        # Inputs to list
        input_to_list = inputs[0].tolist()
        input_to_list.append(voc_size+1)

        # Time labels to list
        time_labels_to_list = time_labels[0].tolist()
        time_labels_to_list.append(time_labels_to_list[-1])

        # Append
        inputs_list.append(input_to_list)
        labels_list.append(labels[0].tolist())
        time_labels_list.append(time_labels_to_list)
    # end for

    return inputs_list, labels_list, time_labels_list
# end dataset_to_list


# Data generator
def data_generator(batch_size, data_inputs, data_labels, num_classes=15):
    """
    Train generator
    :return:
    """
    for i in range(len(data_inputs)):
        x_train = np.zeros((batch_size, ))
        x_train = np.array(data_inputs[i])
        y_train = to_categorical(np.array(data_labels[i]), num_classes=num_classes)
        yield x_train, y_train
    # end for
# end data_generator
