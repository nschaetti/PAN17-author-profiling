#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
import torch
from keras.utils import to_categorical
import numpy as np


# PAN17BatchGenerator
class PAN17BatchGenerator(object):
    """
    PAN17 Batch Generator
    """

    # Constructor
    def __init__(self, data_inputs, data_labels, batch_size, num_classes, max_index, many_to_many=True, pretrained=False):
        """
        Constructor
        :param data:
        :param num_steps:
        :param batch_size:
        :param voc_size:
        """
        self.data_inputs = data_inputs
        self.data_labels = data_labels
        self.data_size = len(self.data_inputs)
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.many_to_many = many_to_many
        self.current_index = 0
        self.max_index = max_index
        self.pretrained = pretrained
        self.sample_lengths = None
    # end __init__

    # Length
    def __len__(self):
        """
        Length
        :return:
        """
        return int(np.ceil(len(self.data_inputs) / self.batch_size))
    # end __len__

    # Generate one batch of data
    def __getitem__(self, item):
        """
        Generate one batch of data
        :param item:
        :return:
        """
        # Generate indexes of the batch
        indexes = range(item * self.batch_size, (item+1) * self.batch_size)

        # Filter
        indexes = [x for x in indexes if x < self.data_size]

        # Generate data
        X, Y = self._data_generation(indexes)

        return X, Y
    # end __getitem__

    # Generate indexes
    def generate_indexes(self, return_lengths=False):
        """
        Generate indexes
        :return:
        """
        while True:
            # Generate indexes of the batch
            indexes = range(self.current_index * self.batch_size, (self.current_index + 1) * self.batch_size)

            # Filter
            indexes = [x for x in indexes if x < self.data_size]

            # Next index
            self.current_index += 1
            if self.current_index * self.batch_size >= self.data_size:
                self.current_index = 0
            # end if

            # Batch size
            batch_size = len(indexes)

            # Sample lengths
            self.sample_lengths = np.zeros(batch_size, dtype=int)

            # Max length
            max_length = 0
            for i, ix in enumerate(indexes):
                self.sample_lengths[i] = len(self.data_inputs[ix])
                if len(self.data_inputs[ix]) > max_length:
                    max_length = len(self.data_inputs[ix])
                # end if
            # end for
            xl = self.sample_lengths

            # Initialization X
            x = np.zeros((batch_size, max_length))

            # Initialization Y
            if self.many_to_many:
                y = np.zeros((batch_size, max_length, self.num_classes))
            else:
                y = np.zeros((batch_size, self.num_classes))
            # end if

            # Generate each sample
            for i, ix in enumerate(indexes):
                # To numpy and category
                x_train = np.array(self.data_inputs[ix])

                # Zero for index above max_index
                if self.max_index != -1:
                    x_train[x_train >= self.max_index] = 0
                # end if

                # Labels
                y_train = to_categorical(np.array(self.data_labels[ix]), num_classes=self.num_classes)

                # Sample length
                sample_length = x_train.shape[0]

                # Set X
                x[i, :sample_length] = x_train

                # Set Y
                if self.many_to_many:
                    y[i, :sample_length] = y_train
                else:
                    y[i] = y_train
                # end if
            # end for

            if return_lengths:
                yield x, y, xl
            else:
                yield x, y
            # end if
        # end while
    # end _generate_indexes

    # Generate embeddings
    def generate_embeddings(self, return_lengths=False):
        """
        Generate embeddings
        :return:
        """
        while True:
            # Generate indexes of the batch
            indexes = range(self.current_index * self.batch_size, (self.current_index + 1) * self.batch_size)

            # Filter
            indexes = [x for x in indexes if x < self.data_size]

            # Next index
            self.current_index += 1
            if self.current_index * self.batch_size >= self.data_size:
                self.current_index = 0
            # end if

            # Batch size
            batch_size = len(indexes)

            # Embedding size
            embedding_size = self.data_inputs[0].shape[1]

            # Sample lengths
            self.sample_lengths = np.zeros(batch_size, dtype=int)

            # Max length
            max_length = 0
            for i, ix in enumerate(indexes):
                self.sample_lengths[i] = int(self.data_inputs[ix].shape[0])
                if self.data_inputs[ix].shape[0] > max_length:
                    max_length = self.data_inputs[ix].shape[0]
                # end if
            # end for
            xl = self.sample_lengths

            # Initialization X
            x = np.zeros((batch_size, max_length, embedding_size))

            # Initialization Y
            if self.many_to_many:
                y = np.zeros((batch_size, max_length, self.num_classes))
            else:
                y = np.zeros((batch_size, self.num_classes))
            # end if

            # Generate each sample
            for i, ix in enumerate(indexes):
                # To numpy and category
                x_train = self.data_inputs[ix]

                # Zero for index above max_index
                # x_train[x_train >= self.max_index] = 0

                # Labels
                y_train = to_categorical(self.data_labels[ix], num_classes=self.num_classes)

                # Sample length
                sample_length = x_train.shape[0]

                # Set X
                x[i, :sample_length] = x_train

                # Set Y
                if self.many_to_many:
                    y[i, :sample_length] = y_train
                else:
                    y[i] = y_train
                # end if
            # end for

            if return_lengths:
                yield x, y, xl
            else:
                yield x, y
            # end if
        # end while
    # end _generate_embeddings

    # Generate
    def _data_generation(self, indexes):
        """
        Generate batch
        :return:
        """
        # Batch size
        batch_size = len(indexes)

        # Max length
        max_length = 0
        for i, ix in indexes:
            if len(self.data_inputs[ix]) > max_length:
                max_length = len(self.data_inputs[ix])
            # end if
        # end for

        # Initialization X
        x = np.zeros((batch_size, max_length))

        # Initialization Y
        if self.many_to_many:
            y = np.zeros((batch_size, max_length, self.num_classes))
        else:
            y = np.zeros((batch_size, self.num_classes))
        # end if

        # Generate each sample
        for i, ix in indexes:
            # To numpy and category
            x_train = np.array(self.data_inputs[i])
            y_train = to_categorical(np.array(self.data_labels[i]), num_classes=self.num_classes)

            # Sample length
            sample_length = x_train.shape[0]

            # Set X
            x[i, :sample_length] = x_train

            # Set Y
            if self.many_to_many:
                y[i, :sample_length] = y_train
            else:
                y[i] = y_train
        # end for

        return x, y
    # end _data_generation

# end PAN17BatchGenerator
