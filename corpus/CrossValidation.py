# -*- coding: utf-8 -*-
#
# File : corpus/IQLACorpus.py
# Description : .
# Date : 16/08/2017
#
# Copyright Nils Schaetti, University of Neuchâtel <nils.schaetti@unine.ch>

# Imports
import math


# Iterate through a train and test dataset
# for a k-fold cross validation
class CrossValidation(object):
    """
    Iterate through a train and test dataset
    for a k-fold cross validation
    """

    # Constructor
    def __init__(self, authors, k=10):
        """
        Constructor
        """
        # Properties
        self._authors = authors
        self._k = k
        self._pos = 0
        self._n_authors = len(authors)
        self._fold_size = int(math.floor(float(self._n_authors) / float(k)))
    # end __init__

    #################################################
    # Override
    #################################################

    # Iterator
    def __iter__(self):
        """
        Iterator
        :return:
        """
        return self
    # end __iter__

    # Next
    def next(self):
        """
        Next
        :return:
        """
        if self._pos >= self._k:
            raise StopIteration
        # end if

        # Test indexes
        test_set = self._authors[self._pos*self._fold_size:self._pos*self._fold_size+self._fold_size]

        # Remove test indexes
        train_set = self._authors
        for a in test_set:
            train_set.remove(a)
        # end for

        # Next fold
        self._pos += 1

        # Result
        return train_set, test_set
    # end next

# end CrossValidation
