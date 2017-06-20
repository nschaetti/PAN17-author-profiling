# -*- coding: utf-8 -*-
#
# File : core/downloader/PySpeechesConfig.py
# Description : .
# Date : 20th of February 2017
#
# This file is part of pySpeeches.  pySpeeches is free software: you can
# redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, version 2.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 51
# Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# Copyright Nils Schaetti, University of Neuchâtel <nils.schaetti@unine.ch>


# An abstract classifier
class Classifier(object):
    """
    An abstract classifier.
    """

    # Constructor
    def __init__(self, classes):
        """
        Constructor
        :param classes: Classes
        """
        self._classes = classes
        self._n_classes = len(classes)
    # end __init__

    # Add an example
    def add(self, c, x):
        """
        Add an example
        :param c: The instance's class
        :param x: The instance
        """
        pass
    # end add_example

    # Train the model
    def train(self):
        """
        Train the model
        """
        pass
    # end train

    # Predict a new instance
    def predict(self, x):
        """
        Predict a new instance
        :param x: The instance to classify
        :return: The predicted class
        """
        pass
    # end predict

# end Classifier
