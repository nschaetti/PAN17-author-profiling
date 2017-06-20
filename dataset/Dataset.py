#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import
import os
from .Author import Author


# Represents a data set.
class Dataset(object):
    """
    Represents a data set.
    """

    # Constructor
    def __init__(self):
        """
        Constructor
        """
        self._current_author = 0
        self._authors = list()
        self._working_directory = ""
    # end __init__

    ###############################################
    # Public functions
    ###############################################

    # Number of authors
    def n_authors(self):
        """
        Number of authors
        :return: The number of authors
        """
        return len(self._authors)
    # end n_authors

    # Load a directory
    def load(self, directory_path):
        """
        Load a directory.
        :param directory_path: The path of the directory
        """
        # Working directory
        self._working_directory = directory_path

        # List files
        for f in os.listdir(directory_path):
            if os.path.isfile(os.path.join(directory_path, f)) and f[-4:] == ".xml":
                self._authors.append(f)
            # end if
        # end for
    # end load

    # Open a directory
    @staticmethod
    def open(directory):
        """
        Open a directory
        :param directory: Directory to open.
        :return: The Dataset object.
        """
        data_set = Dataset()
        data_set.load(directory)
        return data_set
    # end open

    # Iterator
    def __iter__(self):
        """
        Iterator
        :return:
        """
        return self
    # end __iter__

    # Next element
    def next(self):
        """
        Next element
        :return: The next element.
        """
        if self._current_author >= self.n_authors():
            self._current_author = 0
            raise StopIteration
        else:
            self._current_author += 1
            author = Author.open(self._working_directory, self._authors[self._current_author-1])
            return author
        # end if
    # end next

    ###############################################
    # Private functions
    ###############################################

# end Dataset