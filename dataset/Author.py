#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import
import os
from lxml import etree
from .Document import Document


# Author
class Author(object):
    """
    Author
    """

    # Constructor
    def __init__(self, dataset_path, author_path):
        """
        Constructor
        """
        self._dataset_path = dataset_path
        self._author_path = author_path
        self._documents = []
        self._n_documents = 0
        self._current_document = 0
        self._id = os.path.splitext(author_path)[0]
        self._gender = ""
        self._language_variety = ""

        # Load documents
        self._load_documents(author_path)

        # Load truth
        self._load_truth()
    # end __init__

    ###############################################
    # Public functions
    ###############################################

    # Number of documents
    def n_documents(self):
        """
        Number of documents
        :return: Number of documents
        """
        return self._n_documents
    # end n_documents

    # Get author's ID
    def get_id(self):
        """
        Get author's ID
        :return: Author's ID
        """
        return self._id
    # end get_id

    # Get author's gender
    def get_gender(self):
        """
        Get author's gender
        :return: Author's gender
        """
        return self._gender
    # end get_gender

    # Get author's variety
    def get_variety(self):
        """
        Get author's language variety
        :return: Author's language variety
        """
        return self._language_variety
    # end get_variety

    # Open an author file
    @staticmethod
    def open(working_directory, file_path):
        """
        Open an author file
        :param working_directory:
        :param file_path:
        :return:
        """
        author = Author(working_directory, file_path)
        return author
    # end load

    # Iterator
    def __iter__(self):
        """
        Iterator
        :return: self
        """
        return self
    # end __iter__

    # Next element
    def next(self):
        """
        Next element
        :return: Next element
        """
        if self._current_document >= self._n_documents:
            self._current_document = 0
            raise StopIteration
        else:
            self._current_document += 1
            return Document(self._documents[self._current_document-1].text)
        # end if
    # end next

    ###############################################
    # Private functions
    ###############################################

    # Load documents
    def _load_documents(self, author_path):
        """
        Load documents
        :param author_path: Author's file path.
        """
        # Parse
        tree = etree.parse(os.path.join(self._dataset_path, author_path))

        # For each documents
        for document in tree.xpath("/author/documents/document"):
            self._documents.append(document)
            self._n_documents += 1
        # end for
    # end _load_documents

    # Load the truth
    def _load_truth(self):
        """
        Load the truth
        :param dataset_path:
        :return:
        """
        with open(os.path.join(self._dataset_path, "truth.txt"), 'r') as f:
            # Content
            truths = f.read()

            # For reach lines
            for line in truths.split('\n'):
                infos = line.split(":::")
                if infos[0] == self._id:
                    self._gender = infos[1]
                    self._language_variety = infos[2]
                # end if
            # end for
        # end with
    # end _load_truth

# end Author
