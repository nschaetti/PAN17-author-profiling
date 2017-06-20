#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import
import os
import spacy


# Document
class Document(object):

    # Constructor
    def __init__(self, text):
        """
        Constructor
        :param text:
        """

        self._text = unicode(text)
        # Load language model
        self._nlp = spacy.load('en')
    # end __init__

    ###############################################
    # Public functions
    ###############################################

    # Get document's text
    def get_text(self):
        """
        Get document's text.
        :return:
        """
        return self._text
    # end get_text

    # Get document's tags
    def get_tags(self):
        pass
    # end get_tags

    # Get words
    def get_words(self):
        """
        Get words
        :return:
        """
        return self._nlp(self._text)
    # end get_doc

    ###############################################
    # Private functions
    ###############################################

# end Document
