#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : core.classifiers.RCNLPTextClassifier.py
# Description : Echo State Network for text classification.
# Auteur : Nils Schaetti <nils.schaetti@unine.ch>
# Date : 01.02.2017 17:59:05
# Lieu : Nyon, Suisse
#
# This file is part of the Reservoir Computing NLP Project.
# The Reservoir Computing Memory Project is a set of free software:
# you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Foobar is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
#

import nsNLP
import numpy as np
from tools.functions import create_tokenizer
from corpus.CrossValidation import CrossValidation

####################################################
# Functions
####################################################


####################################################
# Main function
####################################################

# Main function
if __name__ == "__main__":

    # Argument builder
    args = nsNLP.tools.ArgumentBuilder(desc=u"Argument test")

    # Dataset arguments
    args.add_argument(command="--dataset", name="dataset", type=str,
                      help="JSON file with the file description for each authors", required=True, extended=False)

    # Parse arguments
    args.parse()

    # Corpus
    pan17corpus = nsNLP.data.Corpus(args.dataset)

    # Author list
    authors = pan17corpus.get_authors()

    # List of characters
    char_list = list()

    # For each authors
    for author in authors:
        # For each text
        for author_text in author.get_texts():
            text = author_text.x()
            # For each character
            for i in range(len(text)):
                c = text[i]
                if c not in char_list:
                    char_list.append(c)
                # end if
            # end for
        # end for
    # end for

    print(u"alphabet = {}".format(char_list))
# end if
