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

import os
from lxml import etree
import nsNLP
import random
import string
import json
import codecs
import re

####################################################
# Functions
####################################################


# Parse truth file
def parse_truth_file(truth_file):
    """
    Parse truth file
    :param truth_file:
    :return:
    """
    with codecs.open(truth_file, 'r') as f:
        # All lines
        content = f.read()

        # Dictionary if author info
        author_truth = dict()

        # Parse by line
        for line in content.split(u'\n'):
            if len(line) > 0:
                # Name, gender, variety
                name, gender, variety = line.split(u':::')

                # Create dict
                if name not in author_truth.keys():
                    author_truth[name] = dict()
                # end if

                # Set
                author_truth[name]['gender'] = gender
                author_truth[name]['variety'] = variety
            # end if
        # end for

        return author_truth
    # end with
# end parse_truth_file

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
    args.add_argument(command="--k", name="k", type=int, help="K-Fold Cross Validation", extended=False, default=10)
    args.add_argument(command="--output", name="output", type=str, help="Experiment's output directory", required=True,
                      extended=False)

    # Parse arguments
    args.parse()

    # Author informations
    author_infos = dict()

    # For each file in the dataset directory
    for file in os.listdir(args.dataset):
        if file[-4:] == ".xml":
            # Parse the file
            tree = etree.parse(os.path.join(args.dataset, file))

            # Author's name
            author_name = file[:-4]

            # Add author to dict
            if author_name not in author_infos.keys():
                author_infos[author_name] = list()
            # end if

            # Random name
            random_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))

            # Destination
            destination_path = os.path.join(args.output, random_name + ".txt")

            # Log
            print(u"Writing file {}".format(destination_path))

            # Write the file
            with codecs.open(destination_path, 'w', encoding='utf-8') as f:
                # For each document
                for document in tree.xpath("/author/documents/document"):
                    # Get text
                    document_text = document.text

                    # Fin all URLs
                    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                                      document_text)

                    # Remove all URLs
                    for url in urls:
                        document_text = document_text.replace(url, u"")
                    # end for

                    # Write the document
                    f.write(document_text.replace(u'\n', u' ') + u"\n")
                # end for
            # end with

            # Add to author
            author_infos[author_name].append(random_name)
        # end if
    # end for

    # Write JSON
    with open(os.path.join(args.output, "authors.json"), 'w') as f:
        json.dump(author_infos, f, encoding='utf-8', indent=4)
    # end with

    # Write truth
    with open(os.path.join(args.output, "truth.json"), 'w') as f:
        json.dump(parse_truth_file(os.path.join(args.dataset, "truth.txt")), f, encoding='utf-8', indent=4)
    # end with

# end if
