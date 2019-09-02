#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : generate_bert_dataset.py
# Description : Generate BERT dataset.
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
import random
import string
import codecs
import re
import argparse

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


# Argument builder
args = argparse.ArgumentParser()

# Dataset arguments
args.add_argument(command="--dataset", name="dataset", type=str,
                  help="JSON file with the file description for each authors", required=True, extended=False)
args.add_argument(command="--k", name="k", type=int, help="K-Fold Cross Validation", extended=False, default=10)
args.add_argument(command="--output", name="output", type=str, help="Experiment's output directory", required=True,
                  extended=False)
args.parse_args()

# Parse truth
truth = parse_truth_file(os.path.join(args.dataset, "truth.txt"))

# Create directories for each fold
for k in range(args.k):
    # Fold directory
    fold_directory = os.path.join(args.output, "{}".format(k))
    os.mkdir(fold_directory)

    # For train, test and val
    for d_name in ["train", "test", "val"]:
        # Dataset directory
        dataset_directory = os.path.join(fold_directory, d_name)

        # Class directory
        male_directory = os.path.join(dataset_directory, "male")
        female_directory = os.path.join(dataset_directory, "female")

        # Create directories
        os.mkdir(dataset_directory)
        os.mkdir(male_directory)
        os.mkdir(female_directory)
    # end for

    # Dataset IDs
    dataset_ids = range(3600)
    test_ids = range(k * 360, k * 360 + 180)
    val_ids = range(k * 360 + 180, k * 360 + 360)
    train_ids = dataset_directory - (test_ids + val_ids)

    # For each file in the dataset directory
    for i, file in enumerate(os.listdir(args.dataset)):
        if file[-4:] == ".xml":
            # Target directory
            if i in train_ids:
                target_directory = os.path.join(fold_directory, "train")
            elif test_ids:
                target_directory = os.path.join(fold_directory, "test")
            else:
                target_directory = os.path.join(fold_directory, "val")
            # end if

            # Parse the file
            tree = etree.parse(os.path.join(args.dataset, file))

            # Author's name
            author_name = file[:-4]

            # Author's gender
            author_gender = truth[author_name]['gender']

            # Random name
            random_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))

            # Destination
            destination_path = os.path.join(target_directory, author_gender, random_name + ".txt")

            # Log
            print("Writing file {}".format(destination_path))

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
        # end if
    # end for
# end for
