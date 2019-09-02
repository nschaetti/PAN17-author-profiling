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
import numpy as np

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
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="JSON file with the file description for each authors")
parser.add_argument("--k", type=int, help="K-Fold Cross Validation", default=10)
parser.add_argument("--output", type=str, help="Experiment's output directory")
parser.add_argument("--length", type=int, default=3072)
args = parser.parse_args()

# Parse truth
truth = parse_truth_file(os.path.join(args.dataset, "en.txt"))

# Create directories for each fold
for k in range(args.k):
    # Fold directory
    fold_directory = os.path.join(args.output, "{}".format(k))

    # Fold directory
    if not os.path.exists(fold_directory):
        os.mkdir(fold_directory)
    # end if

    # For train, test and val
    for d_name in ["train", "test", "val"]:
        # Dataset directory
        dataset_directory = os.path.join(fold_directory, d_name)

        # Class directory
        male_directory = os.path.join(dataset_directory, "male")
        female_directory = os.path.join(dataset_directory, "female")

        # Create dataset directory
        if not os.path.exists(dataset_directory):
            os.mkdir(dataset_directory)
        # end if

        # Create male directory
        if not os.path.exists(male_directory):
            os.mkdir(male_directory)
        # end if

        # Create female directory
        if not os.path.exists(female_directory):
            os.mkdir(female_directory)
        # end if
    # end for

    # Dataset IDs
    dataset_ids = np.arange(3600)
    test_ids = np.arange(k * 360, k * 360 + 180)
    val_ids = np.arange(k * 360 + 180, k * 360 + 360)
    train_ids = np.delete(dataset_ids, np.append(test_ids, val_ids))

    # File index
    file_index = 0

    # For each file in the dataset directory
    for i, file in enumerate(os.listdir(args.dataset)):
        if file[-4:] == ".xml":
            # Parse the file
            tree = etree.parse(os.path.join(args.dataset, file))

            # Author's name
            author_name = file[:-4]

            # English author
            if author_name in truth.keys():
                # Target directory
                if file_index in train_ids:
                    target_directory = os.path.join(fold_directory, "train")
                elif file_index in test_ids:
                    target_directory = os.path.join(fold_directory, "test")
                else:
                    target_directory = os.path.join(fold_directory, "val")
                # end if

                # Author's gender
                author_gender = truth[author_name]['gender']

                # Destination
                destination_path = os.path.join(target_directory, author_gender)

                # Get complete text
                author_text = ""
                for document in tree.xpath("/author/documents/document"):
                    # Get text
                    document_text = document.text

                    # Fin all URLs
                    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                                      document_text)

                    # Remove all URLs
                    for url in urls:
                        document_text = document_text.replace(url, "")
                    # end for

                    # Write the document
                    author_text += document_text.replace('\n', ' ') + "\n"
                # end for

                # Part index
                part_index = 0

                # For each part
                for pos in range(0, len(author_text), args.length):
                    # Log
                    print("Writing file {}".format(os.path.join(destination_path, "{}.txt".format(file_index))))

                    # Write the file
                    codecs.open(os.path.join(destination_path, "{}-{}.txt".format(file_index, part_index)), 'w', encoding='utf-8').write(author_text[pos:pos+args.length])

                    # Next part index
                    part_index += 1
                # end for

                # Next file index
                file_index += 1
            # end if
        # end if
    # end for
# end for
