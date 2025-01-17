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
from nltk.tokenize import TweetTokenizer

####################################################
# Main function
####################################################

# Main function
if __name__ == "__main__":

    # Argument builder
    args = nsNLP.tools.ArgumentBuilder(desc="Sklearn classifier baseline benchmark")

    # Dataset arguments
    args.add_argument(command="--dataset", name="dataset", type=str,
                      help="JSON file with the file description for each authors", required=True, extended=False)
    args.add_argument(command="--k", name="k", type=int, help="K-Fold Cross Validation", extended=False, default=10)

    # Parameters
    args.add_argument(command="--feature", name="feature", type=str, help="word,char,char_wb", default='word',
                      required=False,
                      extended=True)

    # Experiment output parameters
    args.add_argument(command="--name", name="name", type=str, help="Experiment's name", extended=False, required=True)
    args.add_argument(command="--description", name="description", type=str, help="Experiment's description",
                      extended=False, required=True)
    args.add_argument(command="--output", name="output", type=str, help="Experiment's output directory", required=True,
                      extended=False)
    args.add_argument(command="--verbose", name="verbose", type=int, help="Verbose level", default=2, extended=False)

    # Parse arguments
    args.parse()

    # Corpus
    pan17corpus = nsNLP.data.Corpus(args.dataset)

    # Parameter space
    param_space = nsNLP.tools.ParameterSpace(args.get_space())

    # Experiment
    xp = nsNLP.tools.ResultManager\
    (
        args.output,
        args.name,
        args.description,
        args.get_space(),
        1,
        args.k,
        verbose=args.verbose
    )

    # Author list
    authors = pan17corpus.get_authors()

    # Tweet tokenizer
    tknzr = TweetTokenizer()

    # Iterate
    for space in param_space:
        # Set experience state
        xp.set_state(space)

        # Average sample
        average_sample = np.array([])

        # Set sample
        xp.set_sample_state(0)

        # 10 fold cross validation
        cross_validation = nsNLP.validation.CrossValidation(authors)

        # Average
        average_k_fold = np.array([])

        # Counters
        male_word_counter = 0
        female_word_counter = 0

        # For each fold
        for k, (train_set, test_set) in enumerate(cross_validation):
            # Set k
            xp.set_fold_state(k)

            # Add to author
            for index, author in enumerate(train_set):
                words = tknzr.tokenize(author.get_texts()[0].x())
                if author.truth('gender') == 'male':
                    male_word_counter += len(words)
                else:
                    female_word_counter += len(words)
                # end if
            # end for

            # Test the classifier
            for author in test_set:
                words = tknzr.tokenize(author.get_texts()[0].x())
                if author.truth('gender') == 'male':
                    male_word_counter += len(words)
                else:
                    female_word_counter += len(words)
                # end if
            # end for
            print("Male counter = {}".format(male_word_counter))
            print("Female counter = {}".format(female_word_counter))
        # end for
    # end for

    # Save experiment results
    xp.save()
# end if
