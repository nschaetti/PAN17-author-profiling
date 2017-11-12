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
from tools.ResultManager import ResultManager
from tools.functions import create_tokenizer
from corpus.CrossValidation import CrossValidation
from corpus.Corpus import Corpus

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

    # Naive Bayes classifier arguments
    args.add_argument(command="--smoothing-type", name="smoothing_type", type=float, help="Smoothing type",
                      required=True, extended=True)
    args.add_argument(command="--smoothing-param", name="smoothing_param", type=float, help="Smoothing parameter",
                      required=True, extended=True)

    # Experiment output parameters
    args.add_argument(command="--name", name="name", type=str, help="Experiment's name", extended=False, required=True)
    args.add_argument(command="--description", name="description", type=str, help="Experiment's description",
                      extended=False, required=True)
    args.add_argument(command="--output", name="output", type=str, help="Experiment's output directory", required=True,
                      extended=False)
    args.add_argument(command="--tweets", name="tweets", action='store_true',
                      help="Test tweet classification rate?", default=False, extended=False)
    args.add_argument(command="--verbose", name="verbose", type=int, help="Verbose level", default=2, extended=False)

    # Parse arguments
    args.parse()

    # Corpus
    reteursC50 = Corpus(args.dataset)

    # Parameter space
    param_space = nsNLP.tools.ParameterSpace(args.get_space())

    # Experiment
    xp = nsNLP.tools.ResultManager\
    (
        args.output,
        args.name,
        args.description,
        args.get_space(),
        args.n_samples,
        args.k,
        verbose=args.verbose
    )

    # Author list
    authors = reteursC50.get_authors()[:args.n_authors]
    author_list = reteursC50.get_authors_list()[:args.n_authors]

    # Print authors
    xp.write(u"Authors : {}".format(author_list), log_level=0)

    # Iterate
    for space in param_space:
        # Params
        smoothing_type = int(space['smoothing_type'])
        smoothing_param = space['smoothing_param']

        # Choose the right tokenizer
        tokenizer = create_tokenizer(args.tokenizer, "")

        # Set experience state
        xp.set_state(space)

        # Average sample
        average_sample = np.array([])

        # For each sample
        for n in range(args.n_samples):
            # Set sample
            xp.set_sample_state(n)

            # Naive bayes classifier
            classifier = nsNLP.statistical_models.NaiveBayesClassifier(classes=author_list, smoothing=smoothing_type,
                                                                       smoothing_param=smoothing_param)

            # 10 fold cross validation
            cross_validation = CrossValidation(authors)

            # Average
            average_k_fold = np.array([])

            # For each fold
            for k, (train_set, test_set) in enumerate(cross_validation):
                # Set k
                xp.set_fold_state(k)

                # Add to examples
                for index, text in enumerate(train_set):
                    # Add
                    classifier.train(tokenizer(text.x()), text.y())
                # end for

                # Train
                classifier.finalize(verbose=False)

                # Counters
                successes = 0.0

                # Test the classifier
                for text in test_set:
                    # Predict
                    prediction, probs = classifier.predict(tokenizer(text.x()))

                    # Compare
                    if prediction == text.y():
                        successes += 1.0
                    # end if
                # end for

                # Print success rate
                xp.add_result(successes / float(len(test_set)))
                average_k_fold = np.append(average_k_fold, [successes / float(len(test_set))])

                # Reset classifier
                classifier.reset()
            # end for

            # Add
            average_sample = np.append(average_sample, [np.average(average_k_fold)])

            # Delete classifier
            del classifier
        # end for
    # end for

    # Save experiment results
    xp.save()
# end if
