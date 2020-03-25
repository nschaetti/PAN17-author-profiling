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
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn import svm

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

    # Naive Bayes classifier arguments
    args.add_argument(command="--n-grams-min", name="n_grams_min", type=int, help="N-grams", required=True,
                      extended=True)
    args.add_argument(command="--n-grams-max", name="n_grams_max", type=int, help="N-grams", required=True,
                      extended=True)
    args.add_argument(command="--kernel", name="kernel", type=str, help="linear,poly,rbf", required=True,
                      extended=True)
    args.add_argument(command="--kernel-degree", name="kernel_degree", type=int, help="Kernel degree", default=3,
                      required=False, extended=True)
    args.add_argument(command="--penalty", name="penalty", type=float, help="L2 penalty", default=1.0,
                      required=False, extended=True)
    args.add_argument(command="--tfidf", name="tfidf", type=str, help="tfidf or none", default=False, required=False,
                      extended=True)

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

    # Bag of word features
    bow = nsNLP.features.BagOfWords()

    # Iterate
    for space in param_space:
        # Params
        ngrams_min = int(space['n_grams_min'])
        ngrams_max = int(space['n_grams_max'])
        kernel = space['kernel'][0][0]
        penalty = float(space['penalty'])
        kernel_degree = int(space['kernel_degree'])
        tfidf = space['tfidf'][0][0]

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

        # For each fold
        for k, (train_set, test_set) in enumerate(cross_validation):
            # Set k
            xp.set_fold_state(k)

            # Classifier
            classifier = svm.SVC(kernel=kernel, C=penalty, degree=kernel_degree)

            # Pipeline
            text_clf = Pipeline([
                ('vect', CountVectorizer(ngram_range=(ngrams_min, ngrams_max))),
                ('tfidf', TfidfTransformer(use_idf=True if tfidf == 'tfidf' else False)),
                ('clf', classifier)
            ])

            # Total text for each gender
            profile_X = list()
            profile_Y = list()

            # Add to author
            for index, author in enumerate(train_set):
                profile_X.append(author.get_texts()[0].x())
                profile_Y.append(author.truth('gender'))
            # end for

            # Fit
            text_clf.fit(profile_X, profile_Y)

            # Counters
            successes = 0.0

            # Test the classifier
            for author in test_set:
                # Prediction
                prediction = text_clf.predict([author.get_texts()[0].x()])

                # Compare
                if prediction == author.truth('gender'):
                    successes += 1.0
                # end if
            # end for

            # Print success rate
            xp.add_result(successes / float(len(test_set)))
        # end for
    # end for

    # Save experiment results
    xp.save()
# end if
