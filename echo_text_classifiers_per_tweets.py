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
from alphabet import reservoir_alphabet

####################################################
# Functions
####################################################


# Converter in
def converter_in(converters_desc, converter):
    """
    Is the converter in the desc
    :param converters_desc:
    :param converter:
    :return:
    """
    for converter_desc in converters_desc:
        if converter in converter_desc:
            return True
        # end if
    # end for
    return False
# end converter_in


# Do we have to change W for this property
def change_w(params):
    """
    Do we have to change W for this parameter
    :param params:
    :return:
    """
    for param in params:
        if param in [u"reservoir_size", u"w_sparsity"]:
            return True
        # end if
    # end for
    return False
# end keep_w


# Get changed params
def get_changed_params(new_space, last_space):
    """
    Get changed param
    :param new_space:
    :param last_space:
    :return:
    """
    # Empty last space
    if len(last_space.keys()) == 0:
        return new_space.keys()
    # end if

    # Changed params
    changed_params = list()

    # For each param in new space
    for new_param in new_space.keys():
        if new_param not in last_space.keys():
            changed_params.append(new_param)
        else:
            if new_space[new_param] != last_space[new_param]:
                changed_params.append(new_param)
            # end if
        # end if
    # end for

    return changed_params
# end get_changed_params

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

    # ESN arguments
    args.add_argument(command="--reservoir-size", name="reservoir_size", type=float, help="Reservoir's size",
                      required=True, extended=True)
    args.add_argument(command="--spectral-radius", name="spectral_radius", type=float, help="Spectral radius",
                      default="1.0", extended=True)
    args.add_argument(command="--leak-rate", name="leak_rate", type=str, help="Reservoir's leak rate", extended=True,
                      default="1.0")
    args.add_argument(command="--input-scaling", name="input_scaling", type=str, help="Input scaling", extended=True,
                      default="0.5")
    args.add_argument(command="--input-sparsity", name="input_sparsity", type=str, help="Input sparsity", extended=True,
                      default="0.05")
    args.add_argument(command="--w-sparsity", name="w_sparsity", type=str, help="W sparsity", extended=True,
                      default="0.05")
    args.add_argument(command="--converters", name="converters", type=str,
                      help="The text converters to use (fw, pos, tag, wv, oh)", default='oh', extended=True)
    args.add_argument(command="--pca-path", name="pca_path", type=str, help="PCA model to load", default=None,
                      extended=False)
    args.add_argument(command="--keep-w", name="keep_w", action='store_true', help="Keep W matrix", default=False,
                      extended=False)
    args.add_argument(command="--aggregation", name="aggregation", type=str, help="Output aggregation method", extended=True,
                      default="average")
    args.add_argument(command="--state-gram", name="state_gram", type=str, help="State-gram value",
                      extended=True, default="1")

    # Tokenizer and word vector parameters
    args.add_argument(command="--lang", name="lang", type=str, help="Tokenizer language parameters", default='en',
                      extended=False)

    # Experiment output parameters
    args.add_argument(command="--name", name="name", type=str, help="Experiment's name", extended=False, required=True)
    args.add_argument(command="--description", name="description", type=str, help="Experiment's description",
                      extended=False, required=True)
    args.add_argument(command="--output", name="output", type=str, help="Experiment's output directory", required=True,
                      extended=False)
    args.add_argument(command="--tweets", name="tweets", action='store_true',
                      help="Test tweet classification rate?", default=False, extended=False)
    args.add_argument(command="--n-samples", name="n_samples", type=int, help="Number of different reservoir to test",
                      default=1, extended=False)
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
        args.n_samples,
        args.k,
        verbose=args.verbose
    )

    # Author list
    authors = pan17corpus.get_authors()

    # W index
    w_index = 0

    # Last space
    last_space = dict()

    # Iterate
    for space in param_space:
        # Params
        reservoir_size = int(space['reservoir_size'])
        w_sparsity = space['w_sparsity']
        leak_rate = space['leak_rate']
        input_scaling = space['input_scaling']
        input_sparsity = space['input_sparsity']
        spectral_radius = space['spectral_radius']
        converter_desc = space['converters']
        aggregation = space['aggregation'][0][0]
        state_gram = space['state_gram']

        # Choose the right tokenizer
        if converter_in(converter_desc, "wv") or \
                converter_in(converter_desc, "pos") or \
                converter_in(converter_desc, "tag"):
            tokenizer = create_tokenizer("spacy_wv")
        else:
            tokenizer = create_tokenizer("nltk")
        # end if

        # Set experience state
        xp.set_state(space)

        # Average sample
        average_sample = np.array([])

        # For each sample
        for n in range(args.n_samples):
            # Changed parameter
            changed_params = get_changed_params(space, last_space)

            # Generate a new W if necessary
            if change_w(changed_params) or not args.keep_w:
                xp.write(u"\t\tGenerating new W matrix", log_level=2)
                w = nsNLP.esn_models.ESNTextClassifier.w(rc_size=reservoir_size, rc_w_sparsity=w_sparsity)
                xp.save_object(u"w_{}".format(w_index), w, info=u"{}".format(space))
            # end if

            # Set sample
            xp.set_sample_state(n)

            # Create ESN text classifier
            classifier = nsNLP.esn_models.ESNTextClassifier.create\
            (
                classes=['female', 'male'],
                rc_size=reservoir_size,
                rc_spectral_radius=spectral_radius,
                rc_leak_rate=leak_rate,
                rc_input_scaling=input_scaling,
                rc_input_sparsity=input_sparsity,
                rc_w_sparsity=w_sparsity,
                converters_desc=converter_desc,
                use_sparse_matrix=True if converter_in(converter_desc, "oh") or converter_in(converter_desc, "ch") else False,
                w=w,
                aggregation=aggregation,
                state_gram=state_gram,
                pca_path=args.pca_path,
                alphabet=reservoir_alphabet
            )

            # 10 fold cross validation
            cross_validation = CrossValidation(authors)

            # Average
            average_k_fold = np.array([])

            # For each fold
            for k, (train_set, test_set) in enumerate(cross_validation):
                # Set k
                xp.set_fold_state(k)

                # Add to examples
                for index, author in enumerate(train_set):
                    # Get whole text
                    author_texts = author.get_texts()[0].x()

                    # Split by line
                    tweets = author_texts.split(u'\n')

                    # For each tweets
                    for tweet in tweets:
                        # Add
                        if converter_desc != 'ch':
                            classifier.train(tokenizer(tweet), author.truth('gender'))
                        else:
                            classifier.train(tweet, author.truth('gender'))
                        # end if
                    # end for
                # end for

                # Train
                classifier.finalize(verbose=False)

                # Counters
                successes = 0.0

                # Test the classifier
                for author in test_set:
                    # Get whole text
                    author_texts = author.get_texts()[0].x()

                    # Split by line
                    tweets = author_texts.split(u'\n')

                    # Tweets probs
                    tweets_probs = np.zeros((len(tweets), 2))

                    # For each tweets
                    for t_index, tweet in enumerate(tweets):
                        # Predict
                        if converter_desc != 'ch':
                            prediction, probs = classifier.predict(tokenizer(author.get_texts()[0].x()))
                        else:
                            prediction, probs = classifier.predict(author.get_texts()[0].x())
                        # end if

                        # Add
                        tweets_probs[t_index, :] = probs
                    # end for

                    # Average
                    predict_average = np.average(tweets_probs, axis=0)

                    # Get prediction
                    prediction = 'female' if predict_average[0] > predict_average[1] else 'male'

                    # Compare
                    if prediction == author.truth('gender'):
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

            # Last space
            last_space = space

            # Delete classifier
            del classifier
        # end for

        # W index
        w_index += 1
    # end for

    # Save experiment results
    xp.save()
# end if
