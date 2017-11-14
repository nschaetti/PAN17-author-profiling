#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : word_vector_gender_classifier.py
# Description : Create the data set for PAN17 with pySpeeches.
# Date : 09 March 2017

# Import packages
import nsNLP
import argparse
import numpy as np
import math
import cPickle as pickle
import torch
import codecs

# Accepted characters
alphabet = u" etaoinsrlhducmygpwfbk@v#!jx?z…q😂❤🏻😍😭🏼👍😊👏👌🙌😉🤔🙈🙄😘😩🏽🎉☺😁🔥💕😀😢🙏🎄🙃"


###########################
# Start here
###########################
if __name__ == "__main__":

    # Argument builder
    args = nsNLP.tools.ArgumentBuilder(desc=u"Naive bayes classifier baseline benchmark")

    # Dataset arguments
    args.add_argument(command="--dataset", name="dataset", type=str,
                      help="JSON file with the file description for each authors", required=True, extended=False)
    args.add_argument(command="--k", name="k", type=int, help="K-Fold Cross Validation", extended=False, default=10)

    # CNN arguments
    args.add_argument(command="--batch-size", name="batch_size", type=int, help="Input batch size", extended=False,
                      default=64)
    args.add_argument(command="--lr", name="lr", type=float, help="Learning rate", required=True, extended=True)
    args.add_argument(command="--momentum", name="momentum", type=float, help="SGD momentum", required=True,
                      extended=True)
    args.add_argument(command="--no-cuda", name="no_cuda", action='store_true',
                      help="Disables CUDA training", default=False, extended=False)
    args.add_argument(command="--epoch", name="epoch", type=int, help="Number of epoch", extended=False,
                      default=100)

    # Experiment output parameters
    args.add_argument(command="--name", name="name", type=str, help="Experiment's name", extended=False, required=True)
    args.add_argument(command="--description", name="description", type=str, help="Experiment's description",
                      extended=False, required=True)
    args.add_argument(command="--output", name="output", type=str, help="Experiment's output directory", required=True,
                      extended=False)
    args.add_argument(command="--verbose", name="verbose", type=int, help="Verbose level", default=2, extended=False)

    # Parse arguments
    args.parse()

    # Cuda
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # Corpus
    pan17corpus = nsNLP.data.Corpus(args.dataset)

    # Parameter space
    param_space = nsNLP.tools.ParameterSpace(args.get_space())

    # Experiment
    xp = nsNLP.tools.ResultManager \
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

    # Features: bag of character tensor
    boct = nsNLP.features.BagOfCharactersTensor(alphabet=alphabet, n_gram=2)

    # Iterate
    for space in param_space:
        # Params
        lr = float(space['lr'])
        momentum = float(space['momentum'])

        # Set experience state
        xp.set_state(space)

        # Average sample
        average_sample = np.array([])

        # Set sample
        xp.set_sample_state(0)

        # Convolutional Neural Network
        classifier = nsNLP.deep_models.CNNModel\
        (
            nsNLP.deep_models.modules.ConvNet(n_classes=2, params=[4800, 400]),
            classes=['female', 'male'],
            cuda=use_cuda,
            lr=lr,
            momentum=momentum,
            log_interval=10000
        )

        # 10 fold cross validation
        cross_validation = nsNLP.validation.CrossValidation(authors)

        # Average
        average_k_fold = np.zeros((args.k, args.epoch))

        # For each fold
        for k, (train_set, test_set) in enumerate(cross_validation):
            # For every author in training example
            for index, author in enumerate(train_set):
                # Get author's text
                text = author.get_texts()[0]

                # Print texts
                #print(text.x())
                print(boct(text.x()))
                exit()
                boct(text.x())
            # end for
        # end for
    # end for
# end if
