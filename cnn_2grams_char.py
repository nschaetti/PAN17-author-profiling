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

# Accepted characters
accepted_characters = u""


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

    # Features: letter statistics
    bol = nsNLP.features.LetterStatistics()

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
            nsNLP.deep_models.modules.ConvNet(),
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

        # end for
    # end for

    # Load data set
    with open(args.file, 'r') as f:
        # Load
        print("Loading data set %s" % args.file)
        data_set = pickle.load(f)

        # Sample size
        n_samples = len(data_set['2grams'])
        fold_size = int(math.ceil(n_samples / 10.0))

        # Get truths
        truths = []
        for truth in data_set['labels']:
            truths += [truth[0]]
        # end for

        # Deep-Learning model
        deep_learning_model = PAN17DeepNNModel(PAN17ConvNet(n_classes=2, params=params[args.lang]), classes=("male", "female"),
                                               cuda=args.cuda, lr=args.lr, momentum=args.momentum,
                                               log_interval=args.log_interval, seed=args.seed)

        # K-10 fold
        grams_set = np.array(data_set['2grams'])
        m_height = grams_set.shape[1]
        m_width = grams_set.shape[2]
        truths_set = np.array(truths)
        grams_set.shape = (10, fold_size, m_height, m_width)
        truths_set.shape = (10, fold_size)

        # Select training and test sets
        test = grams_set[-1]
        test_truths = truths_set[-1]
        training = np.delete(grams_set, -1, axis=0)
        training_truths = np.delete(truths_set, -1, axis=0)
        training.shape = (fold_size * 9, m_height, m_width)
        training_truths.shape = (fold_size * 9)

        # Data set
        print("To Torch Tensors....")
        tr_data_set = deep_learning_model.to_torch_data_set(training, training_truths)
        te_data_set = deep_learning_model.to_torch_data_set(test, test_truths)

        # Train with each document
        print("Assessing CNN model...")
        mini = 0
        maxi = 10000000
        for epoch in range(1, args.epoch+1):
            deep_learning_model.train(epoch, tr_data_set, batch_size=args.batch_size)
            success_rate, test_loss = deep_learning_model.test(epoch, te_data_set, batch_size=args.batch_size)
            if test_loss < maxi:
                print("Saving model to %s" % args.output)
                maxi = test_loss
                deep_learning_model.save(args.output)
            # end if
        # end for
    # end with
# end if