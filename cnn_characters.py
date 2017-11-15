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
# Main function
###########################


# Create CNN
def create_cnn(n_grams, conv1_size, max_pool1_size, conv2_size, max_pool2_size, linear1_size, linear2_size, kernel_size, stride_size):
    """
    Create CNN
    :param n_grams:
    :param conv1_size:
    :param max_pool1_size:
    :param conv2_size:
    :param max_pool2_size:
    :param linear1_size:
    :param linear2_size:
    :param kernel_size:
    :param stride_size:
    :return:
    """
    # CNN
    if n_grams == 2:
        convnet = nsNLP.deep_models.modules.ConvNet \
        (
            n_classes=2,
            channels=(1, conv1_size, max_pool1_size, conv2_size, max_pool2_size, linear1_size, linear2_size),
            kernel_size=kernel_size,
            stide=stride_size
        )
    elif n_grams == 3:
        convnet = nsNLP.deep_models.modules.ConvNet3D \
        (
            n_classes=2,
            channels=(1, conv1_size, max_pool1_size, conv2_size, max_pool2_size, linear1_size, linear2_size),
            kernel_size=kernel_size,
            stide=stride_size
        )
    # end if
    return convnet
# end create_cnn

###########################
# Main function
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
    args.add_argument(command="--n-grams", name="n_grams", type=int, help="Gram model", extended=True,
                      default=1)
    args.add_argument(command="--conv1", name="conv1", type=int, help="Number of channels in first layer",
                      extended=True, default=10)
    args.add_argument(command="--conv2", name="conv2", type=int, help="Number of channels in second layer",
                      extended=True, default=20)
    args.add_argument(command="--max-pool1", name="max_pool1", type=int, help="Size of the first max pooling layer",
                      extended=True, default=2)
    args.add_argument(command="--max-pool2", name="max_pool2", type=int, help="Size of the second max pooling layer",
                      extended=True, default=2)
    args.add_argument(command="--linear1", name="linear1", type=int, help="Size of the first linear layer",
                      extended=True, default=4800)
    args.add_argument(command="--linear2", name="linear2", type=int, help="Size of the second linear layer",
                      extended=True, default=50)
    args.add_argument(command="--kernel-size", name="kernel_size", type=int, help="Kernel size",
                      extended=True, default=5)
    args.add_argument(command="--stride", name="stride", type=int, help="Stride",
                      extended=True, default=1)

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
    boct = nsNLP.features.BagOfCharactersTensor(alphabet=alphabet, n_gram=args.n_grams)

    # Iterate
    for space in param_space:
        # Params
        lr = float(space['lr'])
        momentum = float(space['momentum'])
        n_grams = int(space['n_grams'])
        conv1_size = int(space['conv1'])
        conv2_size = int(space['conv2'])
        max_pool1_size = int(space['max_pool1_size'])
        max_pool2_size = int(space['max_pool2_size'])
        linear1_size = int(space['linear1_size'])
        linear2_size = int(space['linear2_size'])
        kernel_size = int(space['kernel_size'])
        stride_size = int(space['stride'])

        # Set experience state
        xp.set_state(space)

        # Average sample
        average_sample = np.array([])

        # Set sample
        xp.set_sample_state(0)

        # 10 fold cross validation
        cross_validation = nsNLP.validation.CrossValidation(authors)

        # Average
        average_k_fold = np.zeros((args.k, args.epoch))

        # For each fold
        for k, (train_set, test_set) in enumerate(cross_validation):
            # CNN
            convnet = create_cnn(args.n_grams, conv1_size, max_pool1_size, conv2_size, max_pool2_size, linear1_size,
                                 linear2_size, kernel_size, stride_size)

            # CNN Model
            classifier = nsNLP.deep_models.CNNModel\
            (
                convnet,
                classes=['female', 'male'],
                cuda=use_cuda,
                lr=lr,
                momentum=momentum,
                log_interval=10000
            )

            # Torch sets
            torch_training_set = list()
            torch_test_set = list()

            # For every author in training example
            for index, author in enumerate(train_set):
                # Get author's text
                text = author.get_texts()[0]

                # Add to training set
                torch_training_set.append((boct(text.x()), author.truth('gender')))
            # end for

            # For every author in the test set
            for index, author in enumerate(test_set):
                # Get author's text
                text = author.get_texts()[0]

                # Add to test set
                torch_test_set.append((boct(text.x()), author.truth('gender')))
            # end for

            # Train with each document
            for epoch in range(1, args.epoch+1):
                classifier.train(epoch, torch_training_set, batch_size=args.batch_size)
                success_rate, test_loss = classifier.test(epoch, torch_test_set, batch_size=args.batch_size)
                average_k_fold[k, epoch] = success_rate
            # end for

            # Delete classifier
            del classifier
        # end for

        # Get max average for each fold
        best_epoch = np.argmax(np.average(average_k_fold, axis=0))

        # Register each fold
        for k in range(args.k):
            xp.set_fold_state(k)
            xp.add_result(average_k_fold[k, best_epoch])
        # end for
    # end for
# end if