#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
import torchlanguage.transforms
import torch
import settings


#########################################
# Dataset
#########################################


# Load PAN17 dataset
def load_pan17_dataset(k=10):
    """
    Load PAN 17 dataset
    :param k:
    :return:
    """
    # Load
    pan17_dataset = torchlanguage.datasets.PAN17AuthorProfiling()

    # Training
    pan17_loader_train = torch.utils.data.DataLoader(
        torchlanguage.utils.CrossValidationWithDev(pan17_dataset, train='train', k=k),
        batch_size=1,
        shuffle=False
    )

    # Validation
    pan17_loader_dev = torch.utils.data.DataLoader(
        torchlanguage.utils.CrossValidationWithDev(pan17_dataset, train='dev', k=k),
        batch_size=1,
        shuffle=False
    )

    # Test
    pan17_loader_test = torch.utils.data.DataLoader(
        torchlanguage.utils.CrossValidationWithDev(pan17_dataset, train='test', k=k),
        batch_size=1,
        shuffle=False
    )

    return pan17_dataset, pan17_loader_train, pan17_loader_dev, pan17_loader_test
# end load_pan17_dataset


# Load dataset
def load_dataset(dataset_size=100, dataset_start=0, shuffle=True, sentence_level=False, n_authors=15, k=5, features=u""):
    """
    Load dataset
    :return:
    """
    # Load from directory
    if sentence_level:
        reutersc50_dataset = torchlanguage.datasets.ReutersC50SentenceDataset(
            n_authors=n_authors,
            download=True,
            dataset_size=dataset_size,
            dataset_start=dataset_start
        )
    else:
        reutersc50_dataset = torchlanguage.datasets.ReutersC50Dataset(
            n_authors=n_authors,
            download=True,
            dataset_size=dataset_size,
            dataset_start=dataset_start,
            load_features=features
        )
    # end if

    # Reuters C50 dataset training
    reuters_loader_train = torch.utils.data.DataLoader(
        torchlanguage.utils.CrossValidationWithDev(reutersc50_dataset, train='train', k=k),
        batch_size=1,
        shuffle=shuffle
    )

    # Reuters C50 dataset dev
    reuters_loader_dev = torch.utils.data.DataLoader(
        torchlanguage.utils.CrossValidationWithDev(reutersc50_dataset, train='dev', k=k),
        batch_size=1,
        shuffle=shuffle
    )

    # Reuters C50 dataset test
    reuters_loader_test = torch.utils.data.DataLoader(
        torchlanguage.utils.CrossValidationWithDev(reutersc50_dataset, train='test', k=k),
        batch_size=1,
        shuffle=shuffle
    )

    return reutersc50_dataset, reuters_loader_train, reuters_loader_dev, reuters_loader_test
# end load_dataset


# Load pretrain dataset
def load_pretrain_dataset():
    """
    Load dataset
    :return:
    """
    # Load from directory
    reutersc50_dataset = torchlanguage.datasets.ReutersC50Dataset(
        n_authors=35,
        download=True,
        dataset_size=100,
        dataset_start=0,
        authors=settings.pretrain_authors
    )

    # Reuters C50 dataset training
    reuters_loader_train = torch.utils.data.DataLoader(
        torchlanguage.utils.CrossValidation(reutersc50_dataset),
        batch_size=1,
        shuffle=True
    )

    # Reuters C50 dataset test
    reuters_loader_test = torch.utils.data.DataLoader(
        torchlanguage.utils.CrossValidation(reutersc50_dataset, train=False),
        batch_size=1,
        shuffle=True
    )
    return reutersc50_dataset, reuters_loader_train, reuters_loader_test
# end load_dataset
