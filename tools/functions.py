#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
import nsNLP
import sys


# Create tokenizer
def create_tokenizer(tokenizer_type, lang="en_core_web_lg"):
    """
    Create tokenizer
    :param tokenizer_type: Tokenizer
    :return:
    """
    print(lang)
    # Tokenizer
    if tokenizer_type == "nltk":
        tokenizer = nsNLP.tokenization.NLTKTokenizer()
    elif tokenizer_type == "nltk-twitter":
        tokenizer = nsNLP.tokenization.NLTKTweetTokenizer()
    elif tokenizer_type == "spacy":
        tokenizer = nsNLP.tokenization.SpacyTokenizer(lang=lang)
    elif tokenizer_type == "spacy_wv":
        tokenizer = nsNLP.tokenization.SpacyTokenizer(lang=lang, original=True)
    else:
        sys.stderr.write(u"Unknown tokenizer type!\n")
        exit()
    # end if

    # Return tokenizer object
    return tokenizer
# end create_tokenizer


# Get params
def get_params(space):
    """
    Get params
    :param space:
    :return:
    """
    # Params
    hidden_size = int(space['hidden_size'])
    cell_size = int(space['cell_size'])
    feature = space['feature'][0][0]
    lang = space['lang'][0][0]
    dataset_start = space['dataset_start']
    window_size = int(space['window_size'])
    learning_window = int(space['learning_window'])
    embedding_size = int(space['embedding_size'])
    rnn_type = space['rnn_type'][0][0]
    num_layers = int(space['num_layers'])
    dropout = float(space['dropout'])
    output_dropout = float(space['output_dropout'])

    return hidden_size, cell_size, feature, lang, dataset_start, window_size, learning_window, embedding_size, rnn_type, num_layers, dropout, output_dropout
# end get_params

