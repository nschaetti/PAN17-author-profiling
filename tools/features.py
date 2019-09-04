#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
import nsNLP
import sys
import torchlanguage.transforms
import os
import torch
import settings
from tools import load_glove_embeddings as gle
import numpy as np


# Function words
function_words = [u"a", u"about", u"above", u"after", u"after", u"again", u"against", u"ago", u"ahead",
                u"all",
                u"almost", u"along", u"already", u"also", u"although", u"always", u"am", u"among", u"an",
                u"and", u"any", u"are", u"aren't", u"around", u"as", u"at", u"away", u"backward",
                u"backwards", u"be", u"because", u"before", u"behind", u"below", u"beneath", u"beside",
                u"between", u"both", u"but", u"by", u"can", u"cannot", u"can't", u"cause", u"'cos",
                u"could",
                u"couldn't", u"'d", u"despite", u"did", u"didn't", u"do", u"does", u"doesn't", u"don't",
                u"down", u"during", u"each", u"either", u"even", u"ever", u"every", u"except", u"for",
                u"forward", u"from", u"had", u"hadn't", u"has", u"hasn't", u"have", u"haven't", u"he",
                u"her", u"here", u"hers", u"herself", u"him", u"himself", u"his", u"how", u"however",
                u"I",
                u"if", u"in", u"inside", u"inspite", u"instead", u"into", u"is", u"isn't", u"it", u"its",
                u"itself", u"just", u"'ll", u"least", u"less", u"like", u"'m", u"many", u"may",
                u"mayn't",
                u"me", u"might", u"mightn't", u"mine", u"more", u"most", u"much", u"must", u"mustn't",
                u"my", u"myself", u"near", u"need", u"needn't", u"needs", u"neither", u"never", u"no",
                u"none", u"nor", u"not", u"now", u"of", u"off", u"often", u"on", u"once", u"only",
                u"onto",
                u"or", u"ought", u"oughtn't", u"our", u"ours", u"ourselves", u"out", u"outside", u"over",
                u"past", u"perhaps", u"quite", u"'re", u"rather", u"'s", u"seldom", u"several", u"shall",
                u"shan't", u"she", u"should", u"shouldn't", u"since", u"so", u"some", u"sometimes",
                u"soon",
                u"than", u"that", u"the", u"their", u"theirs", u"them", u"themselves", u"then", u"there",
                u"therefore", u"these", u"they", u"this", u"those", u"though", u"through", u"thus",
                u"till",
                u"to", u"together", u"too", u"towards", u"under", u"unless", u"until", u"up", u"upon",
                u"us", u"used", u"usedn't", u"usen't", u"usually", u"'ve", u"very", u"was", u"wasn't",
                u"we", u"well", u"were", u"weren't", u"what", u"when", u"where", u"whether", u"which",
                u"while", u"who", u"whom", u"whose", u"why", u"will", u"with", u"without", u"won't",
                u"would", u"wouldn't", u"yet", u"you", u"your", u"yours", u"yourself", u"yourselves", u"X"]


# Create tokenizer
def create_tokenizer(tokenizer_type, lang="en_core_web_lg"):
    """
    Create tokenizer
    :param tokenizer_type: Tokenizer
    :return:
    """
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


# Create transformer
def create_transformer(feature, pretrained=False, path="", lang="en_vectors_web_lg", token2index=None):
    """
    Create the transformer
    :param feature:
    :param path:
    :param lang:
    :param n_gram:
    :return:
    """
    # ## Part-Of-Speech
    if "pos" in feature:
        return torchlanguage.transforms.Compose([
            torchlanguage.transforms.PartOfSpeech(model=lang),
            torchlanguage.transforms.ToIndex(start_ix=1),
            torchlanguage.transforms.Reshape((-1))
        ])
    # ## Function words
    elif "fw" in feature:
        if pretrained:
            return torchlanguage.transforms.Compose([
                torchlanguage.transforms.FunctionWord(model=lang, join=True),
                torchlanguage.transforms.GloveVector(model=lang),
                torchlanguage.transforms.Reshape((-1, 300))
            ])
        else:
            return torchlanguage.transforms.Compose([
                torchlanguage.transforms.FunctionWord(model=lang),
                torchlanguage.transforms.ToIndex(start_ix=1, token_to_ix=token2index),
                torchlanguage.transforms.Reshape((-1))
            ])
        # end if
    # ## Word Vector
    elif "wv" in feature:
        if pretrained:
            return torchlanguage.transforms.Compose([
                torchlanguage.transforms.Token(model=lang),
                torchlanguage.transforms.GensimModel(
                    model_path=os.path.join(path, 'word2vec', "embedding.en.bin")
                ),
                torchlanguage.transforms.Reshape((-1, 300))
            ])
        else:
            return torchlanguage.transforms.Compose([
                torchlanguage.transforms.Token(model=lang),
                torchlanguage.transforms.ToIndex(start_ix=1, token_to_ix=token2index),
                torchlanguage.transforms.Reshape((-1))
            ])
        # end if
    # ## Character embedding
    elif "c1" in feature:
        if pretrained:
            token_to_ix, embedding_weights = load_character_embedding(path)
            embedding_dim = embedding_weights.size(1)
            return torchlanguage.transforms.Compose([
                torchlanguage.transforms.Character(),
                torchlanguage.transforms.ToIndex(token_to_ix=token_to_ix),
                torchlanguage.transforms.Embedding(weights=embedding_weights, voc_size=len(token_to_ix)),
                torchlanguage.transforms.Reshape((-1, embedding_dim))
            ])
        else:
            return torchlanguage.transforms.Compose([
                torchlanguage.transforms.Character(),
                torchlanguage.transforms.ToIndex(start_ix=1, token_to_ix=token2index),
                torchlanguage.transforms.Reshape((-1))
            ])
        # end if
    # ## Character 2-gram embedding
    elif "c2" in feature:
        if pretrained:
            token_to_ix, embedding_weights = load_character_embedding(path)
            embedding_dim = embedding_weights.size(1)
            return torchlanguage.transforms.Compose([
                torchlanguage.transforms.Character2Gram(overlapse=False),
                torchlanguage.transforms.ToIndex(token_to_ix=token_to_ix),
                torchlanguage.transforms.Embedding(weights=embedding_weights, voc_size=len(token_to_ix)),
                torchlanguage.transforms.Reshape((-1, embedding_dim))
            ])
        else:
            return torchlanguage.transforms.Compose([
                torchlanguage.transforms.Character2Gram(overlapse=True),
                torchlanguage.transforms.ToIndex(start_ix=1, token_to_ix=token2index),
                torchlanguage.transforms.Reshape((-1))
            ])
        # end if
    # ## Character 3-gram embedding
    elif "c3" in feature:
        if pretrained:
            token_to_ix, embedding_weights = load_character_embedding(path)
            embedding_dim = embedding_weights.size(1)
            return torchlanguage.transforms.Compose([
                torchlanguage.transforms.Character3Gram(overlapse=False),
                torchlanguage.transforms.ToIndex(token_to_ix=token_to_ix),
                torchlanguage.transforms.Embedding(weights=embedding_weights, voc_size=len(token_to_ix)),
                torchlanguage.transforms.Reshape((-1, embedding_dim))
            ])
        else:
            return torchlanguage.transforms.Compose([
                torchlanguage.transforms.Character3Gram(overlapse=True),
                torchlanguage.transforms.ToIndex(start_ix=1, token_to_ix=token2index),
                torchlanguage.transforms.Reshape((-1))
            ])
        # end if
    # ## Character CNN
    elif "ce" in feature:
        return torchlanguage.transforms.Compose([
            torchlanguage.transforms.Character(),
            torchlanguage.transforms.ToIndex(),
            torchlanguage.transforms.ToNGram(n=settings.ce_text_length, overlapse=False),
            torchlanguage.transforms.Reshape((-1, settings.ce_text_length))
        ])
    else:
        raise NotImplementedError(u"Feature type {} not implemented".format(feature))
    # end if
# end create_transformer


# Load character embedding
def load_character_embedding(emb_path):
    """
    Load character embedding
    :param emb_path:
    :return:
    """
    token_to_ix, weights = torch.load(open(emb_path, 'rb'))
    return token_to_ix, weights
# end load_character_embedding


# Load pretrained weights
def load_pretrained_weights(feature, emb_path, embedding_size):
    """
    Load pretrained weights
    :param emb_path:
    :return:
    """
    if feature == "wv":
        # Load GloVe
        word2index, embedding_matrix = gle.load_glove_embeddings(
            fp=emb_path,
            embedding_dim=embedding_size
        )
        pretrained_vocsize = embedding_matrix.shape[0]
        return word2index, embedding_matrix, pretrained_vocsize
    elif feature == "fw":
        # Load GloVe
        w2i, glove_matrix = gle.load_glove_embeddings(
            fp=emb_path,
            embedding_dim=embedding_size
        )
        embedding_matrix = np.zeros((196, embedding_size))
        word2index = {}
        index = 1
        for w in w2i.keys():
            if w in function_words:
                word2index[w] = index
                embedding_matrix[index] = glove_matrix[w2i[w]]
                index += 1
            # end if
        # end for
        pretrained_vocsize = embedding_matrix.shape[0]
        return word2index, embedding_matrix, pretrained_vocsize
    else:
        token_to_ix, embedding_weights = load_character_embedding(emb_path)
        embedding_matrix = np.zeros((embedding_weights.shape[0]+1, embedding_weights.shape[1]))
        embedding_matrix[1:] = embedding_weights.numpy()
        word2index = {}
        for k in token_to_ix.keys():
            word2index[k] = token_to_ix[k] + 1
        # end for
        pretrained_vocsize = embedding_matrix.shape[0]
        return word2index, embedding_matrix, pretrained_vocsize
    # end if
# end load_pretrained_weights
