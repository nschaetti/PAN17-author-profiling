#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Dropout, Activation, Embedding, GRU, GlobalAveragePooling1D, AveragePooling1D


class GlobalAveragePooling1DMasked(GlobalAveragePooling1D):
    def call(self, x, mask=None):
        if mask is not None:
            return keras.backend.sum(x, axis=1) / keras.backend.sum(keras.backend.cast(mask, 'float32'))
        else:
            return super(self).call(x)
        # end if
    # end call
# end GlobalAveragePooling1DMasked


# Create RNN model with an pre-trained embedding layer
def create_rnn_model_with_pretrained_embedding_layer(rnn_type, embedding_matrix, hidden_size, dense_size, trainable=False, average=True, level=1, use_dropout=True, dropout=0.25):
    """
    Create RNN model with an pre-trained embedding layer
    :param rnn_type:
    :param hidden_size:
    :param dense_size:
    :param level:
    :return:
    """
    # Model
    model = Sequential()

    # Embedding
    model.add(Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=embedding_matrix.shape[1],
        trainable=trainable,
        name='embedding_layer',
        mask_zero=False)
    )

    # Add first RNN
    if rnn_type == 'LSTM':
        model.add(LSTM(hidden_size, return_sequences=True, input_shape=(None, embedding_matrix.shape[1])))
    else:
        model.add(GRU(hidden_size, return_sequences=True, input_shape=(None, embedding_matrix.shape[1])))
    # end if

    # Add other LSTM
    if level > 1:
        for i in range(level-1):
            # Dropout
            if use_dropout:
                model.add(Dropout(dropout))
            # end if

            # Add RNN layer
            if rnn_type == 'LSTM':
                model.add(LSTM(hidden_size, return_sequences=True))
            else:
                model.add(GRU(hidden_size, return_sequences=True))
            # end if
        # end for
    # end if

    # Dropout
    if use_dropout:
        model.add(Dropout(dropout))
    # end if

    # Add time distributed (average over time) with dense layer
    model.add(TimeDistributed(Dense(dense_size)))

    # Average pooling
    if average:
        model.add(GlobalAveragePooling1DMasked())
    # end if

    # Softmax output
    model.add(Activation('softmax'))

    return model
# end create_rnn_model_with_pretrained_embedding_layer


# Create RNN model with an embedding layer
def create_rnn_model_with_embedding_layer(rnn_type, voc_size, embedding_size, hidden_size, dense_size, average=True, level=1, use_dropout=True, dropout=0.25):
    """
    Create RNN model with an embedding layer
    :param rnn_type:
    :param hidden_size:
    :param dense_size:
    :param level:
    :return:
    """
    # Model
    model = Sequential()

    # Embedding
    model.add(Embedding(voc_size, embedding_size, mask_zero=True))

    # Add first RNN
    if rnn_type == 'LSTM':
        model.add(LSTM(hidden_size, return_sequences=True, input_shape=(None, embedding_size)))
    else:
        model.add(GRU(hidden_size, return_sequences=True, input_shape=(None, embedding_size)))
    # end if

    # Add other LSTM
    if level > 1:
        for i in range(level-1):
            # Dropout
            if use_dropout:
                model.add(Dropout(dropout))
            # end if

            # Add RNN layer
            if rnn_type == 'LSTM':
                model.add(LSTM(hidden_size, return_sequences=True))
            else:
                model.add(GRU(hidden_size, return_sequences=True))
            # end if
        # end for
    # end if

    # Dropout
    if use_dropout:
        model.add(Dropout(dropout))
    # end if

    # Add time distributed (average over time) with dense layer
    model.add(TimeDistributed(Dense(dense_size)))

    # Average pooling
    if average:
        model.add(GlobalAveragePooling1DMasked())
    # end if

    # Softmax output
    model.add(Activation('softmax'))

    return model
# end create_rnn_model_with_embedding_layer


# Create RNN model with direct input
def create_rnn_model(rnn_type, embedding_size, hidden_size, dense_size, average=True, level=1, use_dropout=True, dropout=0.25):
    """
    Create RNN model with direct input
    :param rnn_type:
    :param hidden_size:
    :param dense_size:
    :param level:
    :return:
    """
    # Model
    model = Sequential()

    # Add first RNN
    if rnn_type == 'LSTM':
        model.add(LSTM(hidden_size, return_sequences=True, input_shape=(None, embedding_size)))
    else:
        model.add(GRU(hidden_size, return_sequences=True, input_shape=(None, embedding_size)))
    # end if

    # Add other LSTM
    if level > 1:
        for i in range(level-1):
            # Dropout
            if use_dropout:
                model.add(Dropout(dropout))
            # end if

            # Add RNN layer
            if rnn_type == 'LSTM':
                model.add(LSTM(hidden_size, return_sequences=True))
            else:
                model.add(GRU(hidden_size, return_sequences=True))
            # end if
        # end for
    # end if

    # Dropout
    if use_dropout:
        model.add(Dropout(dropout))
    # end if

    # Add time distributed (average over time) with dense layer
    model.add(TimeDistributed(Dense(dense_size)))

    # Average pooling
    if average:
        model.add(GlobalAveragePooling1DMasked())
    # end if

    # Softmax output
    model.add(Activation('softmax'))

    return model
# end create_rnn_model


# Create Stanford article level model
def create_stanford_article_level_model(rnn_type, embedding_matrix, sentence_size, hidden_size, dense_size, trainable=False, use_dropout=True, dropout=0.5):
    """
    Create RNN model
    :param rnn_type:
    :param hidden_size:
    :param dense_size:
    :return:
    """
    # Model
    model = Sequential()

    # Embedding
    model.add(Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=embedding_matrix.shape[1],
        trainable=trainable,
        name='embedding_layer',
        mask_zero=True)
    )

    # Average pool over sentences
    """model.add(AveragePooling1D(
        pool_size=sentence_size,
        strides=sentence_size
    ))"""

    # Add first RNN
    if rnn_type == 'LSTM':
        model.add(LSTM(hidden_size, return_sequences=True, input_shape=(None, embedding_matrix.shape[1])))
    else:
        model.add(GRU(hidden_size, return_sequences=True, input_shape=(None, embedding_matrix.shape[1])))
    # end if

    # Dropout
    if use_dropout:
        model.add(Dropout(dropout))
    # end if

    # Average pooling
    model.add(GlobalAveragePooling1DMasked())

    # Add time distributed (average over time) with dense layer
    # model.add(TimeDistributed(Dense(dense_size)))

    # Add fully-connected layer
    model.add(Dense(dense_size))

    # Softmax output
    model.add(Activation('softmax'))

    return model
# end create_sentence_level_model
