#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
import nsNLP
import torch


#########################################
# Argument parsing
#########################################

# Training argument
def parser_training():
    """
    Training argument
    :return:
    """
    # Argument builder
    args = nsNLP.tools.ArgumentBuilder(desc=u"Argument test")

    # Dataset arguments
    args.add_argument(command="--dataset", name="dataset", type=str, default="data/",
                      help="JSON file with the file description for each authors", required=False, extended=False)
    args.add_argument(command="--dataset-size", name="dataset_size", type=float,
                      help="Ratio of the data set to use (100 percent by default)", extended=False, default=100.0)
    args.add_argument(command="--dataset-start", name="dataset_start", type=float,
                      help="Where to start in the data set", extended=True, default=0)
    args.add_argument(command="--k", name="k", type=int, help="K-Fold Cross Validation", extended=False, default=10)
    args.add_argument(command="--inverse-dev-test", name="inverse_dev_test", action='store_true', help="Inverse Dev Test", extended=False, default=10)

    # Author parameters
    args.add_argument(command="--n-authors", name="n_authors", type=int,
                      help="Number of authors to include in the test", default=15, extended=False)
    for i in range(15):
        args.add_argument(command="--author{}".format(i), name="author{}".format(i), type=str,
                          help="{}th author to test".format(i), extended=False)
    # end for

    # LSTM/GRU arguments
    args.add_argument(command="--rnn-type", name="rnn_type", type=str, help="Type of RNN (rnn, lstm, gru)", default='lstm', extended=True)
    args.add_argument(command="--hidden-size", name="hidden_size", type=float, help="Size of the hidden vector",
                      required=True, extended=True)
    args.add_argument(command="--cell-size", name="cell_size", type=float, help="Size of the cell vector",
                      required=False, extended=True)
    args.add_argument(command="--embedding-size", name="embedding_size", type=float, help="Size of the embedding",
                      required=True, extended=True)
    args.add_argument(command="--learning-window", name="learning_window", type=int, help="Size of the learning window",
                      required=False, extended=True)
    args.add_argument(command="--num-layers", name="num_layers", type=int, help="Number of recurrent layers", default=1, extended=True)
    args.add_argument(command="--dropout", name="dropout", type=float, help="Dropout", default=0, extended=True)
    args.add_argument(command="--output-dropout", name="output_dropout", type=float, help="Output dropout", default=0, extended=True)
    args.add_argument(command="--feature", name="feature", type=str,
                      help="The text transformer to use (fw, pos, tag, wv, c1, c2, c3, cnn)", default='wv',
                      extended=True)
    args.add_argument(command="--pretrained", name="pretrained", action='store_true', help="Use pretrained layer or not", default=False, extended=False)
    args.add_argument(command="--fine-tuning", name="fine_tuning", action='store_true',
                      help="Fine tune the pretrained layer ?", default=False, extended=False)
    args.add_argument(command="--voc-size", name="voc_size", type=int, help="Voc. size",
                      default=30000, extended=False)
    args.add_argument(command="--n-layers", name="n_layers", type=int, help="Number of recurrent layers", extended=True,
                      default="1")
    args.add_argument(command="--embedding-path", name="embedding_path", type=str, help="Embedding directory",
                      default='~/Projets/TURING/Datasets/', extended=False)
    args.add_argument(command="--batch-size", name="batch_size", type=int, help="Size of the batches",
                      default=64, extended=False)
    args.add_argument(command="--test-batch-size", name="test_batch_size", type=int, help="Size of the test batches",
                      default=64, extended=False)
    args.add_argument(command="--max-length", name="max_length", type=int, help="Maximum sequence length",
                      required=False, extended=False)
    args.add_argument(command="--precomputed-features", name="precomputed_features", type=str, help="Precomputed features", required=False, extended=False)

    # Tokenizer and word vector parameters
    args.add_argument(command="--tokenizer", name="tokenizer", type=str,
                      help="Which tokenizer to use (spacy, nltk, spacy-tokens)", default='nltk', extended=False)
    args.add_argument(command="--lang", name="lang", type=str, help="Tokenizer language parameters",
                      default='en_vectors_web_lg', extended=True)

    # Experiment output parameters
    args.add_argument(command="--epoch", name="epoch", type=int, help="How many epoch",
                      extended=False, required=False, default=30)
    args.add_argument(command="--early-stopping", name="early_stopping", type=int, help="How many epoch to wait if no improvement?",
                      extended=False, required=False, default=30)
    args.add_argument(command="--window-size", name="window_size", type=str, help="Window size for prediction",
                      extended=True, required=False, default=0)
    args.add_argument(command="--measure", name="measure", type=str, help="Which measure to test (global/local)", extended=False, required=False, default='global')
    args.add_argument(command="--name", name="name", type=str, help="Experiment's name", extended=False, required=True)
    args.add_argument(command="--description", name="description", type=str, help="Experiment's description",
                      extended=False, required=True)
    args.add_argument(command="--output", name="output", type=str, help="Experiment's output directory", required=True,
                      extended=False)
    args.add_argument(command="--sentence", name="sentence", action='store_true',
                      help="Test sentence classification rate?", default=False, extended=False)
    args.add_argument(command="--n-samples", name="n_samples", type=int, help="Number of different reservoir to test",
                      default=1, extended=False)
    args.add_argument(command="--verbose", name="verbose", type=int, help="Verbose level", default=2, extended=False)
    args.add_argument(command="--cuda", name="cuda", action='store_true',
                      help="Use CUDA?", default=False, extended=False)
    args.add_argument(command="--certainty", name="certainty", type=str, help="Save certainty data", default="", extended=False)

    # Parse arguments
    args.parse()

    # CUDA
    use_cuda = torch.cuda.is_available() if args.cuda else False

    # Parameter space
    param_space = nsNLP.tools.ParameterSpace(args.get_space())

    # Experiment
    xp = nsNLP.tools.ResultManager \
            (
            args.output,
            args.name,
            args.description,
            args.get_space(),
            args.n_samples,
            args.k,
            verbose=args.verbose
        )

    return args, use_cuda, param_space, xp
# end parser_training