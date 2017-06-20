#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
from dataset.Dataset import Dataset


if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="PAN17 - Test dataset loader")

    # Argument
    parser.add_argument("--dataset", type=str, help="Dataset directory", default=".", required=True)
    parser.add_argument("--log-level", type=int, help="Log level", default=20)
    args = parser.parse_args()

    # Logs
    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger(name="PAN17")

    # Load data set
    data_set = Dataset().open(args.dataset)

    # For each author
    for author in data_set:
        print(author.get_id())
        print(author.get_gender())
        print(author.get_variety())
        for document in author:
            #print(document.get_text())
            for word in document.get_words():
                print(word)
                print(word.vector)
            # end for
        # end for
        exit()
    # end for

# end if
