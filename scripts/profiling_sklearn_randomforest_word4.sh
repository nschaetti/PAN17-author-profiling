#!/bin/bash

python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "word" --n-grams-min 1 --n-grams-max 4 --criterion "entropy" --n-estimators 200 --tfidf 'tfidf' --name "AP PAN17 sklearn RF word 1-4grams entro NE-200 no-tfidf" --description "" --output outputs/ --verbose 4
