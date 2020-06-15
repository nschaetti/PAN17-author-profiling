#!/usr/bin/env bash

# SVM (linear) word 1-2 grams TF-IDF
# python3 sklearn_classifier_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --n-grams-min 1 --n-grams-max 2 --kernel "linear" --penalty 1.0 --tfidf 'tfidf' --name "AP PAN17 sklearn SVM linear word 1-2grams P1.0 TFIDF" --description "Author Profiling PAN17 sklearn Linear SVM word 1-2-grams penalty 1.0 tfidf" --output outputs/ --verbose 4

# SVM poly-3 character 1-4 grams TF-IDF
python3 sklearn_classifier_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 4 --kernel "linear" --penalty 1.0 --tfidf 'tfidf' --name "AP PAN17 sklearn SVM linear char 1-4grams P1.0 TFIDF" --description "Author Profiling PAN17 sklearn Linear SVM char 1-2-grams penalty 1.0 tfidf" --output outputs/ --verbose 4

# Random Forest word 1-4 grams 200 trees
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "word" --n-grams-min 1 --n-grams-max 4 --criterion "entropy" --n-estimators 200 --tfidf 'none' --name "AP PAN17 sklearn RF word 1-4grams entro NE-200 no-tfidf" --description "" --output outputs/ --verbose 4

# Random Forest character 1-4 grams 200 trees
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 4 --criterion "entropy" --n-estimators 200 --tfidf 'none' --name "AP PAN17 sklearn RF char 1-4grams entro NE-200 no-tfidf" --description "" --output outputs/ --verbose 4
