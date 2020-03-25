#!/bin/bash

# Random Forest
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 1 --criterion "entropy" --n-estimators 25 --tfidf 'none' --name "AP PAN17 sklearn RF char 1-1grams entro NE-25 no-tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 2 --criterion "entropy" --n-estimators 25 --tfidf 'none' --name "AP PAN17 sklearn RF char 1-2grams entro NE-25 no-tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 3 --criterion "entropy" --n-estimators 25 --tfidf 'none' --name "AP PAN17 sklearn RF char 1-3grams entro NE-25 no-tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 4 --criterion "entropy" --n-estimators 25 --tfidf 'none' --name "AP PAN17 sklearn RF char 1-4grams entro NE-25 no-tfidf" --description "" --output outputs/ --verbose 4

python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 1 --criterion "entropy" --n-estimators 50 --tfidf 'none' --name "AP PAN17 sklearn RF char 1-1grams entro NE-50 no-tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 2 --criterion "entropy" --n-estimators 50 --tfidf 'none' --name "AP PAN17 sklearn RF char 1-2grams entro NE-50 no-tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 3 --criterion "entropy" --n-estimators 50 --tfidf 'none' --name "AP PAN17 sklearn RF char 1-3grams entro NE-50 no-tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 4 --criterion "entropy" --n-estimators 50 --tfidf 'none' --name "AP PAN17 sklearn RF char 1-4grams entro NE-50 no-tfidf" --description "" --output outputs/ --verbose 4

python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 1 --criterion "entropy" --n-estimators 100 --tfidf 'none' --name "AP PAN17 sklearn RF char 1-1grams entro NE-100 no-tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 2 --criterion "entropy" --n-estimators 100 --tfidf 'none' --name "AP PAN17 sklearn RF char 1-2grams entro NE-100 no-tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 3 --criterion "entropy" --n-estimators 100 --tfidf 'none' --name "AP PAN17 sklearn RF char 1-3grams entro NE-100 no-tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 4 --criterion "entropy" --n-estimators 100 --tfidf 'none' --name "AP PAN17 sklearn RF char 1-4grams entro NE-100 no-tfidf" --description "" --output outputs/ --verbose 4

python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 1 --criterion "entropy" --n-estimators 200 --tfidf 'none' --name "AP PAN17 sklearn RF char 1-1grams entro NE-200 no-tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 2 --criterion "entropy" --n-estimators 200 --tfidf 'none' --name "AP PAN17 sklearn RF char 1-2grams entro NE-200 no-tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 3 --criterion "entropy" --n-estimators 200 --tfidf 'none' --name "AP PAN17 sklearn RF char 1-3grams entro NE-200 no-tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 4 --criterion "entropy" --n-estimators 200 --tfidf 'none' --name "AP PAN17 sklearn RF char 1-4grams entro NE-200 no-tfidf" --description "" --output outputs/ --verbose 4


python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 1 --criterion "gini" --n-estimators 25 --tfidf 'none' --name "AP PAN17 sklearn RF char 1-1grams gini NE-25 no-tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 2 --criterion "gini" --n-estimators 25 --tfidf 'none' --name "AP PAN17 sklearn RF char 1-2grams gini NE-25 no-tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 3 --criterion "gini" --n-estimators 25 --tfidf 'none' --name "AP PAN17 sklearn RF char 1-3grams gini NE-25 no-tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 4 --criterion "gini" --n-estimators 25 --tfidf 'none' --name "AP PAN17 sklearn RF char 1-4grams gini NE-25 no-tfidf" --description "" --output outputs/ --verbose 4

python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 1 --criterion "gini" --n-estimators 50 --tfidf 'none' --name "AP PAN17 sklearn RF char 1-1grams gini NE-50 no-tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 2 --criterion "gini" --n-estimators 50 --tfidf 'none' --name "AP PAN17 sklearn RF char 1-2grams gini NE-50 no-tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 3 --criterion "gini" --n-estimators 50 --tfidf 'none' --name "AP PAN17 sklearn RF char 1-3grams gini NE-50 no-tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 4 --criterion "gini" --n-estimators 50 --tfidf 'none' --name "AP PAN17 sklearn RF char 1-4grams gini NE-50 no-tfidf" --description "" --output outputs/ --verbose 4

python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 1 --criterion "gini" --n-estimators 100 --tfidf 'none' --name "AP PAN17 sklearn RF char 1-1grams gini NE-100 no-tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 2 --criterion "gini" --n-estimators 100 --tfidf 'none' --name "AP PAN17 sklearn RF char 1-2grams gini NE-100 no-tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 3 --criterion "gini" --n-estimators 100 --tfidf 'none' --name "AP PAN17 sklearn RF char 1-3grams gini NE-100 no-tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 4 --criterion "gini" --n-estimators 100 --tfidf 'none' --name "AP PAN17 sklearn RF char 1-4grams gini NE-100 no-tfidf" --description "" --output outputs/ --verbose 4

python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 1 --criterion "gini" --n-estimators 200 --tfidf 'none' --name "AP PAN17 sklearn RF char 1-1grams gini NE-200 no-tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 2 --criterion "gini" --n-estimators 200 --tfidf 'none' --name "AP PAN17 sklearn RF char 1-2grams gini NE-200 no-tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 3 --criterion "gini" --n-estimators 200 --tfidf 'none' --name "AP PAN17 sklearn RF char 1-3grams gini NE-200 no-tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 4 --criterion "gini" --n-estimators 200 --tfidf 'none' --name "AP PAN17 sklearn RF char 1-4grams gini NE-200 no-tfidf" --description "" --output outputs/ --verbose 4


# Random Forest
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 1 --criterion "entropy" --n-estimators 25 --tfidf 'tfidf' --name "AP PAN17 sklearn RF char 1-1grams entro NE-25 no-tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 2 --criterion "entropy" --n-estimators 25 --tfidf 'tfidf' --name "AP PAN17 sklearn RF char 1-2grams entro NE-25 no-tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 3 --criterion "entropy" --n-estimators 25 --tfidf 'tfidf' --name "AP PAN17 sklearn RF char 1-3grams entro NE-25 no-tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 4 --criterion "entropy" --n-estimators 25 --tfidf 'tfidf' --name "AP PAN17 sklearn RF char 1-4grams entro NE-25 no-tfidf" --description "" --output outputs/ --verbose 4

python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 1 --criterion "entropy" --n-estimators 50 --tfidf 'tfidf' --name "AP PAN17 sklearn RF char 1-1grams entro NE-50 no-tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 2 --criterion "entropy" --n-estimators 50 --tfidf 'tfidf' --name "AP PAN17 sklearn RF char 1-2grams entro NE-50 no-tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 3 --criterion "entropy" --n-estimators 50 --tfidf 'tfidf' --name "AP PAN17 sklearn RF char 1-3grams entro NE-50 no-tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 4 --criterion "entropy" --n-estimators 50 --tfidf 'tfidf' --name "AP PAN17 sklearn RF char 1-4grams entro NE-50 no-tfidf" --description "" --output outputs/ --verbose 4

python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 1 --criterion "entropy" --n-estimators 100 --tfidf 'tfidf' --name "AP PAN17 sklearn RF char 1-1grams entro NE-100 no-tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 2 --criterion "entropy" --n-estimators 100 --tfidf 'tfidf' --name "AP PAN17 sklearn RF char 1-2grams entro NE-100 no-tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 3 --criterion "entropy" --n-estimators 100 --tfidf 'tfidf' --name "AP PAN17 sklearn RF char 1-3grams entro NE-100 no-tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 4 --criterion "entropy" --n-estimators 100 --tfidf 'tfidf' --name "AP PAN17 sklearn RF char 1-4grams entro NE-100 no-tfidf" --description "" --output outputs/ --verbose 4

python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 1 --criterion "entropy" --n-estimators 200 --tfidf 'tfidf' --name "AP PAN17 sklearn RF char 1-1grams entro NE-200 no-tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 2 --criterion "entropy" --n-estimators 200 --tfidf 'tfidf' --name "AP PAN17 sklearn RF char 1-2grams entro NE-200 no-tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 3 --criterion "entropy" --n-estimators 200 --tfidf 'tfidf' --name "AP PAN17 sklearn RF char 1-3grams entro NE-200 no-tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 4 --criterion "entropy" --n-estimators 200 --tfidf 'tfidf' --name "AP PAN17 sklearn RF char 1-4grams entro NE-200 no-tfidf" --description "" --output outputs/ --verbose 4


python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 1 --criterion "gini" --n-estimators 25 --tfidf 'tfidf' --name "AP PAN17 sklearn RF char 1-1grams gini NE-25 tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 2 --criterion "gini" --n-estimators 25 --tfidf 'tfidf' --name "AP PAN17 sklearn RF char 1-2grams gini NE-25 tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 3 --criterion "gini" --n-estimators 25 --tfidf 'tfidf' --name "AP PAN17 sklearn RF char 1-3grams gini NE-25 tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 4 --criterion "gini" --n-estimators 25 --tfidf 'tfidf' --name "AP PAN17 sklearn RF char 1-4grams gini NE-25 tfidf" --description "" --output outputs/ --verbose 4

python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 1 --criterion "gini" --n-estimators 50 --tfidf 'tfidf' --name "AP PAN17 sklearn RF char 1-1grams gini NE-50 tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 2 --criterion "gini" --n-estimators 50 --tfidf 'tfidf' --name "AP PAN17 sklearn RF char 1-2grams gini NE-50 tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 3 --criterion "gini" --n-estimators 50 --tfidf 'tfidf' --name "AP PAN17 sklearn RF char 1-3grams gini NE-50 tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 4 --criterion "gini" --n-estimators 50 --tfidf 'tfidf' --name "AP PAN17 sklearn RF char 1-4grams gini NE-50 tfidf" --description "" --output outputs/ --verbose 4

python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 1 --criterion "gini" --n-estimators 100 --tfidf 'tfidf' --name "AP PAN17 sklearn RF char 1-1grams gini NE-100 tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 2 --criterion "gini" --n-estimators 100 --tfidf 'tfidf' --name "AP PAN17 sklearn RF char 1-2grams gini NE-100 tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 3 --criterion "gini" --n-estimators 100 --tfidf 'tfidf' --name "AP PAN17 sklearn RF char 1-3grams gini NE-100 tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 4 --criterion "gini" --n-estimators 100 --tfidf 'tfidf' --name "AP PAN17 sklearn RF char 1-4grams gini NE-100 tfidf" --description "" --output outputs/ --verbose 4

python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 1 --criterion "gini" --n-estimators 200 --tfidf 'tfidf' --name "AP PAN17 sklearn RF char 1-1grams gini NE-200 tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 2 --criterion "gini" --n-estimators 200 --tfidf 'tfidf' --name "AP PAN17 sklearn RF char 1-2grams gini NE-200 tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 3 --criterion "gini" --n-estimators 200 --tfidf 'tfidf' --name "AP PAN17 sklearn RF char 1-3grams gini NE-200 tfidf" --description "" --output outputs/ --verbose 4
python3 sklearn_random_forest_baseline.py --dataset ~/Projets/TURING/Datasets/PAN17/AuthorProfiling/json/ --feature "char" --n-grams-min 1 --n-grams-max 4 --criterion "gini" --n-estimators 200 --tfidf 'tfidf' --name "AP PAN17 sklearn RF char 1-4grams gini NE-200 tfidf" --description "" --output outputs/ --verbose 4
