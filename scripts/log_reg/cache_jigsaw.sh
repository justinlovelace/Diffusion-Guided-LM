#!/bin/bash
# Cache Jigsaw dataset embeddings for toxicity classifier training
#
# NOTE: Jigsaw dataset requires manual download from Kaggle:
# https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data
#
# Download using Kaggle CLI (https://www.kaggle.com/docs/api) or manually
# Extract all files to a single folder, then update the data_dir path in
# classify/cls_datasets/cache_jigsaw_datasets.py to point to your extracted data

python -m classify.cls_datasets.cache_jigsaw_datasets
