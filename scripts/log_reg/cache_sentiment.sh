#!/bin/bash
# Cache sentiment dataset embeddings for sentiment classifier training

# Cache SST2 dataset
echo "Caching SST2 dataset..."
python -m classify.cls_datasets.cache_sentiment_datasets --dataset_name sst2

# Cache Amazon Polarity dataset
echo "Caching Amazon Polarity dataset..."
python -m classify.cls_datasets.cache_sentiment_datasets --dataset_name amazon_polarity

# Cache AG News dataset (optional)
# echo "Caching AG News dataset..."
# python -m classify.cls_datasets.cache_sentiment_datasets --dataset_name ag_news
