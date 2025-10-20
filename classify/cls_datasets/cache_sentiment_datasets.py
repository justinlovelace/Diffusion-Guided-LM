import argparse
import torch
import os
from datasets import load_dataset

from sentence_transformers import SentenceTransformer
from text_datasets.CONSTANTS import SST2_CLS_DATA_PATH, AMAZON_CLS_DATA_PATH, AG_NEWS_CLS_DATA_PATH


def main(args):
    print('Loading sentence encoder')
    sentence_encoder = SentenceTransformer('sentence-transformers/sentence-t5-xl').half().cuda()

    # Map dataset names to their paths
    dataset_paths = {
        'sst2': SST2_CLS_DATA_PATH,
        'amazon_polarity': AMAZON_CLS_DATA_PATH,
        'ag_news': AG_NEWS_CLS_DATA_PATH
    }

    if args.dataset_name not in dataset_paths:
        raise ValueError(f"Dataset {args.dataset_name} not supported")

    save_dir = dataset_paths[args.dataset_name]
    os.makedirs(save_dir, exist_ok=True)

    dataset = load_dataset(args.dataset_name)
    if args.dataset_name == "sst2":
        splits = ['train', 'validation']
        text_col = 'sentence'
        save_label = 'sentiment'
    elif args.dataset_name == "amazon_polarity":
        splits = ['train', 'test']
        text_col = 'content'
        save_label = 'sentiment'
    elif args.dataset_name == "ag_news":
        splits = ['train', 'test']
        text_col = 'text'
        save_label = 'class'

    for split in splits:
        split_dataset = dataset[split]
        sentences = split_dataset[text_col]
        sentiment = split_dataset['label']

        print(f'Encoding {split}')
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            embeddings = sentence_encoder.encode(sentences, convert_to_tensor=True, show_progress_bar=True)

        # Save embedding, sentiment pairs
        data = {'embeddings': embeddings, save_label: sentiment}
        output_path = os.path.join(save_dir, f"{split}.pt")
        torch.save(data, output_path)
        print(f'Saved to {output_path}')
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache sentiment dataset embeddings")
    parser.add_argument("--dataset_name", type=str, default='amazon_polarity',
                        choices=['sst2', 'amazon_polarity', 'ag_news'],
                        help="Dataset to cache")

    args = parser.parse_args()

    main(args)
