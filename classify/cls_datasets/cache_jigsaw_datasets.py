import jsonlines
import torch
import datasets
import os

from sentence_transformers import SentenceTransformer
from text_datasets.CONSTANTS import JIGSAW_DATA_PATH


def main():
    sentence_encoder = SentenceTransformer('sentence-transformers/sentence-t5-xl').half().cuda()
    jigsaw_dataset = datasets.load_dataset('google/jigsaw_unintended_bias', data_dir='jigsaw')
    splits = ['train', 'test_public_leaderboard', 'test_private_leaderboard']

    # Create output directory if it doesn't exist
    os.makedirs(JIGSAW_DATA_PATH, exist_ok=True)

    for split in splits:
        sentences = jigsaw_dataset[split]['comment_text']
        toxicities = jigsaw_dataset[split]['target']

        assert len(sentences) == len(toxicities)
        # Print split stats
        print(f'Num sentences: {len(sentences)}')

        print(f'Encoding {split}')
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            embeddings = sentence_encoder.encode(sentences, convert_to_tensor=True, show_progress_bar=True, device='cuda', batch_size=128)

        # Save embedding, toxicity pairs
        data = {'embeddings': embeddings, 'toxicities': toxicities}
        output_path = os.path.join(JIGSAW_DATA_PATH, f"{split}.pt")
        torch.save(data, output_path)
        print(f'Saved to {output_path}')
            

if __name__ == "__main__":
    main()
