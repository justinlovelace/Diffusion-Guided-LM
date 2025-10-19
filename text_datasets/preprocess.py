import torch
import datasets

import os
from itertools import chain

import json
import argparse

from transformers import AutoTokenizer
from CONSTANTS import TOKENIZED_DATA_DIR

def exists(val):
    return val is not None



def load_c4(num_threads=0):
    """Load the C4 dataset."""
    # Load the dataset
    raw_data = datasets.load_dataset('c4', 'en.noblocklist', num_proc=num_threads if num_threads > 0 else None)
    return raw_data['train']


def preprocess_dataset(num_tokens, block_size, max_seq_in_raw_dataset, num_threads, compression_filter_t):
    """Preprocess C4 dataset for training."""
    # Load C4 dataset
    raw_data = load_c4(num_threads)
    raw_data = raw_data.shuffle(seed=89)  # Shuffle once here so that multiproc has shards of similar size!
    # This shuffle is crucial for fast multiprocessing tokenization
    # because datasets.map uses a contiguous sharding under the hood.

    # However, we also shuffle so we can now select a smaller range:
    print(f'Dataset size: {len(raw_data)}')
    if max_seq_in_raw_dataset < len(raw_data):
        raw_data = raw_data.select(range(int(max_seq_in_raw_dataset)))

    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    tokenized_dataset = _huggingface_preprocessing(raw_data, tokenizer, num_tokens, block_size, num_threads=num_threads, compression_filter_t=compression_filter_t)
    print('Tokenized dataset size:', len(tokenized_dataset))

    return tokenized_dataset


def _huggingface_preprocessing(raw_dataset, tokenizer, num_tokens, block_size, num_threads, compression_filter_t):
    """Dataset preprocessing and tokenization.

    This is basically the default HF routine from
    https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py
    """
    map_setup = dict(
        batched=True,
        batch_size=1024,
        num_proc=num_threads if num_threads > 0 else None,
        # load_from_cache_file=False,
        # keep_in_memory=False,
    )
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = getattr(raw_dataset, "column_names", "text")
    text_column_name = "text" if "text" in column_names else column_names[0]

    assert block_size < tokenizer.model_max_length, "Block size is too large for the selected tokenizer"
    model_max_length = tokenizer.model_max_length
    max_seq_length = block_size
    
    # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
    # The Collator is modified not to read special_masks anyway:
    if num_threads > 0:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def tokenize_function(examples):
        text_list = examples[text_column_name]
        tokenized_examples = tokenizer(
            text_list,
            return_special_tokens_mask=False,
            return_attention_mask=False,
        )
        if compression_filter_t == 0.0:
            return tokenized_examples
        filtered_input_ids = [tokenized_examples['input_ids'][idx] for idx in range(len(tokenized_examples['input_ids'])) if len(tokenized_examples['input_ids'][idx]) < compression_filter_t * len(text_list[idx])]
        tokenized_examples['input_ids'] = filtered_input_ids
        return tokenized_examples
    
    tokenized_dataset = raw_dataset.map(
        tokenize_function, remove_columns=column_names, desc="Running tokenizer on every text in dataset", **map_setup
    )

    # Data processing function that splits the dataset in chunks of block_size.
    def split_entries(examples):
        result = {'input_ids': []}
        input_ids = examples['input_ids']
        for input_id in input_ids:
            len_seq = len(input_id)
            if len_seq > max_seq_length:
                # Split into chunks of max_seq_length and discard the last chunk if it is smaller than 1/2 max_seq_length
                num_chunks = len_seq // max_seq_length
                result['input_ids'].extend([input_id[i*max_seq_length:(i+1)*max_seq_length] for i in range(num_chunks)])
                if len_seq % max_seq_length > max_seq_length // 2:
                    result['input_ids'].append(input_id[num_chunks*max_seq_length:])
            elif len_seq > max_seq_length // 2:
                # If the sequence is longer than 1/2 max_seq_length, keep it
                result['input_ids'].append(input_id)

        return result

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of
    # max_seq_length.
    def group_texts(examples):
        
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)] for k, t in concatenated_examples.items()}
        return result

    tokenized_dataset = tokenized_dataset.map(split_entries, desc=f"Splitting texts in chunks of {max_seq_length}", **map_setup)

    # Shuffle dataset
    tokenized_dataset = tokenized_dataset.shuffle(seed=233)

    # However, we also shuffle so we can now select a smaller range:
    print(f'Dataset size: {len(tokenized_dataset)}')
    if exists(num_tokens):
        max_seq_in_tokenized_dataset = num_tokens // block_size
        if max_seq_in_tokenized_dataset < len(tokenized_dataset):
            print(f"Selecting {max_seq_in_tokenized_dataset} from {len(tokenized_dataset)}")
            tokenized_dataset = tokenized_dataset.select(range(int(max_seq_in_tokenized_dataset)))
    else:
        print(f"Selecting all {len(tokenized_dataset)}")
    # Finally flatten
    # This is necessary for the save_to_disk call that comes next. If skipped here, the call will be invoked from save_to_disk
    # This way, atleast it shares the same batch parameters and prints a progress bar.
    tokenized_dataset = tokenized_dataset.map(desc="Flattening the indices", **map_setup)
    return tokenized_dataset

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_threads", type=int, default=0)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--num_tokens", type=int, default=None)
    parser.add_argument("--max_seq_in_raw_dataset", type=int, default=1e8)
    parser.add_argument("--compression_filter_t", type=float, default=0.0)
    args = parser.parse_args()

    tokenized_dataset = preprocess_dataset(num_tokens=args.num_tokens, 
                                                    block_size=args.block_size, 
                                                    max_seq_in_raw_dataset=args.max_seq_in_raw_dataset,
                                                    num_threads=args.num_threads,
                                                    compression_filter_t=args.compression_filter_t)
    
    # Save to disk - always C4 dataset
    output_path = os.path.join(TOKENIZED_DATA_DIR, f"gpt2_tokenized_c4_tokens{args.num_tokens}_block{args.block_size}_maxseq{args.max_seq_in_raw_dataset}_compression{args.compression_filter_t}")
    tokenized_dataset.save_to_disk(output_path)
