import torch
import datasets

import os
from itertools import chain

import json
import argparse

from transformers import AutoTokenizer, PreTrainedTokenizerBase
from CONSTANTS import C4_PROMPT_CONTINUATIONS_PATH, OPENWEBTEXT_PROMPT_CONTINUATIONS_PATH

def exists(val):
    return val is not None

def _concatenate_entries(dataset, num_entries_in_group, num_threads):
    if num_threads > 0:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def group_texts(examples):
        result = dict()
        for key, entries in examples.items():
            reduced_list = []
            state, num_collected = None, 0
            for entry in entries:
                num_collected += 1
                if num_collected == 1:
                    state = entry
                else:
                    state += entry
                if num_collected == num_entries_in_group:
                    reduced_list.append(state)
                    state, num_collected = None, 0

            result[key] = reduced_list

        return result

    map_setup = dict(
        batched=True,
        batch_size=1024,
        num_proc=num_threads if num_threads > 0 else None,
        # load_from_cache_file=False,
        # keep_in_memory=True,
    )
    dataset = dataset.map(group_texts, desc="Concatenating examples", **map_setup)
    return dataset


def load_openwebtext(num_threads=0):
    """Load the openwebtext dataset."""
    # Load the dataset
    raw_data = datasets.load_dataset('Skylion007/openwebtext', split='train', num_proc=num_threads if num_threads > 0 else None)
    return raw_data

def load_c4(num_threads=0):
    """Load the C4 dataset."""
    # Load the dataset
    raw_data = datasets.load_dataset('c4', 'en', num_proc=num_threads if num_threads > 0 else None)
    return raw_data['validation']

def preprocess_dataset(name, num_threads, num_samples):
    """A lot of loading and preprocessing."""
    # 1) Collect raw source datasets
    if name == "openwebtext":
        raw_data = load_openwebtext(num_threads)
    elif name == 'c4':
        raw_data = load_c4(num_threads)
    else:
        raise ValueError(f"Invalid dataset name {name}")
    raw_data = raw_data.shuffle(seed=89)  # Shuffle once here so that multiproc has shards of similar size!
    # This shuffle is crucial for fast multiprocessing tokenization
    # because datasets.map uses a contiguous sharding under the hood.

    # However, we also shuffle so we can now select a smaller range:
    print(f'Dataset size: {len(raw_data)}')
    raw_data = raw_data.select(range(int(20000)))
    print(f'Selecting 20k samples. Dataset size: {len(raw_data)}')

    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    tokenized_dataset = _huggingface_preprocessing(raw_data, tokenizer, num_threads=num_threads)  # Tokenize, group, sort...
    print('Tokenized dataset size:', len(tokenized_dataset))

    # Extract num_samples
    tokenized_dataset = tokenized_dataset.select(range(num_samples))

    # Truncate to 64 tokens
    tokenized_dataset = tokenized_dataset.map(lambda x: {'input_ids': x['input_ids'][:64]})

    # Split into prompt and continuation at 32 tokens
    tokenized_dataset = tokenized_dataset.map(lambda x: {'prompt': x['input_ids'][:32], 'continuation': x['input_ids'][32:]})

    # Detokenize input_ids as well
    tokenized_dataset = tokenized_dataset.map(lambda x: {'prompt_text': tokenizer.decode(x['prompt']), 'continuation_text': tokenizer.decode(x['continuation'])})

    # Save to disk using paths from CONSTANTS.py
    if name == 'c4':
        save_path = C4_PROMPT_CONTINUATIONS_PATH
    elif name == 'openwebtext':
        save_path = OPENWEBTEXT_PROMPT_CONTINUATIONS_PATH
    else:
        raise ValueError(f"Unknown dataset name: {name}")
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    tokenized_dataset.save_to_disk(save_path)
    


def _huggingface_preprocessing(raw_dataset, tokenizer, num_threads):
    """Dataset preprocessing and tokenization.

    This is basically the default HF routine from
    https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py
    """
    map_setup = dict(
        batched=True,
        batch_size=1024,
        num_proc=num_threads if num_threads > 0 else None,
    )
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = getattr(raw_dataset, "column_names", "text")
    text_column_name = "text" if "text" in column_names else column_names[0]
    
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
            truncation=True,
        )
        return tokenized_examples
    
    tokenized_dataset = raw_dataset.map(
        tokenize_function, remove_columns=column_names, desc="Running tokenizer on every text in dataset", **map_setup
    )

    # Filter examples with less than 64 tokens
    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x['input_ids']) >= 64)


    # Shuffle dataset
    tokenized_dataset = tokenized_dataset.shuffle(seed=233)

    # This is necessary for the save_to_disk call that comes next. If skipped here, the call will be invoked from save_to_disk
    # This way, atleast it shares the same batch parameters and prints a progress bar.
    tokenized_dataset = tokenized_dataset.map(desc="Flattening the indices", **map_setup)
    return tokenized_dataset

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_threads", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=5000)
    args = parser.parse_args()

    preprocess_dataset(name='openwebtext', num_threads=args.num_threads, num_samples=args.num_samples)

    preprocess_dataset(name='c4', num_threads=args.num_threads, num_samples=args.num_samples)
    
