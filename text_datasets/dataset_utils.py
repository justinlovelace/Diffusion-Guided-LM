import datasets
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import os
import torch
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from dataclasses import dataclass
from einops import repeat
from .CONSTANTS import CLEAN_C4_TOKENIZED_PATH, C4_PROMPT_CONTINUATIONS_PATH, OPENWEBTEXT_PROMPT_CONTINUATIONS_PATH, TOXICITY_TRAIN_FILE, TOXICITY_VAL_FILE, TOXICITY_TEST_FILE, get_sentiment_file_path
 
MAX_LENGTH = 96


@dataclass
class DataCollatorWithDiffusionTokens:
    """
    Data collator used for language modeling with diffusion prompts.
    - Input: A batch of examples with the following fields:
        - input_ids: Indices of input sequence tokens in the vocabulary.
        - num_tokens: Number of tokens in the input sequence.
    - Output: A dictionary with the following fields:
        - input_ids: Indices of input sequence tokens in the vocabulary.
        - labels: Indices of output sequence tokens in the vocabulary.
        - diffusion_token_mask: Mask of diffusion tokens.
        - continuation_start: Index of the first token in the continuation.
        - prompt: Text of the prompt.
        - continuation: Text of the continuation.
        - clean_input_ids: Indices of input sequence tokens in the vocabulary without diffusion tokens.
        - clean_labels: Indices of output sequence tokens in the vocabulary without diffusion tokens.
    """

    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = MAX_LENGTH
    return_tensors: str = "pt"
    num_diffusion_tokens: int = 8
    validation: bool = False,

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if 'num_prompt_tokens' in features[0]:
            # Use specified prompt length
            continuation_start = torch.tensor([feature['num_prompt_tokens'] for feature in features], dtype=torch.long)
        else:
            # Sample a continuation start
            continuation_start = torch.rand(size=(len(features),))
            # Discretize continuation_start into 0, 1, ..., num_tokens - 1
            num_tokens = torch.stack([feature['num_tokens'] for feature in features])
            continuation_start = torch.round(continuation_start * (num_tokens-1)).long()

        prompts = [features[idx]['input_ids'][:continuation_start[idx].item()] for idx in range(len(features))]
        continuations = [features[idx]['input_ids'][continuation_start[idx].item():] for idx in range(len(features))]

        diffusion_token_prompt = torch.tensor([self.tokenizer.pad_token_id]*self.num_diffusion_tokens, dtype=prompts[0].dtype, device=prompts[0].device)
        input_ids = [torch.cat((prompts[idx],diffusion_token_prompt,continuations[idx]), dim=0) for idx in range(len(features))]
        
        batch = self.tokenizer.pad(
            {'input_ids': input_ids},
            padding='longest',
            max_length=self.max_length,
            return_tensors=self.return_tensors,
            return_attention_mask=True,
        )

        batch_size, seq_len = batch['input_ids'].shape

        seq_ids = torch.arange(seq_len).repeat((batch_size,1))
        batch['diffusion_token_mask'] = torch.logical_and((seq_ids >= continuation_start[:, None]), (seq_ids < continuation_start[:, None] + self.num_diffusion_tokens))

        # Map labels to IDs
        batch["labels"] = batch["input_ids"].clone()
        
        # Replace prompt and diffusion tokens with -100
        batch['labels'][batch['labels'] == self.tokenizer.pad_token_id] = -100

        batch['continuation_start'] = continuation_start

        batch['prompt'] = [self.tokenizer.decode(prompt, skip_special_tokens=True) for prompt in prompts]
        batch['continuation'] = [self.tokenizer.decode(continuation, skip_special_tokens=True) for continuation in continuations]
        
        if self.validation:
            clean_batch = self.tokenizer.pad(
                {'input_ids': [feature['input_ids'] for feature in features]},
                padding='longest',
                max_length=self.max_length,
                return_tensors=self.return_tensors,
                return_attention_mask=True,
                )
            batch['clean_input_ids'] = clean_batch['input_ids']
            batch['clean_labels'] = clean_batch['input_ids'].clone()
            # Replace pad tokens with -100
            batch['clean_labels'][batch['clean_labels'] == self.tokenizer.pad_token_id] = -100


        return batch

def get_real_toxicity_prompts_dataset_eval_split(tokenizer, num_diffusion_tokens, validation=True):
    dataset = datasets.load_dataset('json', data_files={'train': TOXICITY_TRAIN_FILE, 
                                                        'val': TOXICITY_VAL_FILE, 
                                                        'test': TOXICITY_TEST_FILE})
    def preprocess_function(examples):
        combined_text = [p['text']+c['text'] for p, c in zip(examples['prompt'], examples['continuation'])]

        # Tokenize the input
        tokenized_inputs = tokenizer(combined_text, max_length=MAX_LENGTH-num_diffusion_tokens, truncation=True, padding='max_length')
        num_tokens = [len(input_ids) for input_ids in tokenized_inputs['input_ids']]
        
        tokenized_inputs['num_tokens'] = num_tokens


        prompt_input_ids = tokenizer([p['text'] for p in examples['prompt']])['input_ids']
        num_prompt_tokens = [len(input_ids) for input_ids in prompt_input_ids]
        tokenized_inputs['num_prompt_tokens'] = num_prompt_tokens

        return tokenized_inputs
    
    dataset = dataset.map(preprocess_function, batched=True, remove_columns=['filename', 'begin', 'end', 'challenging'])
    dataset = dataset.with_format('pt')
    return dataset

def get_openwebtext_dataset_eval_split(tokenizer, num_diffusion_tokens, split='neutral'):
    data_file = get_sentiment_file_path(split)
    dataset = datasets.load_dataset('json', data_files={'test': data_file})
    def preprocess_function(examples):
        combined_text = [p['text']+c['text'] for p, c in zip(examples['prompt'], examples['continuation'])]

        # Tokenize the input
        tokenized_inputs = tokenizer(combined_text, max_length=MAX_LENGTH-num_diffusion_tokens, truncation=True, padding='max_length')
        num_tokens = [len(input_ids) for input_ids in tokenized_inputs['input_ids']]
        
        tokenized_inputs['num_tokens'] = num_tokens


        prompt_input_ids = tokenizer([p['text'] for p in examples['prompt']])['input_ids']
        num_prompt_tokens = [len(input_ids) for input_ids in prompt_input_ids]
        tokenized_inputs['num_prompt_tokens'] = num_prompt_tokens

        return tokenized_inputs
    dataset = dataset.map(preprocess_function, batched=True, remove_columns=['md5_hash', 'num_positive'])
    dataset = dataset.with_format('pt')
    return dataset

def get_c4_eval_split(tokenizer, num_diffusion_tokens, ):
    data_path = C4_PROMPT_CONTINUATIONS_PATH
    dataset = datasets.load_from_disk(data_path)
    dataset = datasets.DatasetDict({'test': dataset})
    
    def preprocess_function(examples):

        # Tokenize the input
        input_ids = [p+c for p,c in zip(examples['prompt'], examples['continuation'])]
        tokenized_inputs = tokenizer.pad(
            {'input_ids': input_ids},
            padding='max_length',
            max_length=MAX_LENGTH-num_diffusion_tokens,
        )
        num_tokens = [len(input_ids) for input_ids in tokenized_inputs['input_ids']]
        
        tokenized_inputs['num_tokens'] = num_tokens

        prompt_input_ids = examples['prompt']
        num_prompt_tokens = [len(input_ids) for input_ids in prompt_input_ids]
        tokenized_inputs['num_prompt_tokens'] = num_prompt_tokens

        return tokenized_inputs
    dataset = dataset.map(preprocess_function, batched=True,)
    dataset = dataset.with_format('pt')
    return dataset

def get_openwebtext_eval_split(tokenizer, num_diffusion_tokens, ):
    data_path = OPENWEBTEXT_PROMPT_CONTINUATIONS_PATH
    dataset = datasets.load_from_disk(data_path)
    dataset = datasets.DatasetDict({'test': dataset})
    
    def preprocess_function(examples):

        # Tokenize the input
        input_ids = [p+c for p,c in zip(examples['prompt'], examples['continuation'])]
        tokenized_inputs = tokenizer.pad(
            {'input_ids': input_ids},
            padding='max_length',
            max_length=MAX_LENGTH-num_diffusion_tokens,
        )
        num_tokens = [len(input_ids) for input_ids in tokenized_inputs['input_ids']]
        
        tokenized_inputs['num_tokens'] = num_tokens

        prompt_input_ids = examples['prompt']
        num_prompt_tokens = [len(input_ids) for input_ids in prompt_input_ids]
        tokenized_inputs['num_prompt_tokens'] = num_prompt_tokens

        return tokenized_inputs
    dataset = dataset.map(preprocess_function, batched=True,)
    dataset = dataset.with_format('pt')
    return dataset


def get_clean_c4_dataset(tokenizer, num_diffusion_tokens):
    path = CLEAN_C4_TOKENIZED_PATH
    # Assert directory exists
    assert os.path.isdir(path), f"Dataset directory not found: {path}"
    dataset = datasets.load_from_disk(path)
    def preprocess_function(examples):
        tokenized_inputs = {}
        tokenized_inputs['input_ids'] = [input_ids[:MAX_LENGTH-num_diffusion_tokens] for input_ids in examples['input_ids']]

        num_tokens = [len(input_ids) for input_ids in tokenized_inputs['input_ids']]
        
        tokenized_inputs['num_tokens'] = num_tokens

        return tokenized_inputs
    
    dataset = dataset.map(preprocess_function, batched=True,)
    dataset = dataset.with_format('pt')
    return dataset



def get_dataset(dataset_name, tokenizer, num_diffusion_tokens):
    if dataset_name == 'real-toxicity-prompts-eval-split':
        dataset = get_real_toxicity_prompts_dataset_eval_split(tokenizer, num_diffusion_tokens)
        # Split train into train/validation/test
    elif 'openwebtext' in dataset_name and 'eval_split' not in dataset_name:
        split = dataset_name.split('_')[-1]
        dataset = get_openwebtext_dataset_eval_split(tokenizer, num_diffusion_tokens, split=split)
    elif dataset_name == 'clean_c4':
        dataset = get_clean_c4_dataset(tokenizer, num_diffusion_tokens)
        # Split train into train/validation/test
        dataset = dataset.train_test_split(test_size=0.01, seed=42)
        val_test_ds = dataset['test'].train_test_split(test_size=0.5, seed=42)
        dataset['val'] = val_test_ds['train']
        dataset['test'] = val_test_ds['test']
    elif dataset_name == 'c4_eval_split':
        dataset = get_c4_eval_split(tokenizer, num_diffusion_tokens)
    elif dataset_name == 'openwebtext_eval_split':
        dataset = get_openwebtext_eval_split(tokenizer, num_diffusion_tokens)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")

    return dataset

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    # Add padding token to vocabulary
    tokenizer.pad_token = tokenizer.eos_token
    # dataset = get_dataset('allenai/real-toxicity-prompts', tokenizer, 4)
    dataset = get_real_toxicity_prompts_dataset_eval_split(tokenizer, 4)
