import os

# =============================================================================
# DERIVED PATHS - Automatically constructed
# =============================================================================

# Text dataset paths
TEXT_DATASETS_DIR = 'text_datasets/'

# Data statistics paths
DATA_STATS_PATH = {
    'clean_c4': os.path.join(TEXT_DATASETS_DIR, 'data_stats/clean_c4/'),
}

# Classification dataset paths
TOXICITY_DATA_PATH = '/path/to/toxicity/data'
SENTIMENT_DATA_PATH = '/path/to/sentiment/data'

# Specific data files
TOXICITY_TRAIN_FILE = os.path.join(TOXICITY_DATA_PATH, 'train.jsonl')
TOXICITY_VAL_FILE = os.path.join(TOXICITY_DATA_PATH, 'val.jsonl')
TOXICITY_TEST_FILE = os.path.join(TOXICITY_DATA_PATH, 'test.jsonl')

# Helper function for sentiment data files
def get_sentiment_file_path(split):
    """Get the path for a sentiment data file by split name"""
    return os.path.join(SENTIMENT_DATA_PATH, f'{split}_prompts.jsonl')

# Evaluation data paths
EVAL_DATA_DIR = os.path.join(TEXT_DATASETS_DIR, 'eval')
C4_PROMPT_CONTINUATIONS_PATH = os.path.join(EVAL_DATA_DIR, 'c4_prompt_continuations')
OPENWEBTEXT_PROMPT_CONTINUATIONS_PATH = os.path.join(EVAL_DATA_DIR, 'openwebtext_prompt_continuations')

# Tokenized data paths
TOKENIZED_DATA_DIR = os.path.join(TEXT_DATASETS_DIR, 'tokenized')
# Update as needed for different tokenization settings
CLEAN_C4_TOKENIZED_PATH = os.path.join(TOKENIZED_DATA_DIR, 'gpt2_tokenized_c4_tokensNone_block128_maxseq10000000_compression0.25')

# =============================================================================
# MODEL AND TOKENIZER SETTINGS
# =============================================================================
DEFAULT_TOKENIZER = 'gpt2-large'
MAX_LENGTH = 96
DEFAULT_NUM_DIFFUSION_TOKENS = 8