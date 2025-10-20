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

# Classification dataset paths for logistic regression
CLS_DATASETS_PATH = 'classify/cls_datasets'

# Jigsaw dataset paths (used for toxicity classifier training)
JIGSAW_DATA_PATH = os.path.join(CLS_DATASETS_PATH, 'jigsaw')
JIGSAW_TRAIN_FILE = os.path.join(JIGSAW_DATA_PATH, 'train.pt')
JIGSAW_TEST_PUBLIC_FILE = os.path.join(JIGSAW_DATA_PATH, 'test_public_leaderboard.pt')
JIGSAW_TEST_PRIVATE_FILE = os.path.join(JIGSAW_DATA_PATH, 'test_private_leaderboard.pt')

# Sentiment classifier dataset paths
SST2_CLS_DATA_PATH = os.path.join(CLS_DATASETS_PATH, 'sentiment/sst2')
SST2_CLS_TRAIN_FILE = os.path.join(SST2_CLS_DATA_PATH, 'train.pt')
SST2_CLS_VAL_FILE = os.path.join(SST2_CLS_DATA_PATH, 'validation.pt')

AMAZON_CLS_DATA_PATH = os.path.join(CLS_DATASETS_PATH, 'sentiment/amazon_polarity')
AMAZON_CLS_TRAIN_FILE = os.path.join(AMAZON_CLS_DATA_PATH, 'train.pt')
AMAZON_CLS_TEST_FILE = os.path.join(AMAZON_CLS_DATA_PATH, 'test.pt')

AG_NEWS_CLS_DATA_PATH = os.path.join(CLS_DATASETS_PATH, 'sentiment/ag_news')
AG_NEWS_CLS_TRAIN_FILE = os.path.join(AG_NEWS_CLS_DATA_PATH, 'train.pt')
AG_NEWS_CLS_TEST_FILE = os.path.join(AG_NEWS_CLS_DATA_PATH, 'test.pt')

# Saved model paths
SAVED_MODELS_PATH = 'saved_models'
JIGSAW_LOG_REG_PATH = os.path.join(SAVED_MODELS_PATH, 'jigsaw/log_reg/')
SST_AMAZON_LOG_REG_PATH = os.path.join(SAVED_MODELS_PATH, 'sst_amazon/log_reg/')

# Evaluation data paths
EVAL_DATA_DIR = os.path.join(TEXT_DATASETS_DIR, 'data/eval')
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