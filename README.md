# Diffusion Guided Language Modeling

This is the official code release for the ACL Findings 2024 paper:

**Diffusion Guided Language Modeling**.

by Justin Lovelace, Varsha Kishore, Yiwei Chen, and Kilian Q. Weinberger

**Paper**: https://arxiv.org/abs/2408.04220

### Abstract
Current language models demonstrate remarkable proficiency in text generation. However, for many applications it is desirable to control attributes, such as sentiment, or toxicity, of the generated language -- ideally tailored towards each specific use case and target audience. For auto-regressive language models, existing guidance methods are prone to decoding errors that cascade during generation and degrade performance. In contrast, text diffusion models can easily be guided with, for example, a simple linear sentiment classifier -- however they do suffer from significantly higher perplexity than auto-regressive alternatives. In this paper we use a guided diffusion model to produce a latent proposal that steers an auto-regressive language model to generate text with desired properties. Our model inherits the unmatched fluency of the auto-regressive approach and the plug-and-play flexibility of diffusion. We show that it outperforms previous plug-and-play guidance methods across a wide range of benchmark data sets. Further, controlling a new attribute in our framework is reduced to training a single logistic regression classifier.


## Training

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Preprocess the dataset. The default configuration uses the C4 dataset:
```bash
cd text_datasets
bash scripts/c4_10mill.sh
```

This will download and preprocess up to 10 million sequences from C4 with compression filtering.

### Training the Diffusion Model

The diffusion model learns to generate latent proposals that guide text generation. Train it using:

```bash
bash scripts/train/diffusion/default.sh
```

Or run directly with custom parameters:
```bash
python train_diff.py \
  wandb_name=diff_c4 \
  train_mode=diffusion \
  dataset_name=clean_c4 \
  diffusion.train.num_train_steps=100000
```

Key configuration options:
- `wandb_name`: Experiment name for W&B logging
- `train_mode`: Set to `diffusion` for diffusion model training
- `dataset_name`: Dataset to use (e.g., `clean_c4`)
- `diffusion.train.num_train_steps`: Number of training steps

The diffusion model will be saved to `saved_models/` by default.

### Training the Prompt Model

The prompt model is the autoregressive language model that will be guided by the diffusion model. Train it using:

```bash
bash scripts/train/prompt/default.sh
```

Or run directly:
```bash
python train_diff.py \
  wandb_name=prompt_c4 \
  train_mode=prompt \
  prompt.train.num_train_steps=100000
```

### Configuration System

This project uses Hydra for configuration management. Configurations are organized in `configs/`:

- **`configs/config.yaml`**: Main configuration file
- **`configs/diffusion/`**: Diffusion model settings (architecture, loss, sampling, training)
- **`configs/prompt/`**: Prompt model settings (architecture, augmentation, training)
- **`configs/eval/`**: Evaluation parameters

### Monitoring

Training progress is logged to Weights & Biases (W&B). Make sure you have W&B configured:
```bash
wandb login
```


### Training Attribute Classifiers

The framework uses simple logistic regression classifiers for plug-and-play attribute guidance. Train classifiers for sentiment and toxicity control:

#### 1. Cache Dataset Embeddings

First, cache the sentence embeddings for the classifier training datasets:

```bash
# Cache sentiment datasets (SST2 + Amazon Polarity)
bash scripts/log_reg/cache_sentiment.sh

# Cache Jigsaw toxicity dataset
bash scripts/log_reg/cache_jigsaw.sh
```

**Note**: The Jigsaw dataset requires manual download from Kaggle:
https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data

Download the data, extract all files to a folder, then update the `data_dir` parameter in `classify/cls_datasets/cache_jigsaw_datasets.py`.

#### 2. Train Classifiers

Once embeddings are cached, train the logistic regression classifiers:

```bash
# Train sentiment classifier (on SST2 + Amazon Polarity)
bash scripts/log_reg/train_sentiment.sh

# Train toxicity classifier (on Jigsaw)
bash scripts/log_reg/train_jigsaw.sh
```

The trained classifiers will be saved to `saved_models/sst_amazon/log_reg/` and `saved_models/jigsaw/log_reg/` respectively.



### Citation
```bibtex
@inproceedings{lovelace2024dglm,
  title={Diffusion Guided Language Modeling},
  author={Lovelace, Justin and Kishore, Varsha and Chen, Yiwei and Weinberger, Kilian Q},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2024},
  year={2024},
  publisher={Association for Computational Linguistics},
}
