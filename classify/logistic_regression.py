import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import os
import json

from text_datasets.CONSTANTS import (
    SST2_CLS_TRAIN_FILE, SST2_CLS_VAL_FILE,
    AMAZON_CLS_TRAIN_FILE, AMAZON_CLS_TEST_FILE,
    SST_AMAZON_LOG_REG_PATH
)

task = 'sentiment'
# Load the training and validation data
train_data_sst2 = torch.load(SST2_CLS_TRAIN_FILE)
val_data_sst2 = torch.load(SST2_CLS_VAL_FILE)

train_data_amazon = torch.load(AMAZON_CLS_TRAIN_FILE)
val_data_amazon = torch.load(AMAZON_CLS_TEST_FILE)

train_data = {}
val_data = {}
train_data['embeddings'] = torch.cat([train_data_sst2['embeddings'], train_data_amazon['embeddings']])
val_data['embeddings'] = torch.cat([val_data_sst2['embeddings'], val_data_amazon['embeddings']])
train_data['sentiment'] = np.concatenate(([train_data_sst2['sentiment'], train_data_amazon['sentiment']]))
val_data['sentiment'] = np.concatenate(([val_data_sst2['sentiment'], val_data_amazon['sentiment']]))
# shuffle train data using torch
torch.manual_seed(42)
idx = torch.randperm(train_data['embeddings'].shape[0])
train_data['embeddings'] = train_data['embeddings'][idx]
train_data['sentiment'] = train_data['sentiment'][idx]

train_labels = np.array(train_data['sentiment'])
val_labels = np.array(val_data['sentiment'])

save_path = SST_AMAZON_LOG_REG_PATH


train_embeddings = train_data['embeddings'].cpu().numpy()
val_embeddings = val_data['embeddings'].cpu().numpy()

# Create model with best regularization
l2_reg = 1e-3
if l2_reg == 0:
    model = LogisticRegression(random_state=42, max_iter=10000, penalty=None)
else:
    model = LogisticRegression(C=1/l2_reg, random_state=42, max_iter=10000, verbose=1)
print("Training model...")
model.fit(train_embeddings, train_labels)

val_preds = model.predict(val_embeddings)

acc = accuracy_score(val_labels, val_preds)
f1 = f1_score(val_labels, val_preds)  
auc = roc_auc_score(val_labels, val_preds)
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {auc:.4f}")
# Save metrics to json file
metrics = {'accuracy': acc, 'f1': f1, 'auc': auc}
# Create directory if it doesn't exist
if not os.path.isdir(save_path):
    os.makedirs(save_path)

with open(save_path + 'metrics.json', 'w') as f:
    json.dump(metrics, f)

# Save logistic regression weights
lr_model = {'coef': model.coef_, 'intercept': model.intercept_}
torch.save(lr_model, save_path + 'lr_model.pt')