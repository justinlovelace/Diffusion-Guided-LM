import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import os
import json

from text_datasets.CONSTANTS import (
    JIGSAW_TRAIN_FILE, JIGSAW_TEST_PUBLIC_FILE,
    JIGSAW_LOG_REG_PATH
)

# Load the training and validation data
train_data = torch.load(JIGSAW_TRAIN_FILE)
val_data = torch.load(JIGSAW_TEST_PUBLIC_FILE)

# Binarize toxicity scores at 0.5
train_embeddings = train_data['embeddings'].cpu().numpy()
val_embeddings = val_data['embeddings'].cpu().numpy()
train_labels = np.array(train_data['toxicities']) > 0.5
val_labels = np.array(val_data['toxicities']) > 0.5


# Create model with best regularization        
best_reg = 1e-3
if best_reg == 0:
    model = LogisticRegression(random_state=42, max_iter=10000, penalty=None, solver='lbfgs', verbose=1, class_weight='balanced')
else:
    model = LogisticRegression(C=1/best_reg, random_state=42, max_iter=10000, solver='lbfgs', verbose=1, class_weight='balanced')
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
os.makedirs(JIGSAW_LOG_REG_PATH, exist_ok=True)

with open(os.path.join(JIGSAW_LOG_REG_PATH, 'metrics.json'), 'w') as f:
    json.dump(metrics, f)

# Save logistic regression weights
lr_model = {'coef': model.coef_, 'intercept': model.intercept_}
torch.save(lr_model, os.path.join(JIGSAW_LOG_REG_PATH, 'lr_model.pt'))