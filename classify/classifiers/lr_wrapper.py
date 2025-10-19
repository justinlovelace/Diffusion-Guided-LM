import torch
import torch.nn as nn
import torch.nn.functional as F

class LogisticRegression(nn.Module):
    def __init__(self, weights, bias):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(768, 1)
        self.linear.weight = nn.Parameter(weights)
        self.linear.bias = nn.Parameter(bias)
        self.loss = nn.BCEWithLogitsLoss(reduction='sum')

    def get_loss(self, data, labels, disable_reduction=False):
        # data is a tensor of embeddings of shape (batch_size, 768)
        # labels is a float tensor labels of shape (batch_size)
        # Returns the loss (a single value)
        logits = self.forward(data).squeeze(-1)
        if disable_reduction:
            loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
            return loss
        loss = self.loss(logits, labels)
        return loss
        
    def forward(self, x):
        # x is a batch of embeddings of shape (batch_size, 768)
        return self.linear(x)