from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import AbstractNNModel, NNModelArgs, NNModel


class NNModel_TextCNN(NNModel):
    """
    TextCNN model for text classification
    Expects input from tokenizer: batch with 'input_ids' tensor of shape (batch_size, seq_len)
    """

    def __init__(self):
        super().__init__()

    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)

        vocab_size = int(getattr(args, "vocab_size", 30522))  # Default BERT vocab size
        embedding_dim = int(getattr(args, "embedding_dim", 128))
        num_classes = int(getattr(args, "num_classes", 2))
        num_filters = int(getattr(args, "num_filters", 100))
        filter_sizes = getattr(args, "filter_sizes", [3, 4, 5])
        dropout = float(getattr(args, "dropout", 0.5))

        if isinstance(filter_sizes, (list, tuple)):
            self.filter_sizes = filter_sizes
        else:
            self.filter_sizes = [filter_sizes]

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Convolutional layers for different filter sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
            for fs in self.filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(self.filter_sizes) * num_filters, num_classes)

        return self

    def forward(self, batch):
        # Handle both dict (from tokenizer) and tensor inputs
        if isinstance(batch, dict):
            input_ids = batch.get("input_ids", batch.get("input_ids"))
        else:
            input_ids = batch

        # Embedding: (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(input_ids)
        
        # Conv1d expects (batch_size, embedding_dim, seq_len)
        embedded = embedded.permute(0, 2, 1)
        
        # Apply convolutions and max pooling
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))  # (batch_size, num_filters, conv_seq_len)
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))  # (batch_size, num_filters, 1)
            conv_outputs.append(pooled.squeeze(2))  # (batch_size, num_filters)
        
        # Concatenate all filter outputs
        concatenated = torch.cat(conv_outputs, dim=1)  # (batch_size, len(filter_sizes) * num_filters)
        
        # Dropout and classification
        output = self.dropout(concatenated)
        output = self.fc(output)
        
        return output

