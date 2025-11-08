from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import AbstractNNModel, NNModelArgs, NNModel


class NNModel_LSTM(NNModel):
    """
    LSTM model for text classification
    Expects input from tokenizer: batch with 'input_ids' tensor of shape (batch_size, seq_len)
    """

    def __init__(self):
        super().__init__()

    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)

        vocab_size = int(getattr(args, "vocab_size", 30522))  # Default BERT vocab size
        embedding_dim = int(getattr(args, "embedding_dim", 128))
        hidden_dim = int(getattr(args, "hidden_dim", 256))
        num_layers = int(getattr(args, "num_layers", 2))
        num_classes = int(getattr(args, "num_classes", 2))
        dropout = float(getattr(args, "dropout", 0.3))
        bidirectional = bool(getattr(args, "bidirectional", True))

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, num_classes)

        return self

    def forward(self, batch):
        # Handle both dict (from tokenizer) and tensor inputs
        if isinstance(batch, dict):
            input_ids = batch.get("input_ids", batch.get("input_ids"))
        else:
            input_ids = batch

        # Embedding
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last hidden state
        if self.lstm.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        
        # Dropout and classification
        output = self.dropout(hidden)
        output = self.fc(output)
        
        return output

