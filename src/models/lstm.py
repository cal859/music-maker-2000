import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        seq_size: int,
        embedding_size: int,
        lstm_size: int,
        n_layers: int,
        dropout: float,
    ):
        super(LSTMModel, self).__init__()
        self.seq_size = seq_size
        self.lstm_size = lstm_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(n_vocab, embedding_size)
        self.lstm = nn.LSTM(
            embedding_size,
            lstm_size,
            num_layers=self.n_layers,
            dropout=self.dropout,
            batch_first=True,
        )
        self.dense = nn.Linear(lstm_size, n_vocab)
        self.model_type = "lstm"

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.dense(output)

        return logits, state

    def zero_state(self, batch_size):
        """LSTM have hidden and memory states, so need to set both to zero"""
        return (
            torch.zeros(self.n_layers, batch_size, self.lstm_size),
            torch.zeros(self.n_layers, batch_size, self.lstm_size),
        )
