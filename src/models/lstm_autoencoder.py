import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.decoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.output_layer = nn.Linear(
            hidden_dim,
            vocab_size
        )

    def forward(self, x):

        embedded = self.embedding(x)

        _, (hidden, cell) = self.encoder(
            embedded
        )

        decoded, _ = self.decoder(
            embedded,
            (hidden, cell)
        )

        output = self.output_layer(
            decoded
        )

        return output
