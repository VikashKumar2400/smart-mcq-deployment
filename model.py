import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=256,
        hidden_dim=256,
        num_layers=2,
        num_classes=5,
        dropout=0.3
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_dim*2, num_classes)

    def forward(self, input_ids):

        x = self.embedding(input_ids)

        output, (hidden, cell) = self.lstm(x)

        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)

        hidden = self.dropout(hidden)

        logits = self.fc(hidden)

        return logits