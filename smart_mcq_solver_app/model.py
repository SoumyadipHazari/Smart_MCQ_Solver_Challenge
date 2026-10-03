import torch
import torch.nn as nn

class Attention(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.attention = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_output):

        attention_scores = self.attention(lstm_output)

        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )

        context_vector = torch.sum(
            attention_weights * lstm_output,
            dim=1
        )

        return context_vector


class BiLSTMClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        dropout=0.3
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.attention = Attention(hidden_dim)

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            hidden_dim * 2,
            1
        )

    def forward(self, x):

        x = self.embedding(x)

        lstm_output, _ = self.bilstm(x)

        context = self.attention(lstm_output)

        context = self.dropout(context)

        logits = self.classifier(context)

        return logits.squeeze(1)