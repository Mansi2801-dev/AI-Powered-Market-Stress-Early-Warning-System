import torch
import torch.nn as nn

class MarketStressTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        # input projection
        self.input_projection = nn.Linear(4, 128)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=3
        )

        # better pooling: learn importance instead of mean
        self.attention_pool = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.output_layer = nn.Linear(128, 1)

    def forward(self, x):
        x = self.input_projection(x)          # (B, T, 128)
        x = self.transformer(x)               # (B, T, 128)

        # attention pooling (IMPORTANT FIX)
        weights = self.attention_pool(x)      # (B, T, 1)
        weights = torch.softmax(weights, dim=1)

        x = torch.sum(x * weights, dim=1)     # (B, 128)

        x = self.output_layer(x)              # (B, 1)

        return x.squeeze(-1)