import torch
import torch.nn as nn

class MarketStressTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        d_model = 64
        n_layers = 2
        n_features = 4

        self.input_projection = nn.Linear(n_features, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=256,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer(x)
        x = self.output_layer(x[:, -1])
        return x.squeeze(-1)