import torch
import torch.nn as nn

class MarketStressTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.input_projection = nn.Linear(in_features = 4, out_features = 64)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )
        self.output_layer = nn.Linear(
            in_features = 64, out_features= 1
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.output_layer(x)

        return x.squeeze(-1)



