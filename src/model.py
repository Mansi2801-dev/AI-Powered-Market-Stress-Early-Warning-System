import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """Injects sinusoidal positional embeddings so the Transformer understands sequence time."""
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # persistent=False keeps it compatible with existing weights checkpoints
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

class MarketStressTransformer(nn.Module):
    def __init__(
        self,
        n_features: int = 4,
        d_model: int = 64,
        nhead: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_projection = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (Batch, Sequence=60, Features=4)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        # Pool the final timestep token (t=60)
        out = self.output_layer(x[:, -1])
        return out.squeeze(-1)

if __name__ == "__main__":
    model = MarketStressTransformer()
    dummy = torch.randn(8, 60, 4)
    pred = model(dummy)
    print(f"[+] Model instantiated successfully.")
    print(f"[+] Forward pass shape: {pred.shape} (Expected: torch.Size([8]))")