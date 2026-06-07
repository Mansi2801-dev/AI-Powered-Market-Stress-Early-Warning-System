import torch
import torch.nn as nn

class MarketStressTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_projection = nn.Linear(in_features = 4, out_features = 64)
