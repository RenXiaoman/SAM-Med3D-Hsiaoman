import torch
import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connection=True):
        super().__init__()
        self.skip_connection = skip_connection
        self.D_hidden_features = int(D_features * mlp_ratio)
        self.D_fc1 = nn.Linear(D_features, self.D_hidden_features)
        self.D_fc2 = nn.Linear(self.D_hidden_features, D_features)
        self.act = act_layer()

    def forward(self, x):
        # x is [B, HW+1, D]
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        return (x + xs) if self.skip_connection else xs
