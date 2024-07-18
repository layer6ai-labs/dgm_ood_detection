import torch
import torch.nn as nn

"""
Small MLP for toy experiments
"""

class SimpleMLP(nn.Module):
    """Simple MLP for flat data"""
    def __init__(self, in_dim, out_dim, hidden_sizes=(32, 32)):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_sizes = hidden_sizes

        layers = []
        prev_size = in_dim
        for size in self.hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.SiLU())
            prev_size = size

        layers.append(nn.Linear(prev_size, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SimpleDiffusionMLP(SimpleMLP):
    def __init__(self, data_dim, hidden_sizes=(32, 32)):
        super().__init__(in_dim=data_dim+1, out_dim=data_dim, hidden_sizes=hidden_sizes)
        self.sample_size = 1

    def forward(self, x, t):
        t = t.to(x.device)
        return self.net(torch.cat([x, t[...,None]], dim=-1))