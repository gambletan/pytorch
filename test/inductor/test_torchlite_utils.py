"""Shared test model classes for torchlite tests.

These model classes are used across multiple test files (test_torchlite_passes,
test_torchlite_training, test_torchlite_perf) and are centralized here to
avoid duplication.
"""

import torch
import torch.nn as nn


class TrainStep(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, target):
        out = self.model(x)
        return ((out - target) ** 2).mean()


class SimpleLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_dim, out_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        return x @ self.weight + self.bias


class TwoLayerMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(in_dim, hidden_dim) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.w2 = nn.Parameter(torch.randn(hidden_dim, out_dim) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        h = torch.relu(x @ self.w1 + self.b1)
        return h @ self.w2 + self.b2


class ThreeLayerSinCos(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(in_dim, hidden_dim) * 0.01)
        self.w2 = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.w3 = nn.Parameter(torch.randn(hidden_dim, out_dim) * 0.01)

    def forward(self, x):
        h = torch.sin(x @ self.w1)
        h = torch.cos(h @ self.w2)
        return h @ self.w3
