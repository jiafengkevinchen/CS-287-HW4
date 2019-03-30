import torch
from torch import nn
from namedtensor import ntorch, NamedTensor
from namedtensor.nn import nn as nnn

class LayerNorm(nnn.Module):
    def __init__(self, features, dim, eps=1e-6):
        super().__init__()
        self.a_2 = ntorch.ones(features, names=(dim,))
        self.b_2 = ntorch.ones(features, names=(dim,))
        self.register_parameter("layernorma", self.a_2)
        self.register_parameter("layernormb", self.b_2)
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        mean = x.mean(self.dim)
        std = x.std(self.dim)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
