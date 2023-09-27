import torch

from layers import FMLinear


class LRModel(torch.nn.Module):

    def __init__(self, field_dims):
        super().__init__()
        self.linear = FMLinear(field_dims)

    def forward(self, x):
        return torch.sigmoid(self.linear(x).squeeze(1))
