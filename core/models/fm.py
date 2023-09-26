import torch
import torch.nn as nn
from layers import FMEmbedding, FactoMachine, FMLinear

class FMModel(nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = FMEmbedding(field_dims, embed_dim)
        self.linear = FMLinear(field_dims)
        self.fm = FactoMachine(reduce_sum=True)


    def forward(self, x):
        x = self.linear(x) + self.fm(self.embedding(x))
        return torch.sigmoid(x.squeeze(1))
    
class FMwModel(nn.Module):

    def __init__(self, fm_field_dims, embed_dim, linear_field_dims):
        super().__init__()
        self.embedding = FMEmbedding(fm_field_dims, embed_dim)
        self.linear = FMLinear(fm_field_dims)
        self.fm = FactoMachine(reduce_sum=True)
        self.linear_linear = FMLinear(linear_field_dims)

    def forward(self, fm_x, linear_x):
        x = self.linear(fm_x) + self.fm(self.embedding(fm_x)) + self.linear_linear(linear_x)
        return torch.sigmoid(x.squeeze(1))