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
    
    def forward(self, linear_x, fm_x):
        x = self.linear(linear_x) + self.linear(fm_x) + self.fm(self.embedding(fm_x))
        return torch.sigmoid(x.squeeze(1))