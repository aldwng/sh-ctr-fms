import torch
import torch.nn as nn

class FM(nn.Module):

    def __init__(self, n, k):
        super(FM, self).__init__()
        self.n = n
        self.k = k
        self.linear = nn.Linear(self.n, 1, bias=True)
        self.vm = nn.Parameter(torch.Tensor(self.k, self.n))
        nn.init.xavier_uniform_(self.vm)

    def forward(self, x):
        x1 = self.linear(x)
        square_of_sum = torch.mm(x, self.vm.T) * torch.mm(x, self.vm.T)
        sum_of_square = torch.mm(x * x, self.vm.T * self.vm.T)
        x2 = 0.5 * torch.sum((square_of_sum - sum_of_square), dim=-1, keepdim=True)
        x = x1 + x2
        return x