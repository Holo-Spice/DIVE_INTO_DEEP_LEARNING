"""自定义层"""

import torch
from torch import nn
import torch.nn.functional as F

"""不带参数的层"""

class CenbterdLayer(nn.Module):
    def __init__(self):
        super().__init__()

    # 从其输入中减去均值
    def forward(self, X):
        return X - X.mean()


layer = CenbterdLayer()
print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))

net = nn.Sequential(nn.Linear(8, 128),
                    CenbterdLayer())
Y = net(torch.randn(4, 8))
print(Y.mean())

"""带参数的层"""
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)

linear = MyLinear(5, 3)
print(linear.weight)
print(linear(torch.rand(2, 5)))

net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
print(net(torch.rand(2, 64)))
