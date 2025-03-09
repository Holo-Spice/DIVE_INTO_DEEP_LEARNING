import math

import numpy as np
import torch
from tools.utils import Timer
from tools.utils import plot

""" 矢量化加速"""
n = 100000
a = torch.ones([n])
b = torch.ones([n])

c = torch.zeros(n)
timer = Timer()
for i in range(n):
    c[i] = a[i] + b[i]
print(f'{timer.stop():.8f} sec')

timer.start()
d = a + b
print(f'{timer.stop():.8f} sec')

"""正态分布与平方损失"""


def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma ** 2)
    return p * np.exp(-0.5 / sigma ** 2 * (x - mu) ** 2)


x = np.arange(-7, 7, 0.01)
# 均值和标准差
params = [(0, 1), (0, 2), (3, 1)]
plot(x, [normal(x, mu, sigma) for mu, sigma in params],
     xlabel='x', ylabel='p(x)',figsize=(4.5, 2.5),
     legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])