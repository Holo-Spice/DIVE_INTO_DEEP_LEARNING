import torch
from tools.utils import plot

# 激活函数：通过加权和并加上偏置判断神经元是否需要激活 引入非线性

# ReLU函数  ReKU(x) = max(0, x)
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
y.backward(torch.ones_like(x), retain_graph=True)
plot(x.detach(), x.grad, 'x', 'grade of relu', figsize=(5, 2.5))

# sigmod函数 sigmoid(x) = 1 / (1 + exp(-x)) 将输入变换为区间(0, 1)上的输出
y = torch.sigmoid(x)
plot(x.detach(), y.detach(), 'x', 'sigmod(x)', figsize=(5, 2.5))
# 清除梯度
x.grad.zero_()
y.backward(torch.ones_like(x), retain_graph=True)
plot(x.detach(), x.grad, 'x', 'grade of sigmod', figsize=(5, 2.5))

# tanH函数 tanh(x) = 1- exp(-2x) / (1 + exp(-2x)) 将输入变换为区间(-1, 1)上的输出
y = torch.tanh(x)
plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
# 清除梯度
x.grad.zero_()
y.backward(torch.ones_like(x), retain_graph=True)
plot(x.detach(), x.grad, 'x', 'grade of tanh', figsize=(5, 2.5))