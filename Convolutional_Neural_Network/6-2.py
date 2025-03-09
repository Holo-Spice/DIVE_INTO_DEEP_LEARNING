import torch
from cv2.gapi import kernel
from torch import nn
import tools.utils as d2l


"""互相关运算"""
X = torch.tensor([[0.0, 1.0, 2.0],
                  [3.0, 4.0, 5.0],
                  [6.0, 7.0, 8.0],])
K = torch.tensor([[0.0, 1.0],
                  [2.0, 3.0]])

print(d2l.corr2d(X, K))


"""卷积层"""
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return d2l.corr2d(x, self.weight) + self.bias


"""图像中目标的边缘检测"""
X = torch.ones((6, 8))
X[:, 2:6] = 0
print(f'0为黑 1为白：\n{X}')
k = torch.tensor([[1.0, -1.0]])
print(f'卷积核：\n{k}')
Y = d2l.corr2d(X, k)
print(f'1 白到黑 -1 黑到白:\n{Y}')
print(f'翻转X后：(卷积核只能检测垂直边缘)\n{d2l.corr2d(X.t(), k)}')


"""学习卷积核"""
# 构造一个二维卷积层，单通道，形状为（1，2）的卷积核
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
# 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度）
# 其中批量大小和通道数都为1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2

for i in range(20):
    # 前向传播
    Y_hat = conv2d(X)
    # 计算均方误差
    l = (Y_hat - Y) ** 2
    # 清除梯度
    conv2d.zero_grad()
    # 计算损失函数 l 的梯度
    l.sum().backward()
    # 迭代卷积核，梯度下降
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if(i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {l.sum():.3f}')

print(f"{k}:{conv2d.weight.data.reshape(1, 2)}")