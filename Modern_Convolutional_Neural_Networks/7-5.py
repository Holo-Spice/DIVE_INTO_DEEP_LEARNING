import numpy as np
import torch
from sympy import shape
from torch import nn
import tools.utils as d2l

device = d2l.try_gpu()

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 判断是训练还是预测模式
    if not torch.is_grad_enabled():
        # 预测模式下，直接使用传入的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        # 训练模式使用计算得到的均值和方差标准化
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 全连接层， 计算特征维度上的均值和方差
            mean = X.mean(dim=0)
            var = ((X - mean).pow(2).mean(dim=0))
        else:
            # 二维卷积层， 计算通道维度上(axis=1)的均值和方差
            # 保持X的形状以便后面可以做广播运算
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean).pow(2)).mean(dim=(0, 2, 3), keepdim=True)
        # 更新参数
        X_hat = (X - mean) / torch.sqrt(var + eps)
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta
    return Y, moving_mean.data, moving_var.data


class BatchNorm(nn.Module):
    # num_features: 完全连接层的输出量或卷积层的输出通道数
    # num_dims: 2全连接层 4卷积层
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度与迭代的拉伸和偏移参数， 初始化为1和0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 非模型参数的变量初始化为0和1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # 确保参数在X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新后参数
        Y, self.moving_mean, self.moving_var = batch_norm(X, self.gamma, self.beta, self.moving_mean, self.moving_var, eps=1e-5, momentum=0.9)
        return Y

timer = d2l.Timer()

# net = nn.Sequential(
#     nn.Conv2d(1, 6, kernel_size=5),
#     BatchNorm(6, 4),
#     nn.Sigmoid(),
#     nn.AvgPool2d(2, 2),
#     nn.Conv2d(6, 16, 5),
#     BatchNorm(16, 4),
#     nn.Sigmoid(),
#     nn.AvgPool2d(2, 2),
#     nn.Flatten(),
#     nn.Linear(16 * 4 * 4, 120),
#     BatchNorm(120, 2),
#     nn.Sigmoid(),
#     nn.Linear(120, 84),
#     BatchNorm(84, 2),
#     nn.Sigmoid(),
#     nn.Linear(84, 10),
# ).to(device)

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10)).to(device)


lr, num_epochs, batch_size = 0.2, 15, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, device)
# Stop the timer after training finishes
total_time = timer.stop()
# Use the format_time method to format the total time
formatted_time = timer.format_time(total_time)
print(f'Total training time: {formatted_time}')

d2l.predict_ch6(net, test_iter, (28, 28))