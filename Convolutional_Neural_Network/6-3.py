import torch
from torch import nn

"""填充"""

# 初始化卷积层权重，并对输入和输出提高和缩减相应的维数
def comp_conv2d(conv2d, X):
    # 这里的（1，1）表示批量大小和通道数都是1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 省略前两个维度：批量大小和通道
    return Y.reshape(Y.shape[2:])


print(f'[(N_h - K_h + P_h(总) + S_h) / S_h] * [(N_w - K_w + P_w + S_w) / S_w]')
# 边都填充了1行或1列，因此总共添加了2行或2列 8 + 2 - 3 + 1 = 8
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
print(
    f'input：{X.shape} kernel_size：{conv2d.kernel_size} padding：{conv2d.padding} output：{comp_conv2d(conv2d, X).shape} ')

# 当卷积核的高度和宽度不同时，我们可以填充不同的高度和宽度
conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
print(f'input：{X.shape} kernel_size：{conv2d.kernel_size} padding：{conv2d.padding} output：{comp_conv2d(conv2d, X).shape} ')



"""步幅"""

conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
print(f'input：{X.shape} kernel_size：{conv2d.kernel_size} padding：{conv2d.padding} stride:{conv2d.stride} output：{comp_conv2d(conv2d, X).shape} ')

conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
print(f'input：{X.shape} kernel_size：{conv2d.kernel_size} padding：{conv2d.padding} stride:{conv2d.stride} output：{comp_conv2d(conv2d, X).shape} ')