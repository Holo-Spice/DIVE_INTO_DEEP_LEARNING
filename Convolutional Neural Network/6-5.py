import torch
from torch import nn
import torch.nn.functional as F


def pool2d(X, pool_size, mode = 'max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i:i + p_h, j:j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i:i + p_h, j:j + p_w].mean()
    return Y

X = torch.arange(49, dtype=torch.float32).reshape((1, 1, 7, 7))  # 7x7 输入
print("input1: \n", X)

conv2d = nn.Conv2d(1, 1, 3, stride=1, padding=1)  # 3x3 卷积核
X_l = F.pad(X, (1, 0), "constant", 0)[:, :, :, :-1]  # X 向右移动一格
print("input2: \n", X_l)

output_conv1 = conv2d(X)
output_conv2 = conv2d(X_l)

pool = nn.MaxPool2d(kernel_size=6, stride=1)
output_pool1 = pool(output_conv1)
output_pool2 = pool(output_conv2)

print("\n卷积层输出（input1）：")
print(output_conv1)
print("卷积层输出（input2）：")
print(output_conv2)

print("\n池化层输出（input1）：")
print(output_pool1)
print("池化层输出（input2）：")
print(output_pool2)


