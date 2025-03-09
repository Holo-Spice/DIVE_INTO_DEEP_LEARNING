import torch
from torch import nn
import torch.optim
from tools.utils import load_data_fashion_mnist
from tools.utils import train_ch3
from tools.utils import predict_ch3

# 初始化模型参数
num_inputs, num_outputs, num_hiddens = 784, 10, 256
# 第一层：输入到隐藏层
w1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
# 第二层：隐藏层到输出层
w2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))


# 激活函数
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(a, X)


# 模型
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X @ w1 + b1)  # @ 矩阵乘法
    return H @ w2 + b2


def main():
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    params = [w1, b1, w2, b2]

    # 定义交叉熵损失函数，不对批量中的损失进行聚合
    # 这将返回每个样本的损失值，方便后续对每个样本的损失进行单独处理或分析
    loss = nn.CrossEntropyLoss(reduction='none')

    # 训练
    num_epochs, lr = 20, 0.05
    # 使用随机梯度下降（SGD）优化器
    updater = torch.optim.SGD(params, lr=lr)
    # 训练模型
    train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
    # 在测试集上进行预测，评估模型的性能
    predict_ch3(net, test_iter)


if __name__ == '__main__':
    main()
