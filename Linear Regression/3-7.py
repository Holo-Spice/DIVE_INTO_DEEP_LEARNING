import torch
from torch import nn
from tools.utils import load_data_fashion_mnist
from tools.utils import train_ch3

def init_weights(m):
    if type(m) == nn.Linear:
        # 使用正态分布（均值0， 标准差0.01）初始化权重
        nn.init.normal_(m.weight, std=0.01)

def main():
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    # 1. 28*28 图片展平为784维向量
    # 2. 全链接层 784维输入映射到10个类别输出
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    # 初始化权重函数应用到每一层
    net.apply(init_weights)

    # 定义损失函数
    loss = nn.CrossEntropyLoss(reduction='none')
    # 定义优化器
    trainer = torch.optim.SGD(net.parameters(), lr = 0.1)

    num_epochs = 10
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
if __name__ == '__main__':
    main()
