import torch
from torch import nn
from tools.utils import load_data_fashion_mnist
from tools.utils import train_ch3
from tools.utils import predict_ch3


def init_weights(m):
    # 若模型是全连接层，则使用正态分布初始化权重
    if type(m) == nn.Linear:
        # 使用正态分布（均值0， 标准差0.01）初始化权重
        nn.init.normal_(m.weight, std=0.01)


def main():
    net = nn.Sequential(nn.Flatten(),  # 将输入展平 28*28->784
                        nn.Linear(784, 256),  # 全连接层：将784维输入映射到256维隐藏层
                        nn.ReLU(),  # ReLU激活函数，增加网络的非线性表达能力
                        nn.Linear(256, 10)  # 全连接层：将256维隐藏层映射到10个输出
                        )
    net.apply(init_weights)  # initialize

    # 训练
    batch_size, lr, num_epochs = 256, 0.05, 20
    # 交叉熵损失函数
    loss = nn.CrossEntropyLoss()
    # 定义优化器，使用随机梯度下降（SGD）算法来更新网络的所有可学习参数
    trainer = torch.optim.SGD(net.parameters(), lr)

    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    # 开始训练模型
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    # 在测试集上进行预测，评估模型的性能
    predict_ch3(net, test_iter)


if __name__ == '__main__':
    main()
