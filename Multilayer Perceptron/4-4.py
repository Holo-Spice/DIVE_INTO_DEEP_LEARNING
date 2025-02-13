import math
import numpy as np
import torch
from torch import nn
from tools.utils import load_array
from tools.utils import train_epoch_ch3
from tools.utils import Animator
from tools.utils import evaluate_loss


# 训练函数
def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
    # 定义均方误差损失函数
    loss = nn.MSELoss(reduction='none')
    # 获取训练数据的最后一个维度的大小
    # 通常这个值代表每个输入样本的特征数
    input_shape = train_features.shape[-1]
    # 不设置偏置，多项式中已实现
    # 一个全连接层 输入特指数为input_shape 输出为1 无偏置项
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = load_array((train_features, train_labels.reshape(-1, 1)), batch_size)
    test_iter = load_array((test_features, test_labels.reshape(-1, 1)), batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = Animator(xlabel='epoch', ylabel='loss', yscale='log',
                        xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                        legend=['train', 'test'])
    for epoch in range(num_epochs):
        print(f'epoch: {epoch + 1}/ {num_epochs}')
        train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))

    # 训练完成后显示图像
    animator.show()

    print('weight: ', net[0].weight.data.numpy())


# 计算 y = 5 + 1.2 * x - 3.4 * (x^2) / 2! + 5.6 * (x^3) / 3! + ε
# 其中 ε 服从正态分布 N(0, 0.1^2)
def main():
    # 多项式的最高阶数
    max_degree = 20
    # 训练集和测试集的大小
    n_train, n_test = 100, 100
    # 分配大量空间
    true_w = np.zeros(max_degree)
    # 真实参数
    true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

    # 生成特征数据
    features = np.random.normal(size=(n_train + n_test, 1))
    # 随机打乱 features
    np.random.shuffle(features)
    # 构造多项式特征
    poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
    print(poly_features)
    # 将特征从x！调整为x^(i+1)！ 避免非常大的梯度值或损失值
    for i in range(max_degree):
        poly_features[:, i] /= math.gamma(i + 1)

    # labels的维度：（n_train + n_test）
    # 计算标签 labels 矩阵点乘
    # - y = w_0 * x^0 + w_1 * x^1 + w_2 * x^2 + ... + w_19 * x^19
    # - w_0, w_1, ..., w_19 是真实的多项式系数
    # - x^0, x^1, ..., x^19 是对应的多项式特征
    labels = np.dot(poly_features, true_w)
    # 加入噪声 均值为 0，标准差为 0.1 的正态分布
    labels += np.random.normal(scale=0.1, size=labels.shape)

    # numpy ndarray转tensor
    true_w, features, poly_features, labels = [torch.tensor(x, dtype=torch.float32)
                                               for x in [true_w, features, poly_features, labels]]
    print(features[:2], '\n', poly_features[:2, :], '\n', labels[:2])

    # 三阶多项式函数拟合(正常)
    # 从多项式特征中选择前4个维度，即1,x,x^2/2!,x^3/3!
    train(poly_features[:n_train, :4], poly_features[n_train:, :4], labels[:n_train], labels[n_train:])

    # 线性函数拟合(欠拟合)
    # 从多项式特征中选择前2个维度，即1和x
    train(poly_features[:n_train, :2], poly_features[n_train:, :2], labels[:n_train], labels[n_train:])

    # 高阶多项式函数拟合(过拟合)
    # 从多项式特征中选取所有维度
    train(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:], num_epochs=1500)


if __name__ == '__main__':
    main()
