import torch
from torch import nn
from tools.utils import synthetic_data
from tools.utils import load_array
from tools.utils import linreg
from tools.utils import squared_loss
from tools.utils import Animator
from tools.utils import sgd
from tools.utils import evaluate_loss

# 特征数（200）远大于训练样本数（20）过拟合
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = synthetic_data(true_w, true_b, n_train)
train_iter = load_array(train_data, batch_size)
test_data = synthetic_data(true_w, true_b, n_test)
test_iter = load_array(test_data, batch_size, is_train=False)


# 初始化模型参数
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


# 定义L2范数惩罚
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2


# 定义训练代码实现
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: linreg(X, w, b), squared_loss
    num_epochs, lr = 100, 0.003
    animator = Animator(xlabel='epochs', ylabel='loss', yscale='log',
                        xlim=[1, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 增加了L2范数惩罚项，
            # 广播机制使l2_penalty(w)成为一个长度为batch_size的向量
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))

    animator.show()
    print('w的L2范数是：', torch.norm(w).item())


# train简洁实现
def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    # 对所有参数进行正态分布初始化
    for param in net.parameters():
        param.data.normal_()
    # 定义均方误差损失函数
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    trainer = torch.optim.SGD([
        # 对权重参数应用L2正则化（权重衰减）
        {"params": net[0].weight, "weight_decay": wd},  # weight_decay：权重衰减（L2 正则化）权重的衰减系数为 wd
        # 对偏置参数不应用衰减，因为通常偏置不需要进行正则化
        {"params": net[0].bias}  # 偏置没有衰减项
    ], lr=lr)  # 设置学习率为 lr
    animator = Animator(xlabel='epochs', ylabel='loss', yscale='log',
                        xlim=[1, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))

    animator.show()
    print('w的L2范数是：', net[0].weight.norm().item())


# y = 0.05 + ∑(0.01 * x_i) + ε where ε ~ N(0, 0.01^2)
def main():
    # 从零开始实现
    # 忽略正则化直接训练
    train(lambd=0)
    # 使用权重衰减
    train(lambd=3)

    # 简洁实现
    # 忽略正则化直接训练
    train_concise(0)
    # 使用权重衰减
    train_concise(3)


if __name__ == '__main__':
    main()
