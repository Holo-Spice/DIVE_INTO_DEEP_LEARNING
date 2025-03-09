import torch
from torch import nn
import tools.utils as d2l

num_inputs, num_outputs, num_hidden1, num_hidden2 = 784, 10, 256, 256
dropout1, dropout2 = 0.2, 0.3


def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # 丢弃所有元素
    if dropout == 1:
        return torch.zeros_like(X)
    # 保留所有元素
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1 - dropout)


# 定义模型
# 常见的技巧是在靠近输入层的地方设置较低的暂退概率
class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden1, num_hidden2, is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hidden1)
        self.lin2 = nn.Linear(num_hidden1, num_hidden2)
        self.lin3 = nn.Linear(num_hidden2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 训练时使用dropout层
        if self.training == True:
            # 在第一个全连接层后添加一个dropout层
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            # 在第二个全连接层后添加一个dropout层
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


def main():
    # 从零开始实现
    X = torch.arange(16, dtype=torch.float32).reshape((2, 8))
    print(X)
    print(dropout_layer(X, 0.))
    print(dropout_layer(X, 0.5))
    print(dropout_layer(X, 1.0))

    net = Net(num_inputs, num_outputs, num_hidden1, num_hidden2)
    """训练和测试"""
    num_epochs, lr, batch_size = 50, 0.5, 256
    loss = nn.CrossEntropyLoss(reduction='none')
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    trainer = torch.optim.SGD(net.parameters(), lr)
    #d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

    # 简洁实现
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(num_inputs, num_hidden1),
        nn.ReLU(),
        # 在第一个全连接层之后添加一个dropout层
        #nn.Dropout(dropout1),
        nn.Linear(num_hidden1, num_hidden2),
        nn.ReLU(),
        # 在第二个全连接层之后添加一个dropout层
        #nn.Dropout(dropout2),
        nn.Linear(num_hidden2, num_outputs)
    )

    net.apply(init_weights)
    trainer = torch.optim.SGD(net.parameters(), lr)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)


if __name__ == '__main__':
    main()
