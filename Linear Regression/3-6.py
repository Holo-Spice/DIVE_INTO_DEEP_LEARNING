import torch
from IPython import display
from tools.utils import load_data_fashion_mnist, evaluate_accuracy
from tools.utils import sgd
from tools.utils import train_ch3
from tools.utils import predict_ch3

# 初始化模型参数
num_inputs = 784  # 28*28的图像展开
num_outputs = 10
w = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)


# 定义softmax函数：
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition


# 定义模型
def net(X):
    # 将图像展平后再乘以权重，加上偏置，然后经过 softmax 得到概率分布
    return softmax(torch.matmul(X.reshape((-1, w.shape[0])), w) + b)


# 定义交叉熵损失函数
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])


# 使用小批量随机梯度下降优化模型损失函数
def updater(batch_size):
    return sgd([w, b], lr, batch_size)


if __name__ == '__main__':
    # 加载数据集
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)

    # 测试对 tensor 按不同维度求和
    X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print("按列求和：", X.sum(0, keepdim=True))
    print("按行求和：", X.sum(1, keepdim=True))

    # 测试 softmax 函数
    X = torch.normal(0, 1, (2, 5))
    X_prob = softmax(X)
    print("softmax输出：\n", X_prob)
    print("每行求和：", X_prob.sum(1))

    # 测试交叉熵损失函数
    y = torch.tensor([0, 2])
    y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
    print("正确类别概率：", y_hat[[0, 1], y])
    print("交叉熵损失：", cross_entropy(y_hat, y))

    # 评估模型在测试集上的准确率
    accuracy = evaluate_accuracy(net, test_iter)
    print("测试集准确率：", accuracy)

    lr = 0.1
    num_epochs = 10
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

    predict_ch3(net, test_iter)