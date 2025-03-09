import random
import torch
from tools.utils import synthetic_data
from tools.utils import set_figsize
import matplotlib.pyplot as plt
from tools.utils import linreg
from tools.utils import squared_loss
from tools.utils import sgd


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i:min(i + batch_size, num_examples)]
        )
        yield features[batch_indices], labels[batch_indices]


true_w = torch.tensor([2, -3.4])
true_b = 4.2
feature, labels = synthetic_data(true_w, true_b, 1000)
print('feature: ', feature[0], '\nlabel: ', labels[0])

set_figsize()
plt.scatter(feature[:, (1)].detach().numpy(), labels.detach().numpy(), s=5)
plt.xlabel("Feature 1")
plt.ylabel("Labels")
plt.show()

bitch_size = 10
for X, y in data_iter(batch_size=bitch_size, features=feature, labels=labels):
    print(X, '\n', y)
    break

# 初始化模型参数
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 训练
lr = 0.03
num_epochs = 10
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size=bitch_size, features=feature, labels=labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, bitch_size)
    with torch.no_grad():
        train_l = loss(net(feature, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
