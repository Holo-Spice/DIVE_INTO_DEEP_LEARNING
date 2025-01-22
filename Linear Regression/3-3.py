import torch
from tools.utils import synthetic_data
from tools.utils import load_arry
from torch import nn

# 生成数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# 读取数据集
batch_size = 10
data_iter = load_arry((features, labels), batch_size)
print(next(iter(data_iter)))

# 定义模型
net = nn.Sequential(nn.Linear(2, 1))

# 初始化模型参数
net[0].weight.data.normal_(0, 0.01)  # 权重
net[0].bias.data.fill_(0)  # 偏置

# 定义损失函数
loss = nn.MSELoss()  # 均方误差损失函数

# 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)  # 小批量随机梯度下降算法

# 训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, Y in data_iter:
        l = loss(net(X), Y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'eopch:{epoch+1}, loss:{l:f}')



