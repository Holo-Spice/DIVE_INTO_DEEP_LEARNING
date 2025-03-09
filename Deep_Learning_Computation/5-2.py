import torch
from torch import nn

""" 参数访问"""
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
print(net(X))

# 检查第二个全连接层的参数
print(net[2].state_dict())

# 目标参数
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
print(net[2].weight.grad == None)

# 一次性访问所有参数
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.data) for name, param in net[0].named_parameters()])
print(*[(name, param) for name, param in net[0].state_dict().items()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
print(net.state_dict()['2.weight'].data)


# 从嵌套块收集参数
def block1():
    return nn.Sequential(nn.Linear(4, 8),
                         nn.ReLU(),
                         nn.Linear(8, 4),
                         nn.ReLU())


def block2():
    net = nn.Sequential()
    for i in range(4):
        # 嵌套
        net.add_module(f'block{i}', block1())
    return net


rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print(rgnet(X))
print(rgnet)
print(net)
print(rgnet[0][1][0].weight.data)


"""参数初始化"""

def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)


print(f"参数初始化前：")
print(net[0].weight.data[0], net[0].bias.data[0])
net.apply(init_normal)
print(f"参数初始化后：")
print(net[0].weight.data[0], net[0].bias.data[0])


# 将所有参数初始化为给定的常数
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)


print(f"参数初始化前：")
print(net[0].weight.data[0], net[0].bias.data[0])
net.apply(init_constant)
print(f"参数初始化后：")
print(net[0].weight.data[0], net[0].bias.data[0])


# 对某些块应用不同的初始化方法
def init_xzvier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

print(f"参数初始化前：")
print(net[0].weight.data[0], net[2].weight.data[0])
net[0].apply(init_xzvier)
net[2].apply(init_42)
print(f"参数初始化后：")
print(net[0].weight.data[0], net[2].weight.data[0])


"""
w ~ {
    U(5, 10)    with probability 1/4
    0           with probability 1/2
    U(-10, -5)  with probability 1/4
}
"""
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[{name, param.data} for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5


print(f"参数初始化前：")
print(net[0].weight.data, net[2].weight.data)
net.apply(my_init)
print(f"参数初始化后：")
print(net[0].weight.data, net[2].weight.data)

net[0].weight.data[:] += 1
print("net[0].weight.data[:] += 1\n", net[0].weight.data, net[2].weight.data)
net[0].weight.data[0, 0] = 42
print("net[0].weight.data[0, 0] = 42\n",net[0].weight.data, net[2].weight.data)


"""参数绑定"""
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8),
                    nn.ReLU(),
                    shared,
                    nn.ReLU(),
                    shared,
                    nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# 检查参数
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])