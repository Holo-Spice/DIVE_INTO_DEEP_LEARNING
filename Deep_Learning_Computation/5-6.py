import torch
from torch import nn
import tools.utils as dl2

print(torch.device('cpu'),torch.device('cuda'),torch.device('cuda:0'))
print(torch.cuda.device_count())

print(dl2.try_gpu(0))
print(dl2.try_gpu(5))
print(dl2.try_all_gpus())

x = torch.rand(3, 2)
print(x.device)

X = torch.ones(2, 3, device = dl2.try_gpu())
print(X)

Z = x.cuda(0)
print(Z.cuda(0) is Z)
print(Z)
print(X)
print(torch.matmul(X, Z))
#print(torch.matmul(X, x))

net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=dl2.try_gpu())
print(net(X))
print(net[0].weight.data.device)