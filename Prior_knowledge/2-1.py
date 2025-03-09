import torch

x = torch.arange(12)
print(x)
print(x.numel())
print(x.shape, "\n")

y = x.reshape(3, 4)
print(y)
print(y.numel())
print(x.shape, "\n")

x = torch.zeros(2, 3, 4)
print(x)
print(x.numel())
print(x.shape, "\n")

x = torch.ones(2, 3, 4)
print(x)
print(x.numel())
print(x.shape, "\n")

x = torch.randn(3, 4)  # 正态分布
print(x)
print(x.numel())
print(x.shape, "\n")

x = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(x)
print(x.numel())
print(x.shape, "\n")

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(f"{x ** y}\n")

y = torch.exp(x)
print(y)
print(y.numel())
print(x.shape, "\n")

X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(X)
print(Y)
x = torch.cat((X, Y), dim=0)  # x轴
print(x)
print(x.shape)
y = torch.cat((X, Y), dim=1)  # y轴
print(y)
print(f"{y.shape}\n")

print(f"{X == Y}\n")

x = X.sum()  # 对张量中的所有元素进行求和，会产生一个单元素张量
print(f"{x}\n")

# 广播机制
# 矩阵a将复制列， 矩阵b将复制行，然后再按元素相加
a = torch.arange(3).reshape(3, 1)
b = torch.arange(2).reshape(1, 2)
print(a)
print(b, '\n')
print(a + b, '\n')

print(X)
print(X[1:2],'n')

before = id(Y)
Y = Y + X
print(f"{before} {id(Y) == before} {id(Y)}\n")

Z = torch.zeros_like(Y)
print(f"before id(Z):{id(Z)}")
Z[:] = X + Y
print(f"after Z[:] = X + Y id(Z):{id(Z)}")

print(f"id(X) : {id(X)}")
X += Y
print(f"X+=Y id(X): {id(X)}")
X = X + Y
print(f"X = X + Y id(X):{id(X)}")

A = X.numpy()
B = torch.tensor(A)
print(type(A),type(B),'\n')

a = torch.tensor([3.5])
print(a)
print(a.item())
print(float(a))
print(int(a))