import torch

# 标量
x = torch.tensor(3.0)
y = torch.tensor(2.0)
print(x + y)
print(x * y)
print(x / y)
print(x ** y)

# 向量
x = torch.arange(4)
print(x)
print(x[3])

# 长度、维度、形状
print(len(x))
print(x.shape)

# 矩阵
A = torch.arange(20).reshape(5, 4)
print(A)
print(A.T)  # 转置矩阵

# 张量 是描述具有任意数量轴的维数组的通用方法。 例如，向量是一阶张量，矩阵是二阶张量
X = torch.arange(24).reshape(2,3,4)
print(X)

A = torch.arange(20,dtype=torch.float32).reshape(5,4)
B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
print(A)
print(A + B)

a = 2
X = torch.arange(24).reshape(2,3,4)
print(a + X)
print(a * X)

# 降维
x = torch.arange(4, dtype=torch.float32)
print(x)
print(x.sum())
print(A.shape)
print(A.sum())

A_sum_axis0 = A.sum(axis=0)
print(A_sum_axis0)
print(A.shape)

A_sum_axis1 = A.sum(axis=1)
print(A_sum_axis1)
print(A.shape)
print(A.sum(axis=[0, 1]))  # 结果和A.sum()相同

print(A.mean())
print(A.sum() / A.numel())

print(A.mean(axis=0))
print(A.sum(axis=0) / A.shape[0])

sum_A = A.sum(axis=1,keepdim=True)
print(sum_A)
print(A / sum_A)

print(A.cumsum(axis=1))

# 点积+
y = torch.ones(4,dtype=torch.float32)
print(x)
print(y)
print(torch.dot(x,y))

# 向量积
print(A)
print(x)
print(torch.mv(A, x))

B = torch.ones(4, 3)
print(torch.mm(A, B))

# 范数
u = torch.tensor([3.0,-4.0])
print(torch.norm(u))  # L2范数 欧几里得距离(向量元素平方和的平方根)
print(torch.abs(u).sum())  # L1范数 向量元素的绝对值之和
print(torch.norm(torch.ones(4,9)))
