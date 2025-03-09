import torch
import numpy as np
from tools.utils import plot

x = torch.arange(4.0)
print(x)
x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=True)

y = 2 * torch.dot(x, x)
print(y)

y.backward()
print(x.grad)

print(x.grad == 4 * x)

# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)

# 非标量变量的反向传播
# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
# 本例只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
y = x * x
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
print(x.grad)

# 分离计算
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
print(x.grad)
print(x.grad == u)

x.grad.zero_()
y.sum().backward()
print(x.grad == 2 * x)


# Python控制流的梯度计算
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
print(f"Input a: {a.item()}")
print(f"Output d: {d.item()}")
print(f"Gradient of a: {a.grad.item()}")
print(a.grad == d / a)

# 创建一个包含 100 个等间隔值 的 1D 张量，范围从 −2pi - 2pi
x = torch.linspace(-2 * torch.pi, 2 * torch.pi, 100, requires_grad=True)
y = torch.sin(x)  # 计算 f(x) = sin(x)

y.sum().backward()
dy_dx = x.grad  # 获取梯度，即 dy/dx
# 转换为 NumPy 数据以便绘图
x_np = x.detach().numpy()
y_np = y.detach().numpy()
dy_dx_np = dy_dx.detach().numpy()

# 调用 `plot` 函数绘制图像
plot(
    X=[x_np, x_np],  # x 数据 (重复两次)
    Y=[y_np, dy_dx_np],  # 对应的 y 数据和 dy/dx 数据
    xlabel="x",  # x 轴标签
    ylabel="y",  # y 轴标签
    legend=["f(x) = sin(x)", "df/dx"],  # 图例
    xlim=(-2 * np.pi, 2 * np.pi),  # x 轴范围
    ylim=(-1.5, 1.5),  # y 轴范围
    figsize=(6, 4)  # 图像大小
)