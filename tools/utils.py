import matplotlib.pyplot as plt
import time
import numpy as np
import torch


def plot_example():
    """不需要在脚本中使用，因为普通脚本自动设置了渲染格式"""
    pass


def set_figsize(figsize=None):
    """动态设置matplotlib的图表大小"""
    # 如果没有提供 figsize，读取默认值
    if figsize is None:
        figsize = (6, 4.5)  # 默认大小，可从配置文件读取
    plt.rcParams['figure.figsize'] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(X, Y=None, xlabel=None, ylabel=None, legend=None,
         xlim=None, ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    plt.show()


class Timer:
    # 记录多次运行时间
    def __init__(self):
        self.times = []
        self.start()

    # 启动计时器
    def start(self):
        self.tik = time.time()

    # 停止计时器并将时间纪录在列表中
    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    # 返回平均时间
    def avg(self):
        return sum(self.times) / len(self.times)

    # 返回时间总和
    def sum(self):
        return sum(self.times)

    # 返回累计时间
    def cumsum(self):
        return np.array(self.times).cumsum().tolist()


# 生成y = Xw + b + 噪声
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

# 定义定义模型
def linreg(X,w,b):
    """线性回归模型"""
    return torch.matmul(X, w) + b

# 定义损失函数
def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 定义优化算法
def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

