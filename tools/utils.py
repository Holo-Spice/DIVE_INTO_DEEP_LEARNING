import matplotlib.pyplot as plt
import time
import numpy as np
import torch
import torchvision.datasets
from torch.utils import data
from torchvision import transforms
from IPython import display
import hashlib
import os
import tarfile
import zipfile
import requests
from torch import nn


#@save
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
DATA_HUB['kaggle_house_train'] = (
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')


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


# 在n个变量上累加
class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)

    def show(self):
        self.axes[0].cla()
        for x_vals, y_vals, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x_vals, y_vals, fmt)
        self.config_axes()
        plt.show()


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


# 生成y = Xw + b + 噪声
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


# 定义定义模型
def linreg(X, w, b):
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


# 读取数据
def load_arry(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def get_mnist_labels(labels):
    """返回MNIST数据集的文本标签"""
    text_labels = ['0', '1', '2', '3', '4',
                   '5', '6', '7', '8', '9']
    return [text_labels[int(i)] for i in labels]


# 绘制图像列表
def shown_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])

    plt.tight_layout()  # 自动调整子图间距
    plt.show()  # 显示图像
    return axes


# 使用指定线程数读取数据,默认为4
def get_dataloader_workers(num=4):
    print(f'num_workers = {num}')
    return num


# 获取和读取Fashion-MNIST数据集 返回训练集和验证集的数据迭代器
def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True
    )

    return (data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()))


# 计算预测正确的数量
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


# 计算在指定数据集上模型的精度
def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()  # 模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数，预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


# 训练模型一个迭代周期
def train_epoch_ch3(net, train_iter, loss, updater):
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


# 训练模型
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        print(f'epoch {epoch + 1} / {num_epochs}')
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    animator.show()
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


# 预测标签
def predict_ch3(net, test_iter, n=10):
    for X, y in test_iter:
        break
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    shown_images(
        X[0:n].reshape((n, 28, 28)),  # 取前n张图片
        1,  # 1行
        n,  # n列
        titles=titles[0:n]  # 标题
    )


# 评估给定数据集上模型的损失
def evaluate_loss(net, data_iter, loss):
    metric = Accumulator(2)  # 损失的总和，样本数量
    for X, y in data_iter:
        # 将输入 X 送入模型，得到预测输出 out
        out = net(X)
        # 将真实标签 y 的形状调整为与 out 相同，确保后续损失计算无误
        y = y.reshape(out.shape)
        # 计算当前批次的损失
        l = loss(out, y)
        # 累加当前批次的总损失（l.sum）和样本数量（l.numel()）
        metric.add(l.sum(), l.numel())
    # 返回平均损失
    return metric[0] / metric[1]


def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器

    Defined in :numref:`sec_linear_concise`
    """
    # 利用提供的数组构造一个TensorDataset
    # data_arrays 是一个包含多个Tensor的序列，这些Tensor的第一维度长度必须相同，
    # 每个样本的数据由这些Tensor中对应位置的数据组成
    dataset = data.TensorDataset(*data_arrays)

    # 构造并返回一个DataLoader
    # - dataset: 上面构造的TensorDataset
    # - batch_size: 每个小批量的样本数
    # - shuffle=is_train: 如果 is_train 为 True，则在每个epoch开始时将数据打乱
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


# 重塑数据形状
def reshape(x, *args, **kwargs):
    # 调用 x 自带的 reshape 方法，
    # *args 用于指定新的形状，
    # **kwargs 用于传递其他参数（如 order 参数等）
    return x.reshape(*args, **kwargs)


def linreg(X, w, b):
    """线性回归模型

    Defined in :numref:`sec_linear_scratch`"""
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    """均方损失

    Defined in :numref:`sec_linear_scratch`"""
    return (y_hat - reshape(y, y_hat.shape)) ** 2 / 2


def download(name, cache_dir=os.path.join('..', 'data')):
    """下载一个DATA_HUB中的文件，返回本地文件名"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


def download_extract(name, folder=None):
    """下载并解压zip/tar文件"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


def download_all():
    """下载DATA_HUB中的所有文件"""
    for name in DATA_HUB:
        download(name)


# 尝试返回第 i 个GPU
def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    else:
        return torch.device('cpu')


# 返回所有可用的GPU设备
def try_all_gpus():
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if len(devices) > 0 else [torch.device('cpu')]


# 计算二维互相关运算
def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


# 使用GPU计算模型在数据集上的精度
def evaulate_accuracy_gpu(net, data_iter, device=None):
    # 检查是否为pytorch模型
    if isinstance(net, nn.Module):
        # 评估模式
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


# 用GPU训练模型
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            #  Xavier 均匀分布初始化
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print(f'training no: {device}')
    net.to(device)
    # 随机梯度下降优化器
    optimizer = torch.optim.Adam(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            # 梯度清零
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            # 前向传播
            y_hat = net(X)
            # 计算损失
            l = loss(y_hat, y)
            # 反向传播
            l.backward()
            # 更新参数
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % max(1, (num_batches // 10)) == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Batch {i + 1}/{num_batches}, "
                      f"Current train loss: {train_l:.3f}, train acc: {train_acc:.3f}, device:{X.device}")
            if (i + 1) % (num_epochs // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaulate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
    animator.show()


def predict_ch6(net, test_iter, image_size=(224, 224), n=10):
    net.eval()
    device = next(net.parameters()).device
    for X, y in test_iter:
        X, y = X.to(device), y.to(device)
        break

    with torch.no_grad():
        # 前向传播
        y_hat = net(X)
        preds = get_fashion_mnist_labels(y_hat.argmax(axis=1))

    trues = get_fashion_mnist_labels(y.cpu())
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]

    # 根据传入的 image_size 参数调整图像大小
    shown_images(
        X[:n].cpu().reshape(-1, *image_size),  # 保持[B, H, W]格式，使用 image_size 调整形状
        1, n,
        titles=titles[:n]
    )



# 获取和读取MNIST数据集，返回训练集和验证集的数据迭代器
def load_data_mnist(batch_size, resize=None):
    """下载MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    # 加载MNIST训练集和测试集
    mnist_train = torchvision.datasets.MNIST(
        root="../data", train=True, transform=trans, download=True
    )
    mnist_test = torchvision.datasets.MNIST(
        root="../data", train=False, transform=trans, download=True
    )

    # 返回数据加载器
    return (data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True,
                            num_workers=4),
            data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False,
                            num_workers=4))


