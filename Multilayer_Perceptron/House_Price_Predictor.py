from xml.sax.handler import all_features

import numpy as np
import pandas as pd
import torch
from sympy.logic.inference import valid

import tools.utils as dl2
from torch import nn

train_date = pd.read_csv(dl2.download('kaggle_house_train'))
test_date = pd.read_csv(dl2.download('kaggle_house_test'))

"""检查数据"""
print(train_date.shape)
print(test_date.shape)
print(train_date.iloc[:4, [0, 1, 2, 3, -3, -2, -1]])

all_features = pd.concat((train_date.iloc[:, 1:-1], test_date.iloc[:, 1:]))
"""数据预处理"""
# 若无法获得测试数据，则根据训练数据计算均值和标准差
# 对数值特征进行标准化(object为字符串或混合数据) 均值0 方差1
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 将缺失值设置为均值0
all_features[numeric_features] = all_features[numeric_features].fillna(0)
# “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
all_features.fillna({'MasVnrType': 'None'}, inplace=True)
all_features = pd.get_dummies(all_features, dummy_na=True, dtype=int)
print(all_features.shape)

n_train = train_date.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(
train_date.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

"""训练"""
loss = nn.MSELoss()
in_features = train_features.shape[1]


def get_net():
    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(in_features, 1024),
                        nn.ReLU(),
                        #nn.Dropout(0.2),
                        nn.Linear(1024, 1),
                        )
    return net


def log_rmse(net, features, labels):
    # 计算预测值与真实标签之间的对数均方根误差（RMSE）。
    # 为了稳定对数运算，预测值小于1的部分会被截断为1。
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log((labels))))
    return rmse.item()


# 训练
def train(net, train_features, train_labels, test_features, test_labels, num_epochs, lr, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = dl2.load_arry((train_features, train_labels), batch_size)
    # 使用Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        #print(f"epoch: {epoch} / {num_epochs}")
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


# K折交叉验证
def get_k_fold_data(k, i, X, y):
    """
    获取第 i 折的训练集和验证集数据，用于 k 折交叉验证。

    参数：
        k (int): 折数，必须大于1。
        i (int): 当前折的索引（0 <= i < k），作为验证集。
        X (Tensor): 特征数据集，形状为 (样本数, 特征数)。
        y (Tensor): 标签数据集，形状为 (样本数, ) 或 (样本数, 1)。

    返回：
        X_train (Tensor): 训练集特征。
        y_train (Tensor): 训练集标签。
        X_valid (Tensor): 验证集特征。
        y_valid (Tensor): 验证集标签。
    """
    # 断言 k 必须大于 1，否则无法进行交叉验证
    assert k > 1

    # 计算每一折的样本数（整除，结果向下取整）
    fold_size = X.shape[0] // k

    # 初始化训练集特征和标签为 None
    X_train, y_train = None, None

    # 遍历所有折，每折 j 的数据作为一个数据块
    for j in range(k):
        # 定义当前折的样本索引范围：从 j * fold_size 到 (j+1) * fold_size
        idx = slice(j * fold_size, (j + 1) * fold_size)
        # 选取当前折的特征和标签数据
        X_part, y_part = X[idx, :], y[idx]

        # 如果当前折 j 正好是第 i 折，将其作为验证集
        if j == i:
            X_valid, y_valid = X_part, y_part
        # 否则，将当前折的数据加入训练集
        elif X_train is None:
            # 如果训练集还未初始化，直接赋值
            X_train, y_train = X_part, y_part
        else:
            # 如果训练集已经存在，则使用 torch.cat 进行拼接（沿着样本维度 0 拼接）
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)

    # 返回训练集和验证集的数据
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs, lr, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, lr, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            dl2.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k


def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    dl2.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    # 将网络应用于测试集。
    preds = net(test_features).detach().numpy()
    # 将其重新格式化以导出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)


def main():
    k = 5
    num_epochs = 100
    lr = 0.1
    weight_decay = 35
    batch_size = 256
    train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
    print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
          f'平均验证log rmse: {float(valid_l):f}')

    train_and_pred(train_features, test_features, train_labels, test_date,
               num_epochs, lr, weight_decay, batch_size)
if __name__ == '__main__':
    main()
