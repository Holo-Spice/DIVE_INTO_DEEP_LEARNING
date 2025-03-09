import torch
from torch import nn
import copy


def generate_data(num_samples, true_w, true_b, noise_std=0.1):
    """生成带噪声的线性数据"""
    X = torch.randn(num_samples, 2)
    y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
    y += torch.normal(0, noise_std, y.shape)
    return X, y


def train_model(X_train, y_train, X_val, y_val, lr=0.03, patience=5, num_epochs=1000):
    """模型训练函数"""
    # 数据标准化
    X_mean = X_train.mean(dim=0)
    X_std = X_train.std(dim=0)
    X_train = (X_train - X_mean) / X_std
    X_val = (X_val - X_mean) / X_std

    net = nn.Sequential(nn.Linear(2, 1))
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model = copy.deepcopy(net.state_dict())

    for epoch in range(num_epochs):
        # 训练步骤
        y_hat = net(X_train)
        train_loss = loss_fn(y_hat, y_train.reshape(-1, 1))

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # 验证步骤
        with torch.no_grad():
            val_loss = loss_fn(net(X_val), y_val.reshape(-1, 1))

        # 更新最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model = copy.deepcopy(net.state_dict())
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    net.load_state_dict(best_model)
    return net, X_mean, X_std


# 数据生成与处理
num_samples = 10000
true_w = torch.tensor([2.0, -3.0])
true_b = 4.0
lr = 0.03

X, y = generate_data(num_samples, true_w, true_b)
shuffle_idx = torch.randperm(num_samples)
X, y = X[shuffle_idx], y[shuffle_idx]

train_size = int(0.8 * num_samples)
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# 初始化模型用于打印初始参数
net = nn.Sequential(nn.Linear(2, 1))

# 打印训练前参数
print("=== 训练前参数 ===")
print(f"真实权重: {true_w.tolist()}, 真实偏置: {true_b}")
print(f"输入数据维度: {X.shape}, 训练样本数: {len(X_train)}, 验证样本数: {len(X_val)}")
print(f"数据标准化均值: {X_train.mean(dim=0).tolist()}")
print(f"数据标准化方差: {X_train.std(dim=0).tolist()}")
print(f"初始模型权重: {net[0].weight.data.tolist()[0]}")
print(f"初始模型偏置: {net[0].bias.data.item():.4f}")
print(f"学习率: {lr}\n")

# 训练模型
net, X_mean, X_std = train_model(X_train, y_train, X_val, y_val, lr)

# 参数逆标准化
w_trained = net[0].weight.data / X_std
b_trained = net[0].bias.data - torch.sum(net[0].weight.data * X_mean / X_std)

# 打印训练后参数
print("\n=== 最终参数 ===")
print("权重:", w_trained)
print("偏置:", b_trained)