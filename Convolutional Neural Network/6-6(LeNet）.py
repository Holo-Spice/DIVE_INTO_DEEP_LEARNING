import torch
import torch.nn as nn
import tools.utils as d2l

# net = nn.Sequential(
#     nn.Conv2d(1, 6, kernel_size=5, padding=2),
#     nn.Sigmoid(),
#     nn.AvgPool2d(kernel_size=2, stride=2),
#     nn.Conv2d(6, 16, kernel_size=5),
#     nn.Sigmoid(),
#     nn.AvgPool2d(kernel_size=2, stride=2),
#     nn.Flatten(),
#     nn.Linear(16 * 5 * 5, 120),
#     nn.Sigmoid(),
#     nn.Linear(120, 84),
#     nn.Sigmoid(),
#     nn.Linear(84, 10)
# )
# net = nn.Sequential(
#     nn.Conv2d(1, 6, kernel_size=5, padding=2),
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size=2, stride=2),
#     nn.Conv2d(6, 16, kernel_size=5),
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size=2, stride=2),
#     nn.Flatten(),
#     nn.Linear(16 * 5 * 5, 120),
#     nn.ReLU(),
#     nn.Linear(120, 84),
#     nn.ReLU(),
#     nn.Linear(84, 10)
# )
net = nn.Sequential(
    nn.Conv2d(1,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
    nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
    nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
    nn.Flatten(),
    nn.Linear(128*3*3,512), nn.ReLU(),  # 输入维度1152
    nn.Linear(512,256), nn.ReLU(),
    nn.Linear(256,10)
)


def main():
    X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape: \t', X.shape)

    batch_size = 256
    lr, num_epochs = 0.05, 30
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

    d2l.predict_ch6(net, test_iter)

if __name__ == '__main__':
    main()