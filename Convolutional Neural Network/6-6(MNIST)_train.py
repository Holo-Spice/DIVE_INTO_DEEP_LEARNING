import torch
import torch.nn as nn
import tools.utils as d2l

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
    lr, num_epochs = 0.05, 5
    train_iter, test_iter = d2l.load_data_mnist(batch_size)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

    d2l.predict_ch6(net, test_iter)
    torch.save(net.state_dict(), 'mnist_cnn.pth')

if __name__ == '__main__':
    main()