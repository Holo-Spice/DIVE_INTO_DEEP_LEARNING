import torch
from torch import nn
import tools.utils as d2l

device = d2l.try_gpu()
net = nn.Sequential(nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
                    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
                    nn.Conv2d(384, 256, kernel_size=3, padding=1),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nn.Flatten(),
                    nn.Linear(256 * 5 * 5, 4096), nn.ReLU(),
                    nn.Dropout(p=0.5),
                    nn.Linear(4096, 4096), nn.ReLU(),
                    nn.Dropout(p=0.5),
                    nn.Linear(4096, 10)
                    ).to(device)


def main():
    X = torch.randn(1, 1, 224, 224, device=device)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:', X.shape)
    print('net device: ', next(net.parameters()).device)

    lr, num_epochs = 0.03, 15
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, device)
    d2l.predict_ch6(net, test_iter)





if __name__ == '__main__':
    main()


