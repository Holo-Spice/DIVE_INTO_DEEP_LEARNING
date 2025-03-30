import torch
from torch import nn
import tools.utils as d2l

device = d2l.try_gpu()
# blk = d2l.Residual(3, 3)
# X = torch.rand(4, 3, 6, 6)
# Y = blk(X)
# print(Y.shape)
#
# blk = d2l.Residual(3, 6, use_1x1conv=True, strides=2)
# print(blk(X).shape)

# 构建残差块列表
def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(d2l.Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(d2l.Residual(num_channels, num_channels))
    return blk


def main():
    timer = d2l.Timer()

    b1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(3, 2, padding=1)
    ).to(device)
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True)).to(device)
    b3 = nn.Sequential(*resnet_block(64, 128, 2)).to(device)
    b4 = nn.Sequential(*resnet_block(128, 256, 2)).to(device)
    b5 = nn.Sequential(*resnet_block(256, 512, 2)).to(device)

    net = nn.Sequential(b1, b2, b3, b4, b5,
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(),
                        nn.Linear(512, 10)).to(device)

    X = torch.rand(size=(1, 1, 224, 224)).to(device)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, "shape:\t", X.shape)

    lr, num_epochs, batch_size = 0.05, 10, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, device)

    # Stop the timer after training finishes
    total_time = timer.stop()
    # Use the format_time method to format the total time
    formatted_time = timer.format_time(total_time)
    print(f'Total training time: {formatted_time}')

    d2l.predict_ch6(net, test_iter, (96, 96))

if __name__ == '__main__':
    main()

