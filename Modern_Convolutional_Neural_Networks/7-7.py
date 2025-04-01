import torch
from torch import nn
import tools.utils as d2l

device = d2l.try_gpu()

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
    ).to(device)

# 稠密块
class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super().__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer).to(device)

    def forward(self, x):
        for blk in self.net:
            Y = blk(x)
            x = torch.cat((x, Y), dim=1)
        return x

# 过渡层,通过1*1卷积层来减小通道数 控制模型复杂度 并使用步幅为2的平均汇聚层减半高和宽，从而进一步降低模型复杂度
def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),
        nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    ).to(device)

def main():
    timer = d2l.Timer()

    # num_channels为当前的通道数
    num_channels, growth_rate = 64, 32
    num_convs_in_dense_clock = [4, 4, 4, 4]
    blks = []

    for i, num_convs in enumerate(num_convs_in_dense_clock):
        blks.append(DenseBlock(num_convs, num_channels, growth_rate))
        # 上一个稠密块的输出通道数
        num_channels += num_convs * growth_rate
        # 使用过渡层
        if i != len(num_convs_in_dense_clock) - 1:
            blks.append(transition_block(num_channels, num_channels // 2))
            num_channels //= 2

    b1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    net = nn.Sequential(
        b1, *blks,
        nn.BatchNorm2d(num_channels),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(num_channels, 10)
    ).to(device)

    lr, num_epochs, batch_size = 0.001, 15, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, device)


    total_time = timer.stop()
    format_time = timer.format_time(total_time)
    print('Total Training time:', format_time)

    d2l.predict_ch6(net, test_iter, (96, 96))

if __name__ == '__main__':
    main()