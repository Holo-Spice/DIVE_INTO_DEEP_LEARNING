import cv2
import torch
import torchvision
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
import tools.utils as d2l

batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10, 3)

def apply(img, aug, num_rows=2, num_cols=4, scale=4.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.shown_images(Y, num_rows=num_rows, num_cols=num_cols, scale=scale)

def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root="../data", train=is_train,
                                           transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                    shuffle=is_train, num_workers=d2l.get_dataloader_workers())
    return dataloader

def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)


def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, 15, devices)

def main():
    # plt.figure(figsize=(3, 4))
    # img = Image.open("1.jpg")
    # d2l.plt.imshow(img)
    # plt.show()
    #
    # """翻转和裁剪"""
    # # 在给定概率下对输入图像做水平翻转 默认0.5
    # apply(img, torchvision.transforms.RandomHorizontalFlip())
    # # 在给定概率下对输入图像做垂直（上下）翻转的增广操作 默认0.5
    # apply(img, torchvision.transforms.RandomVerticalFlip())
    # # 随机裁剪一个面积为原始面积10%到100%的区域，该区域的宽高比从0.5～2之间随机取值。 然后，区域的宽度和高度都被缩放到200像素
    shape_aug = torchvision.transforms.RandomResizedCrop((200, 200), scale=(0.1, 1), ratio=(0.5, 2))
    # apply(img, shape_aug)
    #
    # """改变颜色"""
    # # 随机更改图像的亮度，随机值为原始图像的50%（1-0.5）到150%（1+0.5）之间
    # apply(img, torchvision.transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0))
    # # 随机更改图像的色调
    # apply(img, torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.5))
    # # 改变亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue）
    color_aug = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    # apply(img, color_aug)
    #
    # """结合多种图像增广方法"""
    # augs = torchvision.transforms.Compose([
    #     torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug
    # ])
    # apply(img, augs)

    """使用图像增广进行训练"""
    all_images = torchvision.datasets.CIFAR10(train=True, root="../data",
                                              download=True)
    d2l.shown_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8);

    train_augs = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),color_aug, shape_aug,
        torchvision.transforms.ToTensor()])

    test_augs = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()])

    net.apply(init_weights)

    train_with_data_aug(train_augs, test_augs, net)


if __name__ == '__main__':
    main()