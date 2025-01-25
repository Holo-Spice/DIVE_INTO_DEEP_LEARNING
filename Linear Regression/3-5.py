import torchvision
from torch.utils import data
from torchvision import transforms
from tools.utils import shown_images
from tools.utils import get_fashion_mnist_labels
from tools.utils import get_dataloader_workers
from tools.utils import Timer
from tools.utils import load_data_fashion_mnist


def main():
    trans = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True
    )

    print(len(mnist_train))
    print(len(mnist_test))

    X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
    shown_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))

    batch_size = 256
    train_iter = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True,
                                 num_workers=get_dataloader_workers(8))

    timer = Timer()
    for X, y in train_iter:
        continue
    print(f'time: {timer.stop():.2f} sec')

    train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
    for X, y in train_iter:
        print(X.shape, X.dtype, y.shape, y.dtype)
        break


if __name__ == '__main__':
    main()
