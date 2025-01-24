import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from tools.utils import shown_images
from tools.utils import get_fashion_mnist_labels

trans = transforms.ToTensor()
mist_train = torchvision.datasets.FashionMNIST(
    root = "../data",train=True,transform=trans,download=True
)
mist_test = torchvision.datasets.FashionMNIST(
    root = "../data",train=False,transform=trans,download=True
)

print(len(mist_train))
print(len(mist_test))

X, y = next(iter(data.DataLoader(mist_train,batch_size=18)))
shown_images(X.reshape(18,28,28),2,9,titles=get_fashion_mnist_labels(y))