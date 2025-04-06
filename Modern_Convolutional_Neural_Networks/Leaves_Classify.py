import os
from PIL import Image
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from sympy.logic.inference import valid
import tools.utils as d2l
from torch import nn
from torchvision import transforms

# 读取 CSV 的标签信息
def get_labels(csv_path):
    df = pd.read_csv(csv_path)
    labels = sorted(list(set(df['label'])))
    return labels

class_to_num = None
num_to_class = None

class LeavesDataset(Dataset):
    def __init__(self, image_path, csv_path, valid_ratio=0.2, mode='train'):
        """
        image_path  :  图片路径
        csv_path    :  csv路径
        valid_ratio :  验证集划分比例
        mode        :  工作模式
        """
        self.image_path = image_path
        self.mode = mode
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        # 读取数据 并打乱顺序
        if mode != 'test':
            self.dataset = pd.read_csv(csv_path).sample(frac=1).reset_index(drop=True)
        else:
            self.dataset = pd.read_csv(csv_path)

        if mode in ('train', 'valid'):
            self.data_len = len(self.dataset.index)
            self.train_len = int(self.data_len * (1 - valid_ratio))
            train_data = self.dataset.iloc[:self.train_len].reset_index(drop=True)
            valid_data = self.dataset.iloc[self.train_len:].reset_index(drop=True)
            # 选择子集
            self.dataset = train_data if mode == 'train' else valid_data
        print(f'mode: {mode}, finish data loading...')


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        fname = row.iloc[0]
        image_path = os.path.join(self.image_path, fname)
        # 读取图片
        image = Image.open(image_path)
        # 处理图像
        image = self.transform(image)

        if self.mode == 'test':
            return image, fname
        else:
            label = row.iloc[1]
            number_label = class_to_num[label]
            return image, number_label

def load_config(path='../cfg/leaves_classify.yaml'):
    """从 YAML 文件加载配置"""
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

# 构建残差块列表
def resnet_block(input_channels, num_channels, num_residuals, first_block=False, dropout_rate=0.2):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(d2l.Residual(input_channels, num_channels, use_1x1conv=True, strides=2, dropout_rate=dropout_rate))
        else:
            blk.append(d2l.Residual(num_channels, num_channels, dropout_rate=dropout_rate))
    return blk

def predict_direct(model, test_loader, device):
    model.eval()
    all_preds = []
    all_fnames = []
    with torch.no_grad():
        for images, fnames in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_fnames.extend(fnames)
    return pd.DataFrame({
        'image': all_fnames,
        'label': [num_to_class[p] for p in all_preds]
    })

def main():
    timer = d2l.Timer()
    # 加载配置
    cfg = load_config()
    image_path = cfg['image_path']
    train_csv = cfg['train_csv']
    test_csv = cfg['test_csv']
    batch_size = cfg.get('batch_size', 32)
    valid_ratio = cfg.get('valid_ratio', 0.2)
    device = cfg.get('device', 'cpu')
    lr = cfg.get('lr', 0.001)
    num_epochs = cfg.get('num_epochs', 10)
    # 打印参数
    print("image_path:", image_path)
    print("train_csv:", train_csv)
    print("test_csv:", test_csv)
    print("batch_size:", batch_size)
    print("valid_ratio:", valid_ratio)
    print("device:", device)
    print("lr:", lr)
    print("num_epochs:", num_epochs)
    # 获取所有的标签，并构建映射字典
    global class_to_num, num_to_class
    leaves_labels = get_labels(train_csv)
    n_classes = len(leaves_labels)
    class_to_num = dict(zip(leaves_labels, range(n_classes)))
    num_to_class = {v: k for k, v in class_to_num.items()}
    print("Number of classes:", n_classes)
    print("Sample mapping:", dict(list(class_to_num.items())[:10]))

    # 构建数据集和 DataLoader
    train_ds = LeavesDataset(image_path, train_csv, valid_ratio=valid_ratio, mode='train')
    valid_ds = LeavesDataset(image_path, train_csv, valid_ratio=valid_ratio, mode='valid')
    test_ds = LeavesDataset(image_path, test_csv, mode='test')

    train_iter = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_iter = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_iter = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    b1 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(3, 2, padding=1)
    ).to(device)
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True, dropout_rate=0.2)).to(device)
    b3 = nn.Sequential(*resnet_block(64, 128, 2)).to(device)
    b4 = nn.Sequential(*resnet_block(128, 256, 2)).to(device)
    b5 = nn.Sequential(*resnet_block(256, 512, 2)).to(device)
    #b6 = nn.Sequential(*resnet_block(512, 1024, 2)).to(device)
    net = nn.Sequential(b1, b2, b3, b4, b5,
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(),
                        nn.Dropout(0.5),
                        nn.Linear(512, n_classes)).to(device)

    d2l.train_ch6(net, train_iter, valid_iter, num_epochs, lr, device)
    # Stop the timer after training finishes
    total_time = timer.stop()
    # Use the format_time method to format the total time
    formatted_time = timer.format_time(total_time)
    print(f'Total training time: {formatted_time}')

    # 保存模型参数
    torch.save(net.state_dict(), 'leaf_classifier.pth')
    print(f'Model saved!')

    def load_and_predict(model_path, test_loader, device):
        # 必须重新初始化模型结构
        loaded_net = nn.Sequential(b1, b2, b3, b4, b5,
                                   nn.AdaptiveAvgPool2d((1, 1)),
                                   nn.Flatten(),
                                   nn.Dropout(0.5),
                                   nn.Linear(512, n_classes)).to(device)
        loaded_net.load_state_dict(torch.load(model_path))
        loaded_net.eval()

        return predict_direct(loaded_net, test_loader, device)

    loaded_result = load_and_predict('leaf_classifier.pth', test_iter, device)
    loaded_result.to_csv('loaded_prediction.csv', index=False)
    print(f'save loaded_prediction.csv!')

if __name__ == '__main__':
    main()