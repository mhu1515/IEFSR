import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class prepare_dataset():
    def __init__(self, train, low_image_size=32, high_image_size=1024, batch_size=1):
        train_dataset_path = os.path.join(train)
        train_lr_dataset_path = os.path.join('./dataset/train/enlighten/')
        x_transform = transforms.Compose([
            transforms.Resize(low_image_size),
            transforms.CenterCrop(low_image_size),
            transforms.ToTensor()
        ])
        y_transform = transforms.Compose([
            transforms.Resize(high_image_size),
            transforms.CenterCrop(high_image_size),
            transforms.ToTensor()
        ])

        x_dataset = datasets.ImageFolder(train_lr_dataset_path, x_transform)
        self.x_train_loader = DataLoader(x_dataset, batch_size=batch_size)

        y_dataset = datasets.ImageFolder(train_dataset_path, y_transform)
        self.y_train_loader = DataLoader(y_dataset, batch_size=batch_size)


class validation_dataset():
    def __init__(self, val, low_image_size=32, high_image_size=1024, batch_size=1):
        val_dataset_path = os.path.join(val)
        val_lr_dataset_path = os.path.join('./dataset/val/enlighten/')
        x_transform = transforms.Compose([
            transforms.Resize(low_image_size),
            transforms.CenterCrop(low_image_size),
            transforms.ToTensor()
        ])
        y_transform = transforms.Compose([
            transforms.Resize(high_image_size),
            transforms.CenterCrop(high_image_size),
            transforms.ToTensor()
        ])

        x_dataset = datasets.ImageFolder(val_lr_dataset_path, x_transform)
        self.x_val_loader = DataLoader(x_dataset, batch_size=batch_size)

        y_dataset = datasets.ImageFolder(val_dataset_path, y_transform)
        self.y_val_loader = DataLoader(y_dataset, batch_size=batch_size)
