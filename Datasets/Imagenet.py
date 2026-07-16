import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMAGENET_DATA_DIR='D:\\jsut\\datasets\\ImageNet'

def GetTransform():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return train_transform,test_transform

def GetImagenet_1000(batch_size=128,num_workers=8):
    train_transform,test_transform=GetTransform()
    train_dataset = datasets.ImageFolder(
        root=f"{IMAGENET_DATA_DIR}/train",
        transform=train_transform
    )

    test_dataset = datasets.ImageFolder(
        root=f"{IMAGENET_DATA_DIR}/val",
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader

if __name__=='__main__':
     train_loader, val_loader=GetImagenet_1000()
     print(len(val_loader))

