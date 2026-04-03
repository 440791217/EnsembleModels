import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
# from torchvision.models
import ResNetConfig


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():

    EPOCHS = 100
    LR = 0.001
    NUM_CLASSES = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    
    if ResNetConfig.modelName==ResNetConfig.ResNet18:
        model = models.resnet18(weights=None)
    elif ResNetConfig.modelName==ResNetConfig.ResNet34:
        model = models.resnet34(weights=None)
    elif ResNetConfig.modelName==ResNetConfig.ResNet50:
        model = models.resnet50(weights=None)
    elif ResNetConfig.modelName==ResNetConfig.ResNet101:
        model = models.resnet101(weights=None)
    elif ResNetConfig.modelName==ResNetConfig.ResNet152:
        model = models.resnet152(weights=None)
    else:
        print("Invalid Models!")
        exit(-1)


    # 改成更适合 CIFAR-10 的结构
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_acc = 0.0

    ####
    train_loader,test_loader=ResNetConfig.GetCifar_10()
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, DEVICE
        )
        scheduler.step()

        print(f"Epoch [{epoch + 1}/{EPOCHS}]")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.4f}")
        print("-" * 50)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model, "best_{}_cifar10.pth".format(ResNetConfig.modelName))

    print(f"Best Test Accuracy: {best_acc:.4f}")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()