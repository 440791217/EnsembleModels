import os
import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import ImageClassification.ResNetConfig as ResNetConfig
import Datasets.CiFar as CiFar


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

    return running_loss / total, correct / total


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

    return running_loss / total, correct / total


def main():
    print(ResNetConfig.MODEL_NAME)

    if ResNetConfig.DATASET == ResNetConfig.DATASET_CIFAR_10:
        train_loader, test_loader = CiFar.GetCifar_10()
        NUM_CLASSES = 10
    elif ResNetConfig.DATASET == ResNetConfig.DATASET_CIFAR_100:
        train_loader, test_loader = CiFar.GetCifar_100()
        NUM_CLASSES = 100
    elif ResNetConfig.DATASET == ResNetConfig.DATASET_CIFAR_100_COARSE:
        train_loader, test_loader = CiFar.GetCifar_100_Coarse()
        NUM_CLASSES = 20
    elif ResNetConfig.DATASET == ResNetConfig.DATASET_CIFAR_100_SUPER:
        train_loader, test_loader = CiFar.GetCifar_100_Super5()
        NUM_CLASSES = 5
    else:
        raise ValueError("Invalid dataset")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if ResNetConfig.MODEL_NAME == ResNetConfig.ResNet18:
        model = models.resnet18(weights=None)
    elif ResNetConfig.MODEL_NAME == ResNetConfig.ResNet34:
        model = models.resnet34(weights=None)
    elif ResNetConfig.MODEL_NAME == ResNetConfig.ResNet50:
        model = models.resnet50(weights=None)
    elif ResNetConfig.MODEL_NAME == ResNetConfig.ResNet101:
        model = models.resnet101(weights=None)
    elif ResNetConfig.MODEL_NAME == ResNetConfig.ResNet152:
        model = models.resnet152(weights=None)
    else:
        raise ValueError("Invalid model")

    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=ResNetConfig.LR,
        momentum=ResNetConfig.MOMENTUM,
        weight_decay=ResNetConfig.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=ResNetConfig.STEP_SIZE,
        gamma=ResNetConfig.GAMMA
    )
    
    latest_path = f"latest_{ResNetConfig.MODEL_NAME}_{ResNetConfig.DATASET}.pth"
    latest_path = os.path.join(ResNetConfig.MODEL_DIR_PATH,latest_path)
    best_path = f"best_{ResNetConfig.MODEL_NAME}_{ResNetConfig.DATASET}.pth"
    best_path = os.path.join(ResNetConfig.MODEL_DIR_PATH,best_path)

    start_epoch = 0
    best_acc = 0.0

    if os.path.exists(latest_path):
        print(f"恢复训练: {latest_path}")
        checkpoint = torch.load(latest_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint.get("best_acc", 0.0)
        print(f"从 epoch {start_epoch} 继续，best_acc={best_acc:.4f}")

    for epoch in range(start_epoch, ResNetConfig.EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
        scheduler.step()

        print(f"{ResNetConfig.MODEL_NAME}: Epoch [{epoch + 1}/{ResNetConfig.EPOCHS}]")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.4f}")
        print("-" * 50)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_acc": best_acc,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }

        torch.save(checkpoint, latest_path)

        if test_acc > best_acc:
            best_acc = test_acc
            checkpoint["best_acc"] = best_acc
            torch.save(checkpoint, best_path)

    print(f"Best Test Accuracy: {best_acc:.4f}")
    # 2. 额外保存一个纯模型（方便测试）
    torch.save(model, best_path.replace(".pth", ".m"))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    modelNames=[
        ResNetConfig.ResNet18,
        ResNetConfig.ResNet34,
        ResNetConfig.ResNet50,
        ResNetConfig.ResNet101,
        # ResNetConfig.ResNet152,
    ]
    for modelName in modelNames:
        ResNetConfig.MODEL_NAME=modelName
        main()