import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
# from torchvision.models
import ImageClassification.ResNetConfig as ResNetConfig
import Datasets.CiFar as CiFar
import os

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
    if ResNetConfig.DATASET == ResNetConfig.DATASET_CIFAR_10:
        train_loader, test_loader = CiFar.GetCifar_10()
        # NUM_CLASSES = 10
    elif ResNetConfig.DATASET == ResNetConfig.DATASET_CIFAR_100:
        train_loader, test_loader = CiFar.GetCifar_100()
        # NUM_CLASSES = 100
    else:
        raise ValueError("Invalid dataset")

    print(ResNetConfig.MODEL_NAME)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # train_loader,test_loader=CiFar.GetCifar_10()
    modelPath="best_{}_{}.m".format(ResNetConfig.MODEL_NAME,ResNetConfig.DATASET)
    modelPath=os.path.join(ResNetConfig.MODEL_DIR_PATH,modelPath)
    model = torch.load(modelPath,weights_only=False)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(
        model, test_loader, criterion, DEVICE
    )
    print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.4f}")

if __name__=='__main__':
    modelNames=[
        ResNetConfig.ResNet18,
        ResNetConfig.ResNet34,
        ResNetConfig.ResNet50,
        ResNetConfig.ResNet101,
        # ResNetConfig.ResNet152
    ]
    for modelName in modelNames:
        ResNetConfig.MODEL_NAME=modelName
        main()
