import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
# from torchvision.models
import ResNetConfig


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

if __name__=='__main__':

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader,test_loader=ResNetConfig.GetCifar_10()
    model = torch.load("best_{}_cifar10.pth".format(ResNetConfig.modelName),weights_only=False)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(
        model, test_loader, criterion, DEVICE
    )
    print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.4f}")