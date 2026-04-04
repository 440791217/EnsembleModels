import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
# from torchvision.models



ResNet18='resnet18'
ResNet34='resnet34'
ResNet50='resnet50'
ResNet101='resnet101'
ResNet152='resnet152'

DATASET_CIFAR_10='cifar10'
DATASET_CIFAR_100='cifar100'

EPOCHS = 200

# DATASET=DATASET_CIFAR_100
DATASET=DATASET_CIFAR_10
# NUM_CLASSES = 10
modelName=ResNet18