import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
# from torchvision.models



ResNet18='resnet18'
ResNet34='resnet34'
ResNet50='resnet50'
ResNet101='resnet101'
ResNet152='resnet152'

# ResNet18='q8resnet18'
# ResNet34='q8resnet34'

DATASET_CIFAR_10='cifar10'
DATASET_CIFAR_100='cifar100'
DATASET_CIFAR_100_COARSE='cifar100-coarse'
DATASET_CIFAR_100_SUPER='cifar100-super'
DATASET_IMAGENET_1000='imagenet1000'

EPOCHS = 300

# DATASET=DATASET_IMAGENET_1000
# DATASET=DATASET_CIFAR_100_COARSE
# DATASET=DATASET_CIFAR_100_SUPER
DATASET=DATASET_CIFAR_10
# MODEL_NAME=ResNet152
MODEL_DIR_PATH=os.path.join('.','models',DATASET)
#optimizer
LR=0.1
MOMENTUM=0.9
WEIGHT_DECAY=5e-4

STEP_SIZE=25
GAMMA=0.5

