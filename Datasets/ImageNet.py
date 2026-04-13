import torchvision
import torch

imagenet_data = torchvision.datasets.ImageNet('data/imagenet/')
val_dataset = torchvision.datasets.ImageFolder("data/imagenet/val", transform=...)