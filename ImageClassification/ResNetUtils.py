import os
import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import ImageClassification.ResNetConfig as ResNetConfig
import Datasets.CiFar as CiFar
import Datasets.Imagenet as Imagenet
from torchvision.models import resnet18,ResNet18_Weights,\
    resnet34,ResNet34_Weights,resnet50,ResNet50_Weights,resnet101,ResNet101_Weights


def GetLoader(dataset,batch_size=1):
    if dataset == ResNetConfig.DATASET_CIFAR_10:
        train_loader, test_loader = CiFar.GetCifar_10(batch_size=batch_size)
    elif dataset == ResNetConfig.DATASET_CIFAR_100:
        train_loader, test_loader = CiFar.GetCifar_100(batch_size=batch_size)
    elif dataset == ResNetConfig.DATASET_CIFAR_100_COARSE:
        train_loader, test_loader = CiFar.GetCifar_100_Coarse(batch_size=batch_size)
    elif dataset == ResNetConfig.DATASET_IMAGENET_1000:
        train_loader, test_loader = Imagenet.GetImagenet_1000(batch_size=batch_size)
    else:
        raise ValueError("Invalid dataset")
    return train_loader, test_loader

def GetClassNum(dataset):
    if dataset == ResNetConfig.DATASET_CIFAR_10:
        classNum = 10
    elif dataset == ResNetConfig.DATASET_CIFAR_100:
        classNum = 100
    elif dataset == ResNetConfig.DATASET_CIFAR_100_COARSE:
        classNum = 20
    elif dataset == ResNetConfig.DATASET_CIFAR_100_SUPER:
        classNum = 5
    else:
        raise ValueError("Invalid dataset")    
    return classNum

def GetTrainedModel(modelName,dataset):
    modelMaps={
        'resnet18':[resnet18,ResNet18_Weights],
        'resnet34':[resnet34,ResNet34_Weights],
        'resnet50':[resnet50,ResNet50_Weights],
        'resnet101':[resnet101,ResNet101_Weights],
    }
    if modelName not in modelMaps.keys():
        raise ValueError("Invalid dataset")
    model=modelMaps[modelName][0](modelMaps[modelName][1].DEFAULT)
    return model


if __name__=='__main__':
    
    pass