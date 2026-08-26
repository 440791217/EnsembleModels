import os
import json
import ImageClassification.ResNetConfig as ResNetConfig
import ImageClassification.ResNetUtils as ResNetUtils
import numpy as np
import torch
import copy
import Logger.Logger as Logger

def LayerValueRangeForNeurons(module, input):
    pass

@torch.no_grad()
def main(modelName,dataset):
    train_loader, test_loader=ResNetUtils.GetLoader(dataset=dataset)


    pass


if __name__=='__main__':
    modelName=ResNetConfig.ResNet34
    dataset=ResNetConfig.DATASET_CIFAR_10
    pass