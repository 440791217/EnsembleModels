import os
import json
import ResNetConfig
import numpy as np
import pandas as pd
import torch

def avg_acc(goldenResultDirs):
    

if __name__=='__main__':
    print("Average!")
    modelNames=[
        ResNetConfig.ResNet18,
        ResNetConfig.ResNet34,
        ResNetConfig.ResNet50,
        # ResNetConfig.ResNet101,
    ]
    dataset=ResNetConfig.DATASET_CIFAR_10
    resultDir=os.path.join('golden',dataset,modelNames[0])
    fns=os.listdir(resultDir)
    goldenResultDirs=[]
    for modelName in modelNames:
        resultDir=os.path.join('golden',ResNetConfig.DATASET,modelName)
        goldenResultDirs.append(resultDir)

