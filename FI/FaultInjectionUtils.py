import os
import json
import ImageClassification.ResNetConfig as ResNetConfig
import ImageClassification.ResNetUtils as ResNetUtils
import ImageClassification.QuantResNet as QuantResNet
import numpy as np
import torch
import FaultInjectionConfig



def LoadModel(modelName,dataset):
    print("ModelName {}, DataSet:{}".format(modelName,dataset))
    if 'cifar' in dataset:
        modelPath="best_{}_{}.m".format(modelName,dataset)
        modelPath=os.path.join('.','models',dataset,modelPath)
        model = torch.load(modelPath,weights_only=False)
    else:
        model=ResNetUtils.GetTrainedModel(modelName=modelName,dataset=dataset)
    model = model.to(FaultInjectionConfig.Device()) 
    model.eval()
    return model

def LoadQ8Model(modelName,dataset):
    print("ModelName {}, DataSet:{}".format(modelName,dataset))
    if 'cifar' in dataset:
        train_loader, test_loader = ResNetUtils.GetLoader(dataset=dataset)
        model=QuantResNet.convert_model(modelName=modelName,dataset=dataset,train_loader=train_loader,test_loader=test_loader)
    else:
        print("Invalid Model!")
        exit(-1)
    return model


def ProfileModelHook(model):
    layerId=1
    layerInfoList=[]    
    for layerName, layer in model.named_modules():
        print(layerName, layer.__class__.__name__)
        layerInfo={
            'id':layerId,
            'name':layerName,
            'className':layer.__class__.__name__,    
        }
        layerId+=1
        layerInfoList.append(layerInfo)
    return layerInfoList
