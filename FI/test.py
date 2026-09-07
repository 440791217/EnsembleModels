# import sys
# print(sys.path)
# import os
# print(os.getenv("API_KEY"))  # 能直接读到
# print("当前工作区根目录 =", os.getcwd())


import os
import json
import ImageClassification.ResNetConfig as ResNetConfig
import ImageClassification.ResNetUtils as ResNetUtils
import numpy as np
import torch

def fault_hook(module, input, output):
    mydata=module.mydata
    print("layerId:{},name:{},shape:{}".format(mydata['layerId'],mydata['name'],output.shape))
    pass


def main(modelName,dataset):
    print("ModelName {}, DataSet:{}".format(modelName,dataset))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    if 'cifar' in dataset:
        modelPath="best_{}_{}.m".format(modelName,dataset)
        modelPath=os.path.join('.','models',dataset,modelPath)
        model = torch.load(modelPath,weights_only=False)
    else:
        model=ResNetUtils.GetTrainedModel(modelName=modelName,dataset=dataset)
    model = model.to(device) 
    model.eval()
    layerNum=0
    for name, module in model.named_modules():
        print(layerNum,name, type(module))
        module.mydata={
            'name':name,
            'layerId':layerNum
        }
        module.register_forward_hook(fault_hook)
        layerNum+=1
    # generic image preperation
    batch_size = 1
    h = 32
    w = 32
    c = 3

    image = torch.rand((batch_size, c, h, w))
    image = image.to(device)
    model(image)
    pass

if __name__=='__main__':
    a=[]
    a.append(1)
    a.append(2)
    print(a)
    exit(0)
    modelNames=[
        # ResNetConfig.ResNet18,
        # ResNetConfig.ResNet34,
        # ResNetConfig.ResNet50,
        ResNetConfig.ResNet101,
        # ResNetConfig.ResNet152
    ]
    main(modelName=modelNames[0],dataset=ResNetConfig.DATASET)  