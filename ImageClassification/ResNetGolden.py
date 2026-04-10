import os
import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import ImageClassification.ResNetConfig as ResNetConfig
import Datasets.CiFar as CiFar
import json


def main(modelName):
    print(modelName)

    if ResNetConfig.DATASET == ResNetConfig.DATASET_CIFAR_10:
        train_loader, test_loader = CiFar.GetCifar_10(batch_size=1)
    elif ResNetConfig.DATASET == ResNetConfig.DATASET_CIFAR_100:
        train_loader, test_loader = CiFar.GetCifar_100(batch_size=1)
    elif ResNetConfig.DATASET == ResNetConfig.DATASET_CIFAR_100_COARSE:
        train_loader, test_loader = CiFar.GetCifar_100_Coarse(batch_size=1)
    else:
        raise ValueError("Invalid dataset")
    
    goldenDir=os.path.join('golden',ResNetConfig.DATASET,modelName)
    if not os.path.exists(goldenDir):
        os.makedirs(goldenDir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    


    modelPath="best_{}_{}.m".format(modelName,ResNetConfig.DATASET)
    modelPath=os.path.join(ResNetConfig.MODEL_DIR_PATH,modelPath)
    model = torch.load(modelPath,weights_only=False)
    model.eval()
    id=0
    for images, labels in test_loader:
        id+=1
        images = images.to(device)
        labels = labels.to(device)
        output = model(images).cpu().tolist()[0]
        label=labels.cpu().tolist()[0]
        # print(output.index(max(output)),label)
        result={
            'label':label,
            'output':output,
        }
        fname='{}.json'.format(str(id).zfill(6))
        fp1=os.path.join(goldenDir,'data.json')
        fp2=os.path.join(goldenDir,fname)
        if os.path.exists(fp2):
            continue
        with open(fp1,'w') as wf:
            json.dump(result,wf,indent=2)
        os.rename(fp1,fp2)
        # exit(-1)
        # i+=1
        # if i==3:
        #     exit(1)

if __name__=='__main__':
    modelNames=[
        ResNetConfig.ResNet18,
        ResNetConfig.ResNet34,
        ResNetConfig.ResNet50,
        ResNetConfig.ResNet101,
        # ResNetConfig.ResNet152
    ]
    for modelName in modelNames:
        # ResNetConfig.MODEL_NAME=modelName
        main(modelName=modelName)   