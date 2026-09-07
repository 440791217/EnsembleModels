import os
import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import ImageClassification.ResNetConfig as ResNetConfig
import Datasets.CiFar as CiFar
import Datasets.Imagenet as Imagenet
import json
import ResNetUtils




def main(modelName,dataset):
    print(modelName)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    train_loader, test_loader=ResNetUtils.GetLoader(dataset=dataset)
    
    goldenDir=os.path.join('golden',dataset,modelName)
    if not os.path.exists(goldenDir):
        os.makedirs(goldenDir)
   

    if 'cifar' in dataset:
        modelPath="best_{}_{}.m".format(modelName,dataset)
        modelPath=os.path.join('.','models',dataset,modelPath)
        model = torch.load(modelPath,weights_only=False)
    else:
        model=ResNetUtils.GetTrainedModel(modelName=modelName,dataset=dataset)
    model = model.to(device) 
    model.eval()
    id=0
    for images, labels in test_loader:
        id+=1
        fname='{}.json'.format(str(id).zfill(6))
        fp1=os.path.join(goldenDir,'data.json')
        fp2=os.path.join(goldenDir,fname)
        if os.path.exists(fp2):
            continue

        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        output=outputs.cpu().tolist()[0]
        labels=labels.cpu().tolist()
        label=labels[0]
        # print(len(output))
        # print(id,output.index(max(output)),label)
        # if id==100:
        #     exit(1)
        result={
            'label':label,
            'output':output,
        }

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
        # ResNetConfig.ResNet34,
        # ResNetConfig.ResNet50,
        # ResNetConfig.ResNet101,
        # ResNetConfig.ResNet152
    ]
    for modelName in modelNames:
        main(modelName=modelName,dataset=ResNetConfig.DATASET)   