import os
import json
import ResNetConfig
import numpy as np
import pandas as pd
import torch
from FI import FaultInjectionConfig

def SoftMaxDetector(data):
    x = np.array(data)
    # print(x)
    # exit(0)
    output = np.exp(x) / np.sum(np.exp(x))
    return output 

def main(goldenResultDirs,faultResultDirs=[]):
    if len(faultResultDirs)!=0:
        fns=os.listdir(faultResultDirs[0])
    else:
        fns=os.listdir(goldenResultDirs[0])
    totalNum=0
    correctNum=0
    ######################################
    lessProbTh=[0.8,0.85,0.9,0.95]
    lessProbNum=[1,1,1,1]
    lessProbTpNum=[0,0,0,0]
    nanNum=0
    infNum=0
    absNum=0
    for fn in fns:
        totalNum+=1
        dataJosnList=[]
        for resultDir in goldenResultDirs:
            fp=os.path.join(resultDir,fn)
            with open(fp,'r') as rf:
                dataJson=json.load(rf)
            dataJosnList.append(dataJson)
        for faultResultDir in faultResultDirs:
            fp=os.path.join(faultResultDir,fn)
            with open(fp,'r') as rf:
                dataJson=json.load(rf)
            dataJosnList.append(dataJson)          

        #平均加权
        outputSum=None#初始值
        modelSize=0
        for id,dataJson in enumerate(dataJosnList):
            label=dataJson['label']
            output=dataJson['output']
            # output=np.array(output)
            softOutput=SoftMaxDetector(data=output)
            nanFlag=np.isnan(softOutput).any()
            infFlag=np.isinf(softOutput).any()
            absFlag=(np.abs(softOutput)>100).any()
            print(np.abs(softOutput))
            if nanFlag:
                nanNum=nanNum+1 
            if infFlag:
                infNum=infNum+1 
            if absFlag:
                absNum=absNum+1
            if (nanFlag or infFlag or absFlag):
                continue
            if outputSum is None:
                outputSum=softOutput
            else:
                outputSum+=softOutput
            modelSize+=1
        outputSum=outputSum/modelSize
        predId=np.argmax(outputSum)
        if label==predId:
            correctNum+=1
        ###统阈值影响
        outMax=np.max(outputSum)
        for i,th in enumerate(lessProbTh):
            if outMax<th:
                continue
            lessProbNum[i]+=1
            if predId==label:
                lessProbTpNum[i]+=1
    print("Nan Rate:{:.5f},Inf Rate:{:.5f},absNum Rate:{:.5f},totalNum:{:},nanNum:{},infNum:{},absNum:{}".format(nanNum/totalNum*100,infNum/totalNum*100,absNum/totalNum*100,totalNum,nanNum,infNum,absNum))
    for i in range(len(lessProbTh)):
        print('Th:{},conf:{:.4f},coverage:{:.4f}'.format(lessProbTh[i],lessProbTpNum[i]/lessProbNum[i]*100,lessProbNum[i]/totalNum*100))    
    Acc=correctNum/totalNum*100
    return Acc

if __name__=='__main__':
    print("Average!")
    modelNames=[
        ResNetConfig.ResNet18,
        ResNetConfig.ResNet34,
        ResNetConfig.ResNet50,
        # ResNetConfig.ResNet101,
    ]
    faultMoldeNames=[
        ResNetConfig.ResNet18,
        ResNetConfig.ResNet34,
        ResNetConfig.ResNet50,
        # ResNetConfig.ResNet101,        
    ]
    dataset=ResNetConfig.DATASET_CIFAR_10
    BerRates=[1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]
    BerRates=[1e-1]
    injTimes=3000
    dataType=torch.float32
    if 0:
        resultDir=os.path.join('golden',dataset,modelNames[0])
        fns=os.listdir(resultDir)
        goldenResultDirs=[]
        for modelName in modelNames:
            resultDir=os.path.join('golden',ResNetConfig.DATASET,modelName)
            goldenResultDirs.append(resultDir)
        Acc=main(goldenResultDirs=goldenResultDirs)
        print("{} Acc:{:.2f}".format(modelNames,Acc))
    else:
        for faultMoldeName in faultMoldeNames:
            resultDir=os.path.join('golden',dataset,modelNames[0])
            fns=os.listdir(resultDir)
            goldenResultDirs=[]
            goldenModels=[]
            for modelName in modelNames:
                if modelName!=faultMoldeName:
                    goldenModels.append(modelName)
                    resultDir=os.path.join('golden',ResNetConfig.DATASET,modelName)
                    goldenResultDirs.append(resultDir)

            for berId,ber in enumerate(BerRates):
                print("="*20+"GoldenModelName:{},faultMoldeName:{},BitErrorRate:{}".format(goldenModels,faultMoldeName,ber)+"="*20)
                dataTypeName=FaultInjectionConfig.GetDTypeName(dataType=dataType)
                dirName=os.path.join('neuron_{}_{}_{}_{}_{}'.format(dataset,dataTypeName,faultMoldeName,str(ber),str(injTimes)))
                layersDir=os.path.join('out',dirName)
                for dirId,dirName in enumerate(os.listdir(layersDir)):
                    faultDir=os.path.join(layersDir,dirName)
                    Acc=main(goldenResultDirs=goldenResultDirs,faultResultDirs=[faultDir])
                    print("GoldenModelName:{}, faultMoldeName:{}, Acc:{:.2f}".format(goldenModels,faultMoldeName,Acc))
