import os
import json
import ResNetConfig
import numpy as np
import pandas as pd
import torch
from FI import FaultInjectionConfig


def MergeResults(goldenResultDirs,faultResultDirs,fn):
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
    return dataJosnList

def PredByAvg(results):
    for id,dataJson in enumerate(results):
        label=dataJson['label']
        output=dataJson['output']
        output=np.array(output)
        if id==0:
            outputSum=output
        else:
            outputSum+=output
    predId=np.argmax(outputSum)
    return predId,label

def PredByVoter(results):
    maxIndexList=[0]*1000
    # maxProbList=[0]*1000
    for dataJson in results:
        label=dataJson['label']
        output=dataJson['output']
        maxOutPut=max(output)
        # if maxOutPut<0.99:
        # continue
        maxIndex=output.index(maxOutPut)
        maxIndexList[maxIndex]+=1
    maxIndexValue=max(maxIndexList)
    if maxIndexValue>1:
        predId=maxIndexList.index(maxIndexValue)
    else:
        predId=-1
    return predId,label

def PredByMixed(results):
    pass

def main(goldenResultDirs,ensembleType,faultResultDirs=[]):
    if len(faultResultDirs)!=0:
        fns=os.listdir(faultResultDirs[0])
    else:
        fns=os.listdir(goldenResultDirs[0])
    tp=0
    fp=0
    totalNum=0
    for fn in fns:
        dataJosnList=MergeResults(goldenResultDirs=goldenResultDirs,faultResultDirs=faultResultDirs,fn=fn)
        if ensembleType==1:     
            predId,label=PredByAvg(results=dataJosnList)
        elif ensembleType==2:
            predId,label=PredByVoter(results=dataJosnList)
        elif ensembleType==3:
            predId,label=PredByAvg(results=dataJosnList)
            predId1,label=PredByVoter(results=dataJosnList)
            if predId==predId1:
                predId=predId1
            else:
                predId=-1
        else:
            exit(-222)
        if label==predId:
            tp+=1
        elif predId!=-1 and predId!=label:
            fp+=1
        if 1:
            totalNum+=1
        elif predId!=-1:
            totalNum+=1
    print('tp:{},fp:{},totalNum:{}'.format(tp,fp,totalNum)) 
    Acc=tp/(totalNum)*100
    return Acc

def EvalGoldenModels(modelNames,dataset,ensembleType):
    goldenResultDirs=[]
    for modelName in modelNames:
        resultDir=os.path.join('golden',dataset,modelName)
        goldenResultDirs.append(resultDir)
    Acc=main(goldenResultDirs=goldenResultDirs,ensembleType=ensembleType)
    print("{} Acc:{:.2f}".format(modelNames,Acc))


def GetGoldenModels(modelNames,faultMoldeName,dataset):
    goldenModels=[]
    goldenResultDirs=[]
    for modelName in modelNames:
        if modelName==faultMoldeName:
            continue
        goldenModels.append(modelName)
        resultDir=os.path.join('golden',dataset,modelName)
        goldenResultDirs.append(resultDir)
    return goldenModels,goldenResultDirs 

def EvalFaultModels(modelNames,faultMoldeNames,dataset,bers,injTimes,dataType,ensembleType):
    with pd.ExcelWriter("result.xlsx") as writer:
        for faultMoldeName in faultMoldeNames:
            top1AccListByBer=[]
            goldenModels,goldenResultDirs = GetGoldenModels(modelNames=modelNames,faultMoldeName=faultMoldeName,dataset=dataset)
            for berId,ber in enumerate(bers):
                top1AccListByBer.append([ber])
            for berId,ber in enumerate(bers):
                print("="*20+"GoldenModelName:{},faultMoldeName:{},BitErrorRate:{}".format(goldenModels,faultMoldeName,ber)+"="*20)
                dataTypeName=FaultInjectionConfig.GetDTypeName(dataType=dataType)
                dirName=os.path.join('neuron_{}_{}_{}_{}_{}'.format(dataset,dataTypeName,faultMoldeName,str(ber),str(injTimes)))
                layersDir=os.path.join('out',dirName)
                for dirId,dirName in enumerate(os.listdir(layersDir)):
                    faultDir=os.path.join(layersDir,dirName)
                    Acc=main(goldenResultDirs=goldenResultDirs,faultResultDirs=[faultDir],ensembleType=ensembleType)
                    print("GoldenModelName:{}, faultMoldeName:{}, Acc:{:.2f}".format(goldenModels,faultMoldeName,Acc))
                    top1AccListByBer[berId].append(Acc)
            data=[]
            columns=['ber']
            data.append(columns)
            for dirId,dirName in enumerate(os.listdir(layersDir)):
                columns.append(dirName)
            for top1Accs in top1AccListByBer:
                data.append(top1Accs)
            data.append(goldenModels)
            data.append([faultMoldeName])
            df = pd.DataFrame(
                data,
            )
            df.to_excel(
                writer,
                sheet_name='{}_{}_avg'.format(dataset,faultMoldeName),
                index=False
            )                

if __name__=='__main__':
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
    bers=[1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]
    injTimes=3000
    dataType=torch.float32
    ensembleType=3#1=>AVG,2=>VOTER,3=>Mixed
    hint={
        1:'AVG',
        2:'VOTER',
        3:'MIXED'
    }
    print(hint[ensembleType])
    if 0:
        EvalFaultModels(modelNames=modelNames,faultMoldeNames=faultMoldeNames,dataset=dataset,\
                        bers=bers,injTimes=injTimes,dataType=dataType,ensembleType=ensembleType)
    else:
        EvalGoldenModels(modelNames=modelNames,dataset=dataset,ensembleType=ensembleType)