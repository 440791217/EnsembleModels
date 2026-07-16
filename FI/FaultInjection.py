import os
import json
import ImageClassification.ResNetConfig as ResNetConfig
import ImageClassification.ResNetUtils as ResNetUtils
import numpy as np
import torch
import copy
import FaultInjectionConfig
import FI.BitFlipTool as BitFlipTool
import Logger.Logger as Logger

HOOK_TYPE_STAT=1
HOOK_TYPE_FAULT=2



def InjectFaultsForNeurons(module, input):
    mydata=module.mydata
    dataType=mydata['dataType']
    ber=mydata['ber']
    if isinstance(input, tuple):
        x = input[0]
    else:
        Logger.Error("Input is not tuple.")
    # print("Layer In Hook==>Id:{},name:{}".format(mydata['layerId'],mydata['layerName']))
    #convert data format
    origDType=x.dtype
    x=x.to(dataType)
    #inject faults
    x_faults=BitFlipTool.Bitflip(x,ber)
    #recover data format
    x_faults=x_faults.to(origDType)
    # print("shape:{}".format(x_faults.shape))
    # return (input,)
    return (x_faults,)


def LoadModel(modelName,dataset):
    print("ModelName {}, DataSet:{}".format(modelName,dataset))
    if 'cifar' in dataset:
        modelPath="best_{}_{}.m".format(modelName,dataset)
        modelPath=os.path.join(ResNetConfig.MODEL_DIR_PATH,modelPath)
        model = torch.load(modelPath,weights_only=False)
    else:
        model=ResNetUtils.GetTrainedModel(modelName=modelName,dataset=dataset)
    model = model.to(FaultInjectionConfig.Device()) 
    model.eval()
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


def MkDataDir(modelName,dataset,injTimes,ber,dataType):
    dataTypeName=FaultInjectionConfig.GetDTypeName(dataType=dataType)
    dirName=os.path.join('neuron_{}_{}_{}_{}_{}'.format(dataset,dataTypeName,modelName,str(ber),str(injTimes)))
    # dirName="{}_{}_{}_{}_{}".format(modelName,dataset,injTimes,bfm,ber)
    resDirPath=os.path.join('out',dirName)
    if not os.path.exists(resDirPath):
        os.makedirs(resDirPath)
    return resDirPath

def InjectNeuronFaultsForLayers(model,mydata,dataLoader):
    print('InjectNeuronFaultsForLayers')
    faultModel=copy.deepcopy(model)
    handles=[]
    for layerName, layer in faultModel.named_modules():
        if mydata['name']==layerName:
            layer.mydata=mydata
            handle=layer.register_forward_pre_hook(InjectFaultsForNeurons)
            handles.append(handle)
    
    injTimes=mydata['injTimes']
    resDir=mydata['resDir']
    device=mydata['device']
    layerId=mydata['id']
    for i, loader in enumerate(dataLoader):
        injId=i+1
        if injId > injTimes:
            break
        print("="*20+"{}".format(injId)+"="*20)
        #创建文件
        fname='{}.json'.format(str(injId).zfill(6))
        fp1=os.path.join(resDir,'data.json')
        fp2=os.path.join(resDir,fname)
        if os.path.exists(fp2):
            continue
        #读取数据
        groundId=loader[1].item()#真实值
        image=loader[0]#原始数据
        image=image.to(device)
        outputs=faultModel(image)
        output=outputs.detach().cpu().numpy().tolist()[0]
        print("outputs:{}".format(output))
        predId=output.index(max(output))
        print('groudId:{},predId:{}'.format(groundId,predId))
        print("LayerId:{},groudId:{},predId:{}".format(layerId,groundId,predId))
        result={
            'label':groundId,
            'predId':predId,
            'output':output,
        }
        with open(fp1,'w') as wf:
            json.dump(result,wf,indent=2)
        if os.path.exists(fp2):
            os.remove(fp2)
        os.rename(fp1,fp2)
    #清理工作
    for handle in handles:
        handle.remove()
    handles.clear()
    del faultModel




@torch.no_grad()
def main(modelName,dataset,injTimes,dataLoader,\
         ber=FaultInjectionConfig.GetBitErrorRate(),\
            device=FaultInjectionConfig.Device(),\
                dataType=FaultInjectionConfig.GetDType()):
    #新建结果目录
    outDir=MkDataDir(modelName=modelName,dataset=dataset,injTimes=injTimes,ber=ber,dataType=dataType)
    #载入模型
    model=LoadModel(modelName=modelName,dataset=dataset)

    #剖析模型
    layerInfoList=ProfileModelHook(model=model)
    # print(layerInfoList)
    #遍历执行故障注入
    for layerInfo in layerInfoList:
        #只对选择的网络层注入故障
        supportLayerType=FaultInjectionConfig.IsFaultLayer(className=layerInfo['className'])
        if supportLayerType==None:
            continue
        resDir=os.path.join(outDir,\
                str(layerInfo['id']).zfill(4)+supportLayerType)
        if not os.path.exists(resDir):
            os.makedirs(resDir)
        mydata={
            'id':layerInfo['id'],
            'name':layerInfo['name'],
            'className':layerInfo['className'],
            'injTimes':injTimes,
            'ber':ber,
            'device':device,
            'dataType':dataType,
            'resDir':resDir,
        }
        InjectNeuronFaultsForLayers(model=model,mydata=mydata,dataLoader=dataLoader)


    pass

if __name__=='__main__':
    modelNames=[
        # ResNetConfig.ResNet18,
        ResNetConfig.ResNet34,
        ResNetConfig.ResNet50,
        ResNetConfig.ResNet101,
        # ResNetConfig.ResNet152
    ]
    BerRates=[1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]
    dataset=ResNetConfig.DATASET_CIFAR_10
    injTimes=3000
    #载入数据集
    train_loader, test_loader=ResNetUtils.GetLoader(dataset=dataset)
    for modelName in modelNames:
        for ber in BerRates:
            main(modelName=modelName,dataset=dataset,injTimes=injTimes,ber=ber,\
                dataLoader=test_loader)  
    pass