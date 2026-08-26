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
import FaultInjectionUtils

HOOK_TYPE_STAT=1
HOOK_TYPE_FAULT=2



# def InjectFaultsForNeurons(module, input):
#     mydata=module.mydata
#     # dataType=mydata['dataType']
#     ber=mydata['ber']
#     if isinstance(input, tuple):
#         x = input[0]
#     else:
#         Logger.Error("Input is not tuple.")
#         exit(-1)
#     # print("Layer In Hook==>Id:{},name:{}".format(mydata['layerId'],mydata['layerName']))
#     #convert data format
#     origDType=x.dtype
#     print('origDType:',origDType)
#     # exit(-1)
#     # x=x.to(dataType)
#     #inject faults
#     x_faults=BitFlipTool.Bitflip(x,ber)
#     #recover data format
#     # x_faults=x_faults.to(origDType)
#     # print("shape:{}".format(x_faults.shape))
#     # return (input,)
#     return (x_faults,)


def InjectFaultsForNeurons(module, input):
    mydata = module.mydata
    ber = mydata['ber']
    
    if isinstance(input, tuple):
        x = input[0]
    else:
        Logger.Error("Input is not tuple.")
        return input

    # 检查是否为量化张量
    if x.is_quantized:
        scale = x.q_scale()
        zero_point = x.q_zero_point()
        device = x.device
        
        # 【关键修复 1】：必须先反量化，剥离 QuantizedCPU 后端，转为普通 FP32 张量
        float_tensor = x.dequantize()
        
        # 【关键修复 2】：现在可以安全地送入你的 Bitflip 工具中进行底层二进制翻转了
        # Bitflip 内部会对普通的 float/int 张量转为 uint8 字节流并执行 XOR
        x_faults = BitFlipTool.Bitflip(float_tensor, ber)
        
        # 确保 scale 和 zero_point 转换为 Tensor 且设备一致
        if not isinstance(scale, torch.Tensor):
            scale_tensor = torch.tensor(scale, dtype=torch.float32, device=device)
        else:
            scale_tensor = scale.to(device)
            
        if not isinstance(zero_point, torch.Tensor):
            zp_tensor = torch.tensor(zero_point, dtype=torch.int64, device=device)
        else:
            zp_tensor = zero_point.to(device)
            
        # 【关键修复 3】：重新量化打包回合法量化张量，供后续神经网络层使用
        x_faults_quantized = torch.quantize_per_tensor(
            x_faults, scale_tensor, zp_tensor, x.dtype
        )
        return (x_faults_quantized,) + input[1:]
    
    else:
        # 非量化模型的原有逻辑（保持不变）
        origDType = x.dtype
        dataType = mydata['dataType']
        x = x.to(dataType)
        x_faults = BitFlipTool.Bitflip(x, ber)
        x_faults = x_faults.to(origDType)
        return (x_faults,) + input[1:]


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
         ber=None,device=None,dataType=None,isQuant=None):
    #新建结果目录
    outDir=MkDataDir(modelName=modelName,dataset=dataset,injTimes=injTimes,ber=ber,dataType=dataType)
    #载入模型
    if isQuant:
        model=FaultInjectionUtils.LoadQ8Model(modelName=modelName,dataset=dataset)
    else:
        model=FaultInjectionUtils.LoadModel(modelName=modelName,dataset=dataset)

    #剖析模型
    layerInfoList=FaultInjectionUtils.ProfileModelHook(model=model)
    # exit(-1)
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
    isQuant=True
    modelNames=[
        # ResNetConfig.ResNet18,
        # ResNetConfig.ResNet34,
        # ResNetConfig.ResNet50,
        ResNetConfig.ResNet101,
        # ResNetConfig.ResNet152
    ]
    BerRates=[1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]
    # BerRates=[1e-1]
    dataset=ResNetConfig.DATASET_CIFAR_10
    injTimes=3000
    if isQuant:
        device=torch.device('cpu')
        dataType=torch.quint8
    else:
        device=torch.device('cuda')
        dataType=torch.float32
    #载入数据集
    train_loader, test_loader=ResNetUtils.GetLoader(dataset=dataset)
    for modelName in modelNames:
        for ber in BerRates:
            main(modelName=modelName,dataset=dataset,injTimes=injTimes,ber=ber,\
                dataLoader=test_loader,device=device,dataType=dataType,isQuant=isQuant)  
    pass