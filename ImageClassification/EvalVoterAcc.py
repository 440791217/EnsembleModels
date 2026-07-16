import os
import json
import ResNetConfig
import torch
import pandas as pd
from FI import FaultInjectionConfig
import numpy as np

# def SoftMaxDetector(data):
#     x = np.array(data)
#     output = np.exp(x) / np.sum(np.exp(x))
#     return output 


def SoftMaxDetector(data):
    # 1. 转换为 numpy 数组
    x = np.array(data)

    # 2. 【核心修改】找出当前这组 Logits 中的最大值
    # 在你的数据里，max_val 就是那个 1.749e+34
    max_val = np.max(x)

    # 3. 【核心修改】全员减去最大值（数学平移）
    # 这样最大的数变成了 0，其余的数都变成了极其恐怖的负数（例如 -3e+34）
    x_stable = x - max_val
    
    # 4. 安全地计算指数
    # e^0 = 1.0（绝对不会溢出），e^(巨大负数) = 0.0（安全下溢）
    exp_x = np.exp(x_stable)
    
    # 5. 计算分母的总和
    sum_exp = np.sum(exp_x)
    
    # 6. 健壮性保护：防止分母极小导致除零异常
    if sum_exp == 0:
        return np.zeros_like(x)
        
    # 7. 计算最终的概率分布
    output = exp_x / sum_exp
    return output

def main(goldenResultDirs,faultResultDirs=[]):
    if len(faultResultDirs)!=0:
        fns=os.listdir(faultResultDirs[0])
    else:
        fns=os.listdir(goldenResultDirs[0])
    totalNum=0
    correctNum=0
    totalNum1=0
    correctNum1=0
    matchNums=[0]*10
    ######################################
    for fn in fns:
        matchNum=0
        dataJosnList=[]
        probT=0
        predId=-1
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
        maxIndexList=[0]*1001
        maxProbList=[0]*1001
        for dataJson in dataJosnList:
            label=dataJson['label']
            output=dataJson['output']
            print('1:',output)
            output=SoftMaxDetector(data=output)
            print('2:',output)
            maxOutPut=max(output)
            maxIndex=output.tolist().index(maxOutPut)
            maxIndexList[maxIndex]+=1
            if maxIndex==label:
                matchNum+=1
            #########
            if maxOutPut>0 and probT<maxOutPut:
                probT=maxOutPut
                predId=maxIndex
        matchNums[matchNum]+=1 #匹配趋势统计
        maxIndex=maxIndexList.index(max(maxIndexList))
        #传统的
        if label==maxIndex:
            correctNum+=1
        totalNum+=1
        if max(maxIndexList)==3:##投票选择性分类
            totalNum1+=1
            if label==maxIndex:
                correctNum1+=1
    # print("{:.2f},{:.2f}".format(correctNum1/totalNum1*100,totalNum1/totalNum*100))       
    Acc=correctNum/totalNum*100
    ACC1=correctNum1/totalNum1*100
    Coverage1=totalNum1/totalNum*100
    print('Coverage1:{:.2f},ACC1:{:.2f}'.format(Coverage1,ACC1))
    return Acc,totalNum,matchNums

if __name__=='__main__':
    print("Voter!")
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
    injTimes=3000
    dataType=torch.float32
    if 1:
        with pd.ExcelWriter("result.xlsx") as writer:
            for faultMoldeName in faultMoldeNames:
                top1AccListByBer=[]
                resultDir=os.path.join('golden',dataset,modelNames[0])
                goldenResultDirs=[]
                goldenModels=[]
                for modelName in modelNames:
                    if modelName!=faultMoldeName:
                        goldenModels.append(modelName)
                        resultDir=os.path.join('golden',ResNetConfig.DATASET,modelName)
                        goldenResultDirs.append(resultDir)
                for berId,ber in enumerate(BerRates):
                    top1AccListByBer.append([ber])
                for berId,ber in enumerate(BerRates):
                    print("="*20+"GoldenModelName:{},faultMoldeName:{},BitErrorRate:{}".format(goldenModels,faultMoldeName,ber)+"="*20)
                    dataTypeName=FaultInjectionConfig.GetDTypeName(dataType=dataType)
                    dirName=os.path.join('neuron_{}_{}_{}_{}_{}'.format(dataset,dataTypeName,faultMoldeName,str(ber),str(injTimes)))
                    layersDir=os.path.join('out',dirName)
                    for dirId,dirName in enumerate(os.listdir(layersDir)):
                        faultDir=os.path.join(layersDir,dirName)
                        Acc,totalNum,matchNums=main(goldenResultDirs=goldenResultDirs,faultResultDirs=[faultDir])
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
        pass
    else:
        goldenResultDirs=[]
        goldenModels=[]
        for modelName in modelNames:
            goldenModels.append(modelName)
            resultDir=os.path.join('golden',ResNetConfig.DATASET,modelName)
            goldenResultDirs.append(resultDir)
        Acc,totalNum,matchNums=main(goldenResultDirs=goldenResultDirs,faultResultDirs=[])
        print("{} Acc:{:.2f}".format(modelNames,Acc))
        print(matchNums,"-------",totalNum)