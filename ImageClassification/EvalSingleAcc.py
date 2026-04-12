import os
import json
import ResNetConfig

def main(modelName):
    resultDir=os.path.join('golden',ResNetConfig.DATASET,modelName)
    totalNum=0
    correctNum=0
    correctNum1=0
    for fn in os.listdir(resultDir):
        fp=os.path.join(resultDir,fn)
        with open(fp,'r') as rf:
            dataJson=json.load(rf)
        label=dataJson['label']
        output=dataJson['output']
        top5Output=sorted(output,reverse=True)[0:5]
        top5Index=[0]*5
        for i,out in enumerate(top5Output):
            top5Index[i]=output.index(out)
        maxIndex=output.index(max(output))
        if label==maxIndex:
            correctNum+=1
        if label in top5Index:
            correctNum1+=1
        totalNum+=1
    Acc=correctNum/totalNum*100
    Acc1=correctNum1/totalNum*100
    print("TOP1:{} Acc:{:.2f}".format(modelName,Acc))
    print("TOP5:{} Acc:{:.2f}".format(modelName,Acc1))
    pass

if __name__=='__main__':
    modelNames=[
        ResNetConfig.ResNet18,
        # ResNetConfig.ResNet34,
        # ResNetConfig.ResNet50,
        # ResNetConfig.ResNet101,
        # ResNetConfig.ResNet152
    ]
    for modelName in modelNames:
        main(modelName=modelName)  