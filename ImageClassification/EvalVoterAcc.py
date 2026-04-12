import os
import json
import ResNetConfig

def main(modelNames):
    resultDir=os.path.join('golden',ResNetConfig.DATASET,modelNames[0])
    fns=os.listdir(resultDir)
    totalNum=0
    correctNum=0
    matchNums=[0]*10
    for fn in fns:
        matchNum=0
        dataJosnList=[]
        for modelName in modelNames:
            fp=os.path.join('golden',ResNetConfig.DATASET,modelName,fn)
            with open(fp,'r') as rf:
                dataJson=json.load(rf)
            dataJosnList.append(dataJson)
        maxIndexList=[0]*1000
        for dataJson in dataJosnList:
            label=dataJson['label']
            output=dataJson['output']
            maxOutPut=max(output)
            # if maxOutPut<0.99:
            # continue
            maxIndex=output.index(maxOutPut)
            maxIndexList[maxIndex]+=1
            if maxIndex==label:
                matchNum+=1
        matchNums[matchNum]+=1
        maxIndex=maxIndexList.index(max(maxIndexList))
        if label==maxIndex:
            correctNum+=1
        totalNum+=1       
    Acc=correctNum/totalNum*100
    print("{} Acc:{:.2f}".format(modelName,Acc))
    print(matchNums,"-------",totalNum)
    pass

if __name__=='__main__':
    print("Voter!")
    modelNames=[
        ResNetConfig.ResNet18,
        ResNetConfig.ResNet34,
        ResNetConfig.ResNet50,
        ResNetConfig.ResNet101,
        # ResNetConfig.ResNet152
    ]
    main(modelNames=modelNames)  