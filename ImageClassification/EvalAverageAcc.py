import os
import json
import ResNetConfig
import numpy as np

def main(modelNames):
    resultDir=os.path.join('golden',ResNetConfig.DATASET,modelNames[0])
    fns=os.listdir(resultDir)
    totalNum=0
    correctNum=0
    for fn in fns:
        dataJosnList=[]
        for modelName in modelNames:
            fp=os.path.join('golden',ResNetConfig.DATASET,modelName,fn)
            with open(fp,'r') as rf:
                dataJson=json.load(rf)
            dataJosnList.append(dataJson)

        for id,dataJson in enumerate(dataJosnList):
            label=dataJson['label']
            output=dataJson['output']
            output=np.array(output)
            if id==0:
                outputSum=output
            else:
                outputSum+=output
        maxIndex=np.argmax(outputSum)
        if label==maxIndex:
            correctNum+=1
        totalNum+=1       
    Acc=correctNum/totalNum*100
    print("{} Acc:{:.2f}".format(modelName,Acc))
    pass

if __name__=='__main__':
    print("Average!")
    modelNames=[
        ResNetConfig.ResNet18,
        ResNetConfig.ResNet34,
        ResNetConfig.ResNet50,
        ResNetConfig.ResNet101,
        # ResNetConfig.ResNet152
    ]
    main(modelNames=modelNames)  