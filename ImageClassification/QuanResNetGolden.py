import os
import copy
import torch
import torchvision
import torch.ao.quantization as quantization
from torch.ao.quantization import quantize_fx
import io
import ImageClassification.ResNetConfig as ResNetConfig
import ResNetUtils
import json
import numpy as np
import QuantResNet
def evaluate_accuracy(model, data_loader, device="cpu"):
    model.eval()
    model.to(torch.device(device))
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(torch.device("cpu"), dtype=torch.float32)
            targets = targets.to(torch.device("cpu"))
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return 100.0 * correct / total

if __name__ == '__main__':
    modelName = ResNetConfig.ResNet101
    dataset = ResNetConfig.DATASET_CIFAR_10

    goldenDir=os.path.join('golden',dataset,modelName+'_q8')
    if not os.path.exists(goldenDir):
        os.makedirs(goldenDir)
    
    print("===> 1. 正在加载数据加载器...")
    train_loader, test_loader = ResNetUtils.GetLoader(dataset=dataset)
    quantized_model=QuantResNet.convert_model(modelName=modelName,dataset=dataset,train_loader=train_loader,test_loader=test_loader)


    # 4. 评估 INT8 量化模型的准确率
    print("===> 6. 评估 INT8 量化模型准确率...")
    correct = 0
    id=0
    with torch.no_grad():
        for images, labels in test_loader:
            id+=1
            fname='{}.json'.format(str(id).zfill(6))
            fp1=os.path.join(goldenDir,'data.json')
            fp2=os.path.join(goldenDir,fname)
            if os.path.exists(fp2):
                continue
            images = images.to("cpu")
            labels = labels.to("cpu")
            outputs = quantized_model(images)
            output=outputs.cpu().tolist()[0]
            labels=labels.cpu().tolist()
            label=labels[0]
            result={
                'label':label,
                'output':output,
            }
            if label==np.argmax(np.array(output)):
                correct+=1
            with open(fp1,'w') as wf:
                json.dump(result,wf,indent=2)
            os.rename(fp1,fp2)
    int8_acc = 100.0 * correct / id
    print(f"INT8 量化模型准确率: {int8_acc:.2f}%")