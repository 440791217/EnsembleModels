import os
import copy
import torch
import torchvision
import torch.ao.quantization as quantization
from torch.ao.quantization import quantize_fx
import io
import ImageClassification.ResNetConfig as ResNetConfig
import ResNetUtils

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
    modelName = ResNetConfig.ResNet18
    dataset = ResNetConfig.DATASET_CIFAR_10
    
    print("===> 1. 正在加载数据加载器...")
    train_loader, test_loader = ResNetUtils.GetLoader(dataset=dataset)
    
    # 2. 加载你本地训练好的 CIFAR-10 完整模型
    if 'cifar' in dataset:
        modelPath = "best_{}_{}.m".format(modelName, dataset)
        modelPath = os.path.join(ResNetConfig.MODEL_DIR_PATH, modelPath)
        print(f"===> 正在加载本地模型: {modelPath}")
        
        # 使用内存流加载，彻底洗掉 GPU 上下文痕迹
        buffer = io.BytesIO()
        with open(modelPath, "rb") as f:
            buffer.write(f.read())
        buffer.seek(0)
        model = torch.load(buffer, map_location="cpu", weights_only=False)
    else:
        model = ResNetUtils.GetTrainedModel(modelName=modelName, dataset=dataset)
        
    model.eval()
    model.to("cpu")  

    # print("===> 2. 评估原始 FP32 模型准确率...")
    # fp32_acc = evaluate_accuracy(model, test_loader, device="cpu")
    # print(f"FP32 模型准确率: {fp32_acc:.2f}%")

    # ==========================================
    # 3. 使用 PyTorch FX Graph Mode 进行静态 PTQ 量化
    # ==========================================
    print("===> 3. 正在准备 FX 模式量化配置...")
    
    # 复制一份模型用于量化
    model_fp32 = model

    # 定义 fbgemm 后端的量化配置字典
    qconfig_dict = {"": quantization.get_default_qconfig('fbgemm')}

    # 步骤 A：准备量化（FX 模式会自动追踪图结构并插入 Observer）
    # 注意：ResNet 的输入示例通常为 (1, 3, 32, 32)
    example_inputs = (next(iter(test_loader))[0][:1].to("cpu"),)
    prepared_model = quantize_fx.prepare_fx(model_fp32, qconfig_dict, example_inputs=example_inputs)

    # 步骤 B：校准 (Calibration) —— 使用部分测试数据收集量化统计信息
    print("===> 4. 正在进行校准 (Calibration)...")
    prepared_model.eval()
    with torch.no_grad():
        batch_count = 0
        for images, _ in test_loader:
            images = images.detach().cpu().to(torch.float32)
            prepared_model(images)
            batch_count += 1
            if batch_count >= 10:  # 取前 10 个 batch 做校准
                break
    print("校准完成！")

    # 步骤 C：转换为最终的 INT8 量化模型 (Convert)
    print("===> 5. 转换为 INT8 量化模型...")
    quantized_model = quantize_fx.convert_fx(prepared_model)
    quantized_model.eval()
    quantized_model.to("cpu")

    # 4. 评估 INT8 量化模型的准确率
    print("===> 6. 评估 INT8 量化模型准确率...")
    int8_acc = evaluate_accuracy(quantized_model, test_loader, device="cpu")
    print(f"INT8 量化模型准确率: {int8_acc:.2f}%")