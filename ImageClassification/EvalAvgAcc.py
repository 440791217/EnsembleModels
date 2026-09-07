import os
import json
import ResNetConfig
import numpy as np
import pandas as pd
import torch


def soft_max_detector(data):
    x = np.array(data)
    output = np.exp(x) / np.sum(np.exp(x))
    return output


def avg_acc(golden_model_dirs):
    fns = os.listdir(golden_model_dirs[0])
    total_num = 0
    correct_num = 0
    correct_num_5 = 0
    ######################################
    for fn in fns:
        total_num += 1
        data_json_list = []
        for result_dir in golden_model_dirs:
            fp = os.path.join(result_dir, fn)
            with open(fp, 'r') as rf:
                data_json = json.load(rf)
            data_json_list.append(data_json)

        # 平均加权
        output_sum = None  # 初始值
        model_size = 0
        for id, data_json in enumerate(data_json_list):
            output = data_json['output']
            soft_output = soft_max_detector(data=output)
            if output_sum is None:
                output_sum = soft_output
            else:
                output_sum += soft_output
            model_size += 1

        output_sum = output_sum / model_size
        pred_id = np.argmax(output_sum)
        label = data_json_list[0]['label']

        if label == pred_id:
            correct_num += 1

        top_5_indices = np.argsort(output_sum)[::-1][:5]
        if label in top_5_indices:
            correct_num_5 += 1

    print("avg top1 acc:{}".format(round(correct_num / total_num * 100, 2)))
    print("avg top5 acc:{}".format(round(correct_num_5 / total_num * 100, 2)))
    pass


if __name__ == '__main__':
    print("Average!")
    isQuant=True
    model_names = [
        ResNetConfig.ResNet18,
        ResNetConfig.ResNet34,
        ResNetConfig.ResNet50,
        ResNetConfig.ResNet101,
    ]
    # dataset = ResNetConfig.DATASET_CIFAR_10
    dataset = ResNetConfig.DATASET_CIFAR_100
    golden_result_dirs = []
    for model_name in model_names:
        result_dir = os.path.join('golden', dataset, model_name+("_q8" if isQuant else ""))
        golden_result_dirs.append(result_dir)
    print('dataset:',dataset,'model_names:',model_names)
    avg_acc(golden_model_dirs=golden_result_dirs)