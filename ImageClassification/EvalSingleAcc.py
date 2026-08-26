import os
import json
import pandas as pd
import ResNetConfig
import numpy as np
from FI import FaultInjectionConfig

def SoftMaxDetector(data):
    x = np.array(data)
    output = np.exp(x) / np.sum(np.exp(x))
    return output 


def analyze_single_experiment(result_dir,thresholds=[0, 0.8, 0.85, 0.9, 0.95]):
    """
    核心分析引擎：解析单次实验目录下的所有 JSON 文件
    返回结构化的统计字典，与打印逻辑解耦
    """
    thresholds = np.array(thresholds)
    num_th = len(thresholds)
    
    total_num = 0
    correct_top1 = 0
    correct_top5 = 0
    nan_num = 0
    inf_num = 0
    
    # 用 NumPy 数组存储各阈值下的通过数和正确数（初始为0，不人为+1）
    th_passed_counts = np.zeros(num_th)
    th_correct_counts = np.zeros(num_th)

    assert os.path.exists(result_dir) and os.listdir(result_dir)

    for fn in os.listdir(result_dir):
        if not fn.endswith('.json'):
            continue
        total_num += 1
        
        fp = os.path.join(result_dir, fn)
        with open(fp, 'r') as rf:
            data_json = json.load(rf)
            
        label = data_json['label']
        output = np.array(data_json['output'])
        
        # 1. 基础预测分析 (利用 NumPy 提升效率)
        pred_id = np.argmax(output)
        top5_indices = np.argsort(output)[::-1][:5]
        
        if label == pred_id:
            correct_top1 += 1
        if label in top5_indices:
            correct_top5 += 1
            
        # 2. 异常值与 Softmax 检测
        soft_output = SoftMaxDetector(output)
        nan_flag = np.isnan(soft_output).any()
        inf_flag = np.isinf(soft_output).any()
        
        if nan_flag: nan_num += 1
        if inf_flag: inf_num += 1

        if 0:#开启nan和inf检查
            if nan_flag or inf_flag:
                continue

        # 3. 选择性分类阈值评估 (利用 NumPy 广播机制消除内部 for 循环)
        max_prob = np.max(soft_output)
        passed_mask = (max_prob >= thresholds)  # 得到布尔阵，例如 [True, True, False, False, False]
        
        th_passed_counts += passed_mask
        if pred_id == label:
            th_correct_counts += passed_mask

    assert total_num!=0

    # 4. 计算最终体系结构与算法指标
    top1_acc = (correct_top1 / total_num) * 100
    top5_acc = (correct_top5 / total_num) * 100
    nan_rate = (nan_num / total_num) * 100
    inf_rate = (inf_num / total_num) * 100
    
    # 安全地计算各阈值下的准确率与覆盖率，防止分母为 0
    th_accs = np.divide(th_correct_counts, th_passed_counts, 
                        out=np.zeros_like(th_correct_counts), where=th_passed_counts > 0) * 100
    th_coverages = (th_passed_counts / total_num) * 100

    # 组装结构化报告
    report = {
        'total_num': total_num,
        'top1_acc': top1_acc,
        'top5_acc': top5_acc,
        'nan_rate': nan_rate,
        'inf_rate': inf_rate,
        'nan_num': nan_num,
        'inf_num': inf_num,
        'thresholds_data': []
    }
    
    for i, th in enumerate(thresholds):
        report['thresholds_data'].append({
            'threshold': th,
            'acc': th_accs[i],
            'coverage': th_coverages[i],
            'correct_num': int(th_correct_counts[i]),
            'avail_num': int(th_passed_counts[i])
        })
        
    return report
    

def print_formatted_report(report, prefix_info=""):
    """专职打印渲染，美化控制台输出"""
    if not report:
        print(f"{prefix_info} No data available.")
        return
        
    if prefix_info:
        print(prefix_info)
    print(f"  [Base Metric] TOP1 ACC: {report['top1_acc']:.4f}%, TOP5 ACC: {report['top5_acc']:.4f}%")
    print(f"  [Anomalies ] Nan Rate: {report['nan_rate']:.5f}%, Inf Rate: {report['inf_rate']:.5f}% (Total: {report['total_num']}, NaN: {report['nan_num']}, Inf: {report['inf_num']})")
    print("  [Selective ] Selection Framework Bounds:")
    for data in report['thresholds_data']:
        print(f"    Th: {data['threshold']:<4} | Acc: {data['acc']:.4f}% | Coverage: {data['coverage']:.4f}% | (Passed/Total: {data['avail_num']}/{report['total_num']})")


# =========================================================================
# 主程序入口
# =========================================================================
if __name__ == '__main__':
    model_names = [
        # ResNetConfig.ResNet18+'_q8',
        ResNetConfig.ResNet34+'_q8',
        # ResNetConfig.ResNet50,
        # ResNetConfig.ResNet101,
    ]
    dataset = ResNetConfig.DATASET_CIFAR_10
    ber_rates = [1e-7,1e-6,1e-5,1e-4, 1e-3,1e-2, 1e-1]
    ber_rates = []
    inj_times = 3000
    data_type = FaultInjectionConfig.GetDType()
    data_type_name = FaultInjectionConfig.GetDTypeName(dataType=data_type)

    for model_name in model_names:
        print("\n" + "*"*40)
        print(f"🚀 Evaluating Model: {model_name}")
        print("*"*40)
        
        # 1. 评估并打印无故障状态 (Golden Baseline)
        golden_dir = os.path.join('golden', dataset, model_name)
        golden_report = analyze_single_experiment(golden_dir)
        print_formatted_report(golden_report, prefix_info="--- [GOLDEN BASELINE] ---")
        if 1:
            continue
        # 2. 遍历故障率进行评估
        for ber in ber_rates:
            print(f"\n" + "="*20 + f" BER: {ber} " + "="*20)
            
            dir_name = f'neuron_{dataset}_{data_type_name}_{model_name}_{ber}_{inj_times}'
            layers_dir = os.path.join('out', dir_name)
            
            if not os.path.exists(layers_dir):
                print(f"Directory not found: {layers_dir}")
                continue
                
            # 搜集当前故障率下所有层/子目录的实验结果
            layer_results = {}
            for dir_name_sub in os.listdir(layers_dir):
                sub_result_dir = os.path.join(layers_dir, dir_name_sub)
                report = analyze_single_experiment(sub_result_dir)
                
                if report:
                    # 以 TOP1 精度为 Key，方便后续升序排序（复现原代码 sorted 逻辑）
                    layer_results[report['top1_acc']] = (dir_name_sub, report)
            
            # 按精度从低到高（最易受袭到最鲁棒的层）依次打印结果
            for idx, k in enumerate(sorted(layer_results.keys())):
                sub_dir_name, rpt = layer_results[k]
                meta_info = f"[{idx}] Layer-Dir: {sub_dir_name} | Overall TOP1: {rpt['top1_acc']:.4f}%"
                print_formatted_report(rpt, prefix_info=meta_info)