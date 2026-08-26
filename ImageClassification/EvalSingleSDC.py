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

def calculate_softmax_entropy(prob_array):
    """
    计算一组已经过 Softmax 的概率数组的香农信息熵
    
    参数:
    prob_array : np.ndarray
        输入的一维或二维 Softmax 概率分布（例如：[0.999, 0.001, 0.0]）
    
    返回:
    entropy : float 或 np.ndarray
        计算出的熵值（单位：bit）。值越接近 0，代表分布越坍缩（可能存在软错误）。
    """
    # 核心安全防护：在学术界，软错误常常把其余类别抹杀成绝对的 0。
    # 如果直接运行 np.log2(0) 会触发运行时警告并返回 NaN。
    # 我们加入一个极小的微量偏移 eps（10的负12次方），既不影响精度，又能保证代码稳如磐石。
    eps = 1e-12
    p = np.clip(prob_array, eps, 1.0)
    
    # 重新归一化，确保概率之和严格为 1（消除 clip 带来的微小误差）
    p = p / np.sum(p, axis=-1, keepdims=True)
    
    # 标准香农熵公式：-∑ (p * log2(p))
    entropy = -np.sum(p * np.log2(p), axis=-1)
    
    return entropy

def is_abnormal_entropy(soft_output):
    entropy_cutoff = 0.1
    res=False
    entropy_value=calculate_softmax_entropy(soft_output)
    if entropy_value < entropy_cutoff:
        res=True
    return res

def ReadResult(result_dir,fn):
    fp = os.path.join(result_dir, fn)
    with open(fp, 'r') as rf:
        data_json = json.load(rf)
    label = data_json['label']
    output = np.array(data_json['output'])
    return label,output

def analyze_single_experiment(result_dir,golden_dir,thresholds=[0, 0.8, 0.85, 0.9, 0.95]):
    nan_inf_skip=False
    th_skip=True
    """
    核心分析引擎：解析单次实验目录下的所有 JSON 文件
    返回结构化的统计字典，与打印逻辑解耦
    """
    thresholds = np.array(thresholds)
    num_th = len(thresholds)
    
    total_num = 0
    nan_num = 0
    inf_num = 0
    
    # 用 NumPy 数组存储各阈值下的通过数和正确数（初始为0，不人为+1）
    th_sdc_counts=np.zeros(num_th)
    th_csdc_counts=np.zeros(num_th)
    th_csdc_reject_counts=np.zeros(num_th)
    th_csdc_pred_counts=np.zeros(num_th)

    assert os.path.exists(result_dir) and os.listdir(result_dir)

    for fn in os.listdir(result_dir):
        if not fn.endswith('.json'):
            continue
        total_num += 1

        #读取真实数据
        gold_label,gold_output=ReadResult(golden_dir,fn)
        golden_soft_output = SoftMaxDetector(gold_output)
        golden_pred_id=np.argmax(golden_soft_output)
        
        #读取故障数据
        label,output=ReadResult(result_dir,fn)
        soft_output = SoftMaxDetector(output)
        pred_id = np.argmax(soft_output)
            
        # 2. 异常值检测
        nan_flag = np.isnan(soft_output).any()
        inf_flag = np.isinf(soft_output).any()
        
        if nan_flag: nan_num += 1
        if inf_flag: inf_num += 1

        if (nan_flag or inf_flag) and nan_inf_skip:
            continue

        for id,th in enumerate(thresholds):
            if (soft_output!=golden_soft_output).any():
                th_sdc_counts[id]+=1
                if th<0.001:
                    if pred_id!=golden_pred_id:
                        th_csdc_counts[id]+=1
                        th_csdc_pred_counts[id]+=1
                else:
                    if np.max(soft_output)<th and th_skip:
                        continue
                    # if is_abnormal_entropy(soft_output):
                    #     continue
                    # if (np.max(soft_output)>th) and (np.max(golden_soft_output)>th):
                    if pred_id!=golden_pred_id:
                        th_csdc_counts[id]+=1
                        th_csdc_pred_counts[id]+=1
                    # else:
                    #     th_csdc_counts[id]+=1
                    #     th_csdc_reject_counts[id]+=1

    assert total_num!=0

    # 4. 计算最终体系结构与算法指标
    nan_rate = (nan_num / total_num) * 100
    inf_rate = (inf_num / total_num) * 100
    
    # 安全地计算各阈值下的准确率与覆盖率，防止分母为 0
    th_sdc = th_sdc_counts/total_num*100
    th_csdc = th_csdc_counts/total_num*100
    th_csed_reject=th_csdc_reject_counts/total_num*100
    th_csed_pred=th_csdc_pred_counts/total_num*100

    # 组装结构化报告
    report = {
        'total_num': total_num,
        'sdc': th_sdc[1] if th_skip else th_sdc[0],
        'csdc': th_csdc[1] if th_skip else th_csdc[0],
        'nan_rate': nan_rate,
        'inf_rate': inf_rate,
        'nan_num': nan_num,
        'inf_num': inf_num,
        'thresholds_data':[]
    }
    
    for i, th in enumerate(thresholds):
        report['thresholds_data'].append({
            'threshold': th,
            'sdc': th_sdc[i],
            'csdc': th_csdc[i],
            'csdc_reject':th_csed_reject[i],
            'csdc_pred':th_csed_pred[i],
        })
        
    return report
    

def print_formatted_report(report, prefix_info=""):
    """专职打印渲染，美化控制台输出"""
    if not report:
        print(f"{prefix_info} No data available.")
        return
        
    if prefix_info:
        print(prefix_info)
    print(f"  [Anomalies ] Nan Rate: {report['nan_rate']:.5f}%, Inf Rate: {report['inf_rate']:.5f}% (Total: {report['total_num']}, NaN: {report['nan_num']}, Inf: {report['inf_num']})")
    print("  [Selective ] Selection Framework Bounds:")
    for data in report['thresholds_data']:
        print(f"    Th: {data['threshold']:<4} | sdc: {data['sdc']:.4f}% | csdc: {data['csdc']:.4f}% | csdc_reject: {data['csdc_reject']:.4f}% | csdc_pred: {data['csdc_pred']:.4f}%")


# =========================================================================
# 主程序入口
# =========================================================================
if __name__ == '__main__':
    isQuant=True
    model_names = [
        ResNetConfig.ResNet18,
        # ResNetConfig.ResNet34,
        # ResNetConfig.ResNet50,
        # ResNetConfig.ResNet101,
    ]
    dataset = ResNetConfig.DATASET_CIFAR_10
    ber_rates = [1e-7,1e-6,1e-5,1e-4, 1e-3,1e-2, 1e-1]
    # ber_rates = [1e-2,1e-1]
    inj_times = 3000
    data_type_name = 'quint8' if isQuant else 'float32'
    

    for model_name in model_names:
        print("\n" + "*"*40)
        print(f"Evaluating Model: {model_name}")
        print("*"*40)
        
        # 1.
        golden_dir = os.path.join('golden', dataset, model_name+('_q8' if isQuant else ''))

        # 2. 遍历故障率进行评估
        for ber in ber_rates:
            print(f"\n" + "="*20 + f" BER: {ber} " + "="*20)
            
            dir_name = f'neuron_{dataset}_{data_type_name}_{model_name}_{ber}_{inj_times}'
            layers_dir = os.path.join('out', dir_name)
            
            if not os.path.exists(layers_dir):
                print(f"Directory not found: {layers_dir}")
                exit(-1)
                
            # 搜集当前故障率下所有层/子目录的实验结果
            layer_results = {}
            for dir_name_sub in os.listdir(layers_dir):
                # print(dir_name_sub)
                sub_result_dir = os.path.join(layers_dir, dir_name_sub)
                report = analyze_single_experiment(sub_result_dir,golden_dir)
                
                if report:
                    # 以 TOP1 精度为 Key，方便后续升序排序（复现原代码 sorted 逻辑）
                    layer_results[report['csdc']] = (dir_name_sub, report)
            # 按精度从低到高（最易受袭到最鲁棒的层）依次打印结果
            for idx, k in enumerate(sorted(layer_results.keys(),reverse=True)):
                sub_dir_name, rpt = layer_results[k]
                meta_info = f"[{idx}] Layer-Dir: {sub_dir_name} | Overall csdc: {rpt['csdc']:.4f}%"
                print_formatted_report(rpt, prefix_info=meta_info)