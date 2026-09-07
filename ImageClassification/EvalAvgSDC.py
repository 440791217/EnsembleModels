import os
import json
import pandas as pd
import numpy as np
import ResNetConfig


def soft_max_detector(data):
    x = np.array(data)
    # 减去最大值防止 exp 溢出，并清洗异常值
    x = np.nan_to_num(x, nan=0.0, posinf=1e10, neginf=-1e10)
    x_shifted = x - np.max(x)
    output = np.exp(x_shifted) / np.sum(np.exp(x_shifted))
    return output


def avg_sdc(fault_result_dir, golden_result_dir, ensemble_models_result_dirs):
    fns = os.listdir(fault_result_dir)
    total_num = 0
    correct_num = 0
    incorrect_num = 0  # 错误/SDC总数
    
    # 统计区间分布列表
    golden_sdc_bins = [0, 0, 0, 0, 0]    # 仅在预测错误时统计的 Golden 区间分布
    golden_total_bins = [0, 0, 0, 0, 0]  # 所有情况下的 Golden 区间分布 (<0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, >=0.8)
    
    fault_sdc_bins = [0, 0, 0, 0, 0, 0]    # 仅在预测错误时统计的 Fault 区间分布 (Exp/Inf, <0.2, ...)
    fault_total_bins = [0, 0, 0, 0, 0, 0]  # 所有情况下的 Fault 区间分布 (Exp/Inf, <0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, >=0.8)

    for fn in fns:
        if not fn.endswith('.json'):
            continue
        total_num += 1
        model_size = 1

        # 读取故障数据
        with open(os.path.join(fault_result_dir, fn), 'r') as rf:
            data_json = json.load(rf)
            output = data_json['output']
            fault_output_sum = soft_max_detector(data=output)

        # 读取黄金（正确）数据
        with open(os.path.join(golden_result_dir, fn), 'r') as rf:
            data_json = json.load(rf)
            output = data_json['output']
            golden_output_sum = soft_max_detector(data=output)

        # 累加集成模型的结果
        for result_dir in ensemble_models_result_dirs:
            fp = os.path.join(result_dir, fn)
            if not os.path.exists(fp):
                continue
            with open(fp, 'r') as rf:
                data_json = json.load(rf)
                output = data_json['output']
                output = soft_max_detector(data=output)
                fault_output_sum += output
                golden_output_sum += output
                model_size += 1

        golden_output_sum = golden_output_sum / model_size
        fault_output_sum = fault_output_sum / model_size
        
        golden_pred_id = np.argmax(golden_output_sum)
        fault_pred_id = np.argmax(fault_output_sum)

        golden_output_prob = np.max(golden_output_sum)
        fault_output_prob = np.max(fault_output_sum)

        # 1. 统计 Golden 的全局总数分布
        if golden_output_prob >= 0.8:
            golden_total_bins[4] += 1
        elif golden_output_prob >= 0.6:
            golden_total_bins[3] += 1
        elif golden_output_prob >= 0.4:
            golden_total_bins[2] += 1
        elif golden_output_prob >= 0.2:
            golden_total_bins[1] += 1
        else:
            golden_total_bins[0] += 1

        # 2. 统计 Fault 的全局总数分布
        if not np.isfinite(fault_output_sum).all():
            fault_total_bins[0] += 1  # 异常/无穷大
        elif fault_output_prob >= 0.8:
            fault_total_bins[5] += 1
        elif fault_output_prob >= 0.6:
            fault_total_bins[4] += 1
        elif fault_output_prob >= 0.4:
            fault_total_bins[3] += 1
        elif fault_output_prob >= 0.2:
            fault_total_bins[2] += 1
        else:
            fault_total_bins[1] += 1

        # 3. 判断正误及错误时的区间统计
        if golden_pred_id == fault_pred_id:
            correct_num += 1
        else:
            incorrect_num += 1
            # 统计 Golden 错误时的区间分布
            if golden_output_prob >= 0.8:
                golden_sdc_bins[4] += 1
            elif golden_output_prob >= 0.6:
                golden_sdc_bins[3] += 1
            elif golden_output_prob >= 0.4:
                golden_sdc_bins[2] += 1
            elif golden_output_prob >= 0.2:
                golden_sdc_bins[1] += 1
            else:
                golden_sdc_bins[0] += 1

            # 统计 Fault 错误时的区间分布
            if not np.isfinite(fault_output_sum).all():
                fault_sdc_bins[0] += 1  # 异常/无穷大
            elif fault_output_prob >= 0.8:
                fault_sdc_bins[5] += 1
            elif fault_output_prob >= 0.6:
                fault_sdc_bins[4] += 1
            elif fault_output_prob >= 0.4:
                fault_sdc_bins[3] += 1
            elif fault_output_prob >= 0.2:
                fault_sdc_bins[2] += 1
            else:
                fault_sdc_bins[1] += 1

    if total_num == 0:
        return 0.0, 0, 0, 0, golden_sdc_bins, golden_total_bins, fault_sdc_bins, fault_total_bins

    sdc_result = round(100 - correct_num / total_num * 100, 2)
    print(f"avg top1 sdc: {sdc_result}")
    
    return sdc_result, total_num, correct_num, incorrect_num, golden_sdc_bins, golden_total_bins, fault_sdc_bins, fault_total_bins


if __name__ == '__main__':
    is_quant = False
    fault_model_name = ResNetConfig.ResNet18
    model_names = [
        # ResNetConfig.ResNet18,
        ResNetConfig.ResNet34,
        ResNetConfig.ResNet50,
        ResNetConfig.ResNet101,
    ]
    dataset = ResNetConfig.DATASET_CIFAR_100
    ber_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    # ber_rates = [1e-5]
    inj_times = 3000
    data_type_name = 'quint8' if is_quant else 'float32'

    ensemble_model_result_dirs = []
    for model_name in model_names:
        result_dir = os.path.join('golden', dataset, model_name)
        ensemble_model_result_dirs.append(result_dir)

    golden_model_result_dir = os.path.join('golden', dataset,
                                           fault_model_name + "{}".format('_q8' if is_quant else ''))

    all_ber_results = {}
    all_layers_set = set()
    all_ber_detail_rows = {}

    for ber in ber_rates:
        print(f"\n" + "=" * 20 + f" BER: {ber} " + "=" * 20)
        fault_model_dir = f'neuron_{dataset}_{data_type_name}_{fault_model_name}_{ber}_{inj_times}'
        fault_layers_dir = os.path.join('out', fault_model_dir)

        if not os.path.exists(fault_layers_dir):
            print(f"Directory not found: {fault_layers_dir}, skipping this BER.")
            continue

        sub_layer_dirs = os.listdir(fault_layers_dir)
        layer_sdc_map = {}
        ber_detail_list = []

        for sub_layer_dir in sub_layer_dirs:
            fault_result_dir = os.path.join(fault_layers_dir, sub_layer_dir)
            print(f"Evaluating Layer: {sub_layer_dir}")

            sdc_res = avg_sdc(
                fault_result_dir=fault_result_dir,
                golden_result_dir=golden_model_result_dir,
                ensemble_models_result_dirs=ensemble_model_result_dirs
            )
            
            if isinstance(sdc_res, tuple) and len(sdc_res) == 8:
                sdc_val, total_num, correct_num, incorrect_num, golden_bins, golden_tot_bins, fault_bins, fault_tot_bins = sdc_res
            else:
                continue

            layer_sdc_map[sub_layer_dir] = sdc_val
            all_layers_set.add(sub_layer_dir)

            # 写入详细统计列表，包含全局总数统计 bins 和错误时 bins
            ber_detail_list.append({
                'Layer_ID': sub_layer_dir,
                'Total_Samples': total_num,
                'Correct_Count': correct_num,
                'Incorrect_SDC_Count': incorrect_num,
                'SDC_Percentage(%)': sdc_val,
                
                # 全局总数分布
                'Golden_Total_<0.2': golden_tot_bins[0],
                'Golden_Total_0.2-0.4': golden_tot_bins[1],
                'Golden_Total_0.4-0.6': golden_tot_bins[2],
                'Golden_Total_0.6-0.8': golden_tot_bins[3],
                'Golden_Total_>=0.8': golden_tot_bins[4],
                
                'Fault_Total_Exp/Inf': fault_tot_bins[0],
                'Fault_Total_<0.2': fault_tot_bins[1],
                'Fault_Total_0.2-0.4': fault_tot_bins[2],
                'Fault_Total_0.4-0.6': fault_tot_bins[3],
                'Fault_Total_0.6-0.8': fault_tot_bins[4],
                'Fault_Total_>=0.8': fault_tot_bins[5],

                # 错误时分布 (SDC Bins)
                'Golden_SDC_<0.2': golden_bins[0],
                'Golden_SDC_0.2-0.4': golden_bins[1],
                'Golden_SDC_0.4-0.6': golden_bins[2],
                'Golden_SDC_0.6-0.8': golden_bins[3],
                'Golden_SDC_>=0.8': golden_bins[4],
                
                'Fault_SDC_Exp/Inf': fault_bins[0],
                'Fault_SDC_<0.2': fault_bins[1],
                'Fault_SDC_0.2-0.4': fault_bins[2],
                'Fault_SDC_0.4-0.6': fault_bins[3],
                'Fault_SDC_0.6-0.8': fault_bins[4],
                'Fault_SDC_>=0.8': fault_bins[5],
            })

        all_ber_results[ber] = layer_sdc_map
        if ber_detail_list:
            all_ber_detail_rows[ber] = ber_detail_list

    # 1. 整理总表数据 (SDC Summary)
    sorted_layers = sorted(list(all_layers_set))
    final_rows = []

    for layer_id in sorted_layers:
        row = {'Layer_ID': layer_id}
        for ber in ber_rates:
            col_name = f"SDC_{ber}"
            row[col_name] = all_ber_results.get(ber, {}).get(layer_id, None)
        final_rows.append(row)

    # 定义 Excel 输出路径
    excel_output_path = os.path.join('ensemble_sdc_report.xlsx')

    if final_rows or all_ber_detail_rows:
        with pd.ExcelWriter(excel_output_path, engine='openpyxl') as writer:
            # 写入总表
            if final_rows:
                df_final = pd.DataFrame(final_rows)
                df_final.to_excel(writer, sheet_name='SDC_Summary', index=False)

            # 为每个 BER 单独写入详细统计表
            for ber, detail_rows in all_ber_detail_rows.items():
                df_detail = pd.DataFrame(detail_rows)
                sheet_name = f"Stats_BER_{ber}"
                df_detail.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"\n[Success] 所有总数统计、全局分布及错误区间结果已成功保存至 Excel 文件: {excel_output_path}")
    else:
        print("\n[Warning] 没有收集到任何有效数据，未生成 Excel 文件。")