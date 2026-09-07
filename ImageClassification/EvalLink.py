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


def get_topk_preds(prob_array, k=5):
    """获取概率数组中前 K 个最大值的类别 ID 列表（Top-K）"""
    if not np.isfinite(prob_array).all():
        return []
    topk_indices = np.argsort(prob_array)[-k:][::-1]
    return topk_indices.tolist()


def evaluate_consistency(fault_result_dir, golden_result_dir, ensemble_models_result_dirs, TOP_K=5):
    fns = os.listdir(fault_result_dir)
    total_num = 0
    consistent_num = 0  # 黄金与故障预测一致的样本数
    mismatch_num = 0    # 预测结果不一致（偏差）的总数
    
    # 统计区间分布列表（基于置信度分段）
    golden_consistent_bins = [0, 0, 0, 0, 0]  # 预测一致时的 Golden 区间分布
    golden_total_bins = [0, 0, 0, 0, 0]       # 所有情况下的 Golden 区间分布
    
    fault_mismatch_bins = [0, 0, 0, 0, 0, 0]  # 仅在预测不一致时统计的 Fault 区间分布 (Exp/Inf, <0.2, ...)
    fault_total_bins = [0, 0, 0, 0, 0, 0]     # 所有情况下的 Fault 区间分布

    for fn in fns:
        if not fn.endswith('.json'):
            continue
        total_num += 1

        # 1. 读取故障数据
        with open(os.path.join(fault_result_dir, fn), 'r') as rf:
            data_json = json.load(rf)
            fault_raw_output = data_json['output']
            fault_prob = soft_max_detector(data=fault_raw_output)

        # 2. 读取黄金（标准）数据
        with open(os.path.join(golden_result_dir, fn), 'r') as rf:
            data_json = json.load(rf)
            golden_raw_output = data_json['output']
            golden_prob = soft_max_detector(data=golden_raw_output)

        # 收集参与环形校验的模型概率列表
        fault_prob_list = [fault_prob]
        golden_prob_list = [golden_prob]
        
        golden_output_sum = golden_prob.copy()
        fault_output_sum = fault_prob.copy()
        model_size = 1

        # 3. 读取并收集集成模型结果
        for result_dir in ensemble_models_result_dirs:
            fp = os.path.join(result_dir, fn)
            if not os.path.exists(fp):
                continue
            with open(fp, 'r') as rf:
                data_json = json.load(rf)
                ens_output = data_json['output']
                ens_prob = soft_max_detector(data=ens_output)
                
                fault_prob_list.append(ens_prob)
                golden_prob_list.append(ens_prob)
                
                fault_output_sum += ens_prob
                golden_output_sum += ens_prob
                model_size += 1

        # 计算加权平均后的概率分布
        golden_output_sum = golden_output_sum / model_size
        fault_output_sum = fault_output_sum / model_size
        
        golden_output_prob = np.max(golden_output_sum)
        fault_output_prob = np.max(fault_output_sum)

        golden_pred_id = np.argmax(golden_output_sum)
        fault_pred_id = np.argmax(fault_output_sum)

        # ==================== 环形 Top-K 一致性筛查函数 ====================
        def check_ring_topk_consensus(prob_list, k=TOP_K):
            n = len(prob_list)
            if n < 2:
                return True
            for i in range(n):
                current_model_prob = prob_list[i]
                next_model_prob = prob_list[(i + 1) % n]

                if not np.isfinite(current_model_prob).all() or not np.isfinite(next_model_prob).all():
                    return False

                current_pred_top1 = np.argmax(current_model_prob)
                next_topk_list = get_topk_preds(next_model_prob, k=k)

                if current_pred_top1 not in next_topk_list:
                    return False
            return True

        fault_ring_passed = check_ring_topk_consensus(fault_prob_list, k=TOP_K)
        golden_ring_passed = check_ring_topk_consensus(golden_prob_list, k=TOP_K)
        # ===================================================================

        # 4. 判断正误及置信度区间统计
        if golden_pred_id == fault_pred_id:
            consistent_num += 1
            # 统计 Golden 全局总数分布
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

            if golden_ring_passed:
                if golden_output_prob >= 0.8:
                    golden_consistent_bins[4] += 1
                elif golden_output_prob >= 0.6:
                    golden_consistent_bins[3] += 1
                elif golden_output_prob >= 0.4:
                    golden_consistent_bins[2] += 1
                elif golden_output_prob >= 0.2:
                    golden_consistent_bins[1] += 1
                else:
                    golden_consistent_bins[0] += 1
        else:
            mismatch_num += 1
            if not np.isfinite(fault_output_sum).all():
                fault_total_bins[0] += 1
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

            if fault_ring_passed:
                if not np.isfinite(fault_output_sum).all():
                    fault_mismatch_bins[0] += 1
                elif fault_output_prob >= 0.8:
                    fault_mismatch_bins[5] += 1
                elif fault_output_prob >= 0.6:
                    fault_mismatch_bins[4] += 1
                elif fault_output_prob >= 0.4:
                    fault_mismatch_bins[3] += 1
                elif fault_output_prob >= 0.2:
                    fault_mismatch_bins[2] += 1
                else:
                    fault_mismatch_bins[1] += 1

    if total_num == 0:
        return 0.0, 0, 0, 0, golden_consistent_bins, golden_total_bins, fault_mismatch_bins, fault_total_bins

    mismatch_rate = round(mismatch_num / total_num * 100, 2)
    print(f"Mismatch Rate: {mismatch_rate}%")
    
    return mismatch_rate, total_num, consistent_num, mismatch_num, golden_consistent_bins, golden_total_bins, fault_mismatch_bins, fault_total_bins


if __name__ == '__main__':
    is_quant = True
    fault_model_name = ResNetConfig.ResNet18
    model_names = [
        ResNetConfig.ResNet34,
        # ResNetConfig.ResNet50,
        # ResNetConfig.ResNet101,
    ]
    dataset = ResNetConfig.DATASET_CIFAR_100
    ber_rates = [1e-7]
    ber_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
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
        layer_mismatch_map = {}
        ber_detail_list = []

        for sub_layer_dir in sub_layer_dirs[0:]:
            fault_result_dir = os.path.join(fault_layers_dir, sub_layer_dir)
            print(f"Evaluating Layer: {sub_layer_dir}")

            eval_res = evaluate_consistency(
                fault_result_dir=fault_result_dir,
                golden_result_dir=golden_model_result_dir,
                ensemble_models_result_dirs=ensemble_model_result_dirs
            )
            
            if isinstance(eval_res, tuple) and len(eval_res) == 8:
                (mismatch_val, total_num, consistent_num, mismatch_num, 
                 golden_bins, golden_tot_bins, fault_bins, fault_tot_bins) = eval_res
            else:
                continue

            layer_mismatch_map[sub_layer_dir] = mismatch_val
            all_layers_set.add(sub_layer_dir)

            # 写入精简后的详细统计报表
            ber_detail_list.append({
                'Layer_ID': sub_layer_dir,
                'Total_Samples': total_num,
                'Consistent_Count': consistent_num,
                'Mismatch_Count': mismatch_num,
                'Mismatch_Percentage(%)': mismatch_val,
                
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

                'Golden_Consistent_<0.2': golden_bins[0],
                'Golden_Consistent_0.2-0.4': golden_bins[1],
                'Golden_Consistent_0.4-0.6': golden_bins[2],
                'Golden_Consistent_0.6-0.8': golden_bins[3],
                'Golden_Consistent_>=0.8': golden_bins[4],
                
                'Fault_Mismatch_Exp/Inf': fault_bins[0],
                'Fault_Mismatch_<0.2': fault_bins[1],
                'Fault_Mismatch_0.2-0.4': fault_bins[2],
                'Fault_Mismatch_0.4-0.6': fault_bins[3],
                'Fault_Mismatch_0.6-0.8': fault_bins[4],
                'Fault_Mismatch_>=0.8': fault_bins[5],
            })

        all_ber_results[ber] = layer_mismatch_map
        if ber_detail_list:
            all_ber_detail_rows[ber] = ber_detail_list

    # 1. 整理总表数据 (Mismatch Summary)
    sorted_layers = sorted(list(all_layers_set))
    final_rows = []

    for layer_id in sorted_layers:
        row = {'Layer_ID': layer_id}
        for ber in ber_rates:
            col_name = f"Mismatch_{ber}"
            row[col_name] = all_ber_results.get(ber, {}).get(layer_id, None)
        final_rows.append(row)

    excel_output_path = os.path.join('consistency_.xlsx')

    if final_rows or all_ber_detail_rows:
        with pd.ExcelWriter(excel_output_path, engine='openpyxl') as writer:
            if final_rows:
                df_final = pd.DataFrame(final_rows)
                df_final.to_excel(writer, sheet_name='Mismatch_Summary', index=False)

            for ber, detail_rows in all_ber_detail_rows.items():
                df_detail = pd.DataFrame(detail_rows)
                sheet_name = f"Stats_BER_{ber}"
                df_detail.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"\n[Success] 一致性偏差统计结果已成功保存至 Excel: {excel_output_path}")
    else:
        print("\n[Warning] 没有收集到任何有效数据，未生成 Excel 文件。")