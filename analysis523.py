# ========================================
# 分析/home/hsong/BS/statistic.py的生成结果
# ========================================


import json
import numpy as np
"""
验证激活单元的激增特性
表3-5
"""
def get_rates_of(json_path: str):
    assert("ZsRE-test-all" in json_path or "counterfact-val" in json_path)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    FDV = data['big_nums_record']
    RDSP = data['eat_all_record']
    RMP = data['eat_mean_record']
    
    # 定义阈值范围
    thres_big_nums = 500
    thres_eat_all_low = 0
    thres_eat_all_high = 10.0
    thres_eat_mean_low = 0
    thres_eat_mean_high = 0.99

    # 统计每个字段在阈值范围内的比例
    proportion_within_threshold = {
        'big_nums_record': np.mean(np.array(FDV) >= thres_big_nums),
        'eat_all_record': np.mean((thres_eat_all_low <= np.array(RDSP)) & (np.array(RDSP) <= thres_eat_all_high)),
        'eat_mean_record': np.mean((thres_eat_mean_low <= np.array(RMP)) & (np.array(RMP) <= thres_eat_mean_high))
    }

    print(proportion_within_threshold)



results = [
    # "/home/hsong/BS/activated_neurons/Llama3-Chinese-8B-Instruct_ZsRE-test-all.json_.json",
    "/home/hsong/BS/activated_neurons/Llama3-Chinese-8B-Instruct_counterfact-val.json_.json"
    # "/home/hsong/BS/activated_neurons/mistral-7b_ZsRE-test-all.json_.json",
    # "/home/hsong/BS/activated_neurons/mistral-7b_counterfact-val.json_.json"
]

for result in results:
    print(result)
    get_rates_of(result)


