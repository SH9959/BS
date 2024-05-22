import json
import math, random, time, argparse, os
from typing import Tuple, Dict, List, Optional, Union, Any
from copy import deepcopy
import torch

loc_file_path = "/home/hsong/BS/DATA/loc_qa_100_for_app.json"
DEBUG = True
def add_new_prompt_and_targetnew_to_loc_file(new_data: Dict, loc_file_path:str=loc_file_path):
    
    with open(loc_file_path, 'r', encoding='utf-8') as file:
        loc_data = json.load(file)

    loc_data.append(new_data)

    with open(loc_file_path, 'w', encoding='utf-8') as file:
        json.dump(loc_data, file, indent=4, ensure_ascii=False)  # 支持中文显示
        
        
def construct_input_data(sro, prompts:Union[List[str],str], origin_response:Optional[Union[List[str],str]],target_new:Union[List[str],str],loc_file_path:str=loc_file_path):
    
    if isinstance(prompts, str):
        prompts = [prompts]
        
    if isinstance(origin_response, str):
        origin_response = [origin_response]
    
    if isinstance(target_new, str):
        target_new = [target_new]
        
    
    with open(loc_file_path, 'r') as f:
        loc_data = json.load(f)
        print(len(loc_data))  # 1037+
        print(type(loc_data))
        print(loc_data[0])
        
        locality_prompts = [[loc_dat['loc'] for loc_dat in loc_data]] * len(prompts) # [locality_prompts[0:20]] * len(prompts)  # 取40条先看看
        locality_ans = [[loc_dat['loc_ans'] for loc_dat in loc_data]]  * len(prompts)# [locality_ans[0:20]] * len(prompts)
        
        locality_inputs = {
            'neighborhood':{
                'prompt': locality_prompts,  # 
                'ground_truth': locality_ans
            },
        }
        
    ret = {
        'prompts': prompts,
        'subject': sro[0],
        'ground_truth': origin_response,
        'target_new': target_new,
        'locality_inputs': locality_inputs,
        'keep_original_weight': True,
    }
    
    return ret
def check_tensors_same(t1:torch.Tensor, t2:torch.Tensor) -> bool:
    if torch.equal(t1, t2):
        print("两个张量完全相同。")
        same1 = True
    else:
        print("两个张量不完全相同。")
        same1 = False

    # 检查两个张量是否在一定的误差范围内每个元素都相等
    # 这个方法更加宽容，允许小的数值差异
    if torch.allclose(t1, t2):
        print("两个张量在一定的误差范围内每个元素都相等。")
        same2 = True
    else:
        print("两个张量在某些元素上超出了误差范围。")
        same2 = False
        
    return same2

def get_ip():
    return os.popen('hostname -I').read().strip()




def to_list(rl):
    wri = rl

    def tensor_to_list(data):
        if isinstance(data, torch.Tensor):
            return data.tolist()
        elif isinstance(data, dict):
            return {k: tensor_to_list(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [tensor_to_list(item) for item in data]
        else:
            return data

    # 遍历 rank_list 并将 tensor 转换为 list
    for layer, layer_info in wri.items():
        # 删除 'probability' 这一项
        del layer_info['probability']
        for target, target_info in layer_info['info'].items():
            target_info['prob'] = [tensor_to_list(prob) for prob in target_info['prob']]
            target_info['delt'] = [tensor_to_list(delt) for delt in target_info['delt']]

    return wri

def count_act_neurous(
    act_tensor:torch.Tensor, 
    thres:float=0.1
) -> Dict[str, Any]:

    positions = torch.nonzero(act_tensor > thres, as_tuple=True)
    # 将位置的tensor转换为坐标元组列表
    coordinates = list(zip(*positions))
    # 创建一个列表，其中包含每个激活元素的值和坐标
    #result = [{"value": act_tensor[coord], "coordinate": coord} for coord in coordinates] #超过thres的值和其对应坐标
    
    #包括统计信息：
    if DEBUG:
        print("\033[34m")
        print(f"层: {layer_index}")
        print(f"激活个数：{len(self.output[layer_index]['act_fn_info'])}")
        print(f"最大值：{max([i['value'] for i in self.output[layer_index]['act_fn_info']])}")
        print(f"平均值：{torch.mean(torch.tensor([i['value'] for i in self.output[layer_index]['act_fn_info']]))}")
        print("\033[0m")
    
    result = {
        "activated_neurons": [{"value": act_tensor[coord], "coordinate": coord} for coord in coordinates],
        "active_num": len(self.output[layer_index]['act_fn_info']),
        "max_value": max([float(i['value']) for i in self.output[layer_index]['act_fn_info']]),
        "mean_value": float(torch.mean(torch.tensor([i['value'] for i in self.output[layer_index]['act_fn_info']]))),
        "std_value": float(torch.std(torch.tensor([i['value'] for i in self.output[layer_index]['act_fn_info']]))),
        "min_value": min([float(i['value']) for i in self.output[layer_index]['act_fn_info']])
    }

    return result