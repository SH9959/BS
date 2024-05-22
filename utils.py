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
        if isinstance(layer, str):
            continue
        
        del layer_info['probability']
        for target, target_info in layer_info['info'].items():
            target_info['prob'] = [tensor_to_list(prob) for prob in target_info['prob']]
            target_info['delt'] = [tensor_to_list(delt) for delt in target_info['delt']]

    return wri

def count_act_neurous(
    act_tensor:torch.Tensor, 
    thres:float=0.085
) -> Dict[str, Any]:
    
    mode = "see_the_big"
    if mode=="see_all":
        thres = 0

    positions = torch.nonzero(act_tensor > thres, as_tuple=True)
    # 将位置的tensor转换为坐标元组列表
    coordinates = list(zip(*positions))
    # 创建一个列表，其中包含每个激活元素的值和坐标
    v_c = [{"value": act_tensor[coord], "coordinate": coord} for coord in coordinates] #超过thres的值和其对应坐标
    # 这里只考量了激活的神经元情况（>0.1视为激活）
    #包括统计信息：
    if len(v_c) > 0: 
        result = {
            "activated_neurons": v_c,
            "active_num": len(v_c), #len(self.output[layer_index]['act_fn_info']),
            "max_value": max([float(i['value']) for i in v_c]),
            "mean_value": float(torch.mean(torch.tensor([i['value'] for i in v_c]))),
            "std_value": float(torch.std(torch.tensor([i['value'] for i in v_c]))),
            "min_value": min([float(i['value']) for i in v_c])
        }
    else:
        result = {
            "activated_neurons": v_c,
            "active_num": len(v_c), #len(self.output[layer_index]['act_fn_info']),
            "max_value": 0, # max([float(i['value']) for i in v_c]),
            "mean_value": 0, # float(torch.mean(torch.tensor([i['value'] for i in v_c]))),
            "std_value": 0, # float(torch.std(torch.tensor([i['value'] for i in v_c]))),
            "min_value": 0 # min([float(i['value']) for i in v_c])
        }
    return result
def get_list_of_act_num_of_all_layers(
    info_list: Dict
)->List[int]:
    """根据处理结果得到激活个数的列表
    一个数据对应一个32大小列表
    
    """
    nums = []
    delta1 = []
    activated_neurons_num_of_last_layer = 0 #记录一下差分值
    for layer_index, outputs in info_list.items():
        nums.append(info_list[layer_index]['act_fn_info']['active_num'])
        # info_list[layer_index]['act_fn_info']
        # diff1 = info_list[layer_index]['act_fn_info']['active_num'] - activated_neurons_num_of_last_layer
        # activated_neurons_num_of_last_layer = info_list[layer_index]['act_fn_info']['active_num']
        # delta1.append(diff1)
    return nums
    # # 增加差分信息，准备取最大值最作为选择的层
    # info_list['act_delta_list']=delta1
    
def get_diff1_of_list(input_list:List[int]) -> List[int]:
    # 获取列表中相邻两个元素之间的差值
    diff1 = [input_list[0]]
    tmp = [input_list[i+1]-input_list[i] for i in range(len(input_list)-1)]
    diff1 = diff1 + tmp
    return diff1
def check_is_big_num(input_num:List[int]):
    thres = 1000
    if max(input_num) > thres:
        return True
    else:
        return False
    
def check_is_eat_all_before(act_nums:List[int], ind:int):
    
    sum_befor_ind = sum(act_nums[:ind])
    fenzi = abs(act_nums[ind] - sum_befor_ind)
    fenmu = abs(act_nums[ind])
    
    if fenzi / fenmu < 0.15:
        return True
    else:
        return False