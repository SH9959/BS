import json
import math, random, time, argparse
from typing import Tuple, Dict, List, Optional, Union
from copy import deepcopy


loc_file_path = "/home/hsong/BS/DATA/loc_qa_100_for_app.json"

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



