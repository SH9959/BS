# ==============================
# Author: hsong
# 前期测试
# ==============================



import os
import sys
sys.path.append('/home/hsong/BS/EasyEdit')
import json
import random
from easyeditor import (
    FTHyperParams, 
    IKEHyperParams, 
    KNHyperParams, 
    MEMITHyperParams, 
    ROMEHyperParams, 
    LoRAHyperParams,
    MENDHyperParams,
    SERACHparams
    )
from easyeditor import BaseEditor
from easyeditor.models.ike import encode_ike_facts
from sentence_transformers import SentenceTransformer
from easyeditor import ZsreDataset

import math, random, time, argparse
from typing import Tuple, Dict, List
from copy import deepcopy

# line 50 保存路径,优先在slurm中指定
MUL_LOCALITY = True  # 表示使用自己随机抽取的locality测试集合，100 items

class MyPipeline:
    def __init__(
        self,
        editing_method:str='ROME', 
        model:str='gpt-j-6B', 
        dataset_dir:str='/home/hsong/BS/DATA/editing-data/data/portability/One Hop', 
        dataset_name:str='zsre_mend_eval_portability_gpt4.json',
        dataset_size=None,
        result_save_root_dir = 'output',
        SAVE_MODE = True,
        SAVE_PATH = ''
        
    ):
        self.SAVE_MODE = SAVE_MODE
        self.SAVE_PATH = SAVE_PATH
        self.dataset_size=dataset_size
        assert(editing_method in ['FT','LoRA','KN', 'ROME', 'MEMIT', 'PMET'])
        self.editing_method = editing_method
        assert(model in ['gpt-j-6B','mistral-7b',"Llama3-Chinese-8B-Instruct"])
        self.model = model
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.hparams_dir = f'/home/hsong/BS/EasyEdit/hparams/{self.editing_method}/{self.model}'
        
        self.data_path =  os.path.join(self.dataset_dir, self.dataset_name)
        
        self.save_path_for_one = f'./{result_save_root_dir}/multi_loc_test/sam_subject_add_re429/{dataset_name.split(".")[0]}/{editing_method}/{model}'  # 保存结果的路径
        
        if self.SAVE_PATH != '':
            self.save_path_for_one = self.SAVE_PATH
        
        tmp = self.save_path_for_one
        if not os.path.exists(tmp):
            os.makedirs(tmp)
        
        self.default_args_dict = {
            "Editing Method": self.editing_method,
            "Hyperparameters Directory": self.hparams_dir,
            "Data Directory": self.dataset_dir,
            "Dataset Size": self.dataset_size,
            "Metrics Save Directory": self.save_path_for_one
        }
        
        self.args_dict = deepcopy(self.default_args_dict)

        self.editing_hparams = None  # CLASS
        self.hparams = None          # METHOD

        the_method = self.args_dict['Editing Method']
        
        if the_method == 'FT':
            self.editing_hparams = FTHyperParams
        elif the_method == 'IKE':
            self.editing_hparams = IKEHyperParams
        elif the_method == 'KN':
            self.editing_hparams = KNHyperParams
        elif the_method == 'MEMIT':
            self.editing_hparams = MEMITHyperParams
        elif the_method == 'ROME':
            self.editing_hparams = ROMEHyperParams
        elif the_method == 'LoRA':
            self.editing_hparams = LoRAHyperParams
        else:
            raise NotImplementedError
        
        self.hparams = self.editing_hparams.from_hparams(self.args_dict['Hyperparameters Directory'])
        
        
    def read_dataset(self, random_rate:float=1) -> List[Dict]: # read the dataset, and choose the dataset size to use
        test_data = json.load(open(self.data_path, 'r', encoding='utf-8'))
        
        l = len(test_data)
        
        sample_num = math.ceil(l * random_rate)
        
        if abs(random_rate -1) <=0.0001:
            tes_dat = test_data
        else:
            tes_dat = random.sample(test_data, sample_num)
        
        if self.args_dict['Dataset Size'] is not None:  # Higher priority
            tes_dat = test_data[0:self.args_dict['Dataset Size']] # random.sample(test_data, self.args_dict['Dataset Size'])
            
        # info
        print("")
        print("\033[43;30m:DATASET INFO:\033[0m")
        print(f"len of {self.data_path}:{l}")
        print(f"len of sampled data: {len(tes_dat)}")
        print(f"The first 5 items of the datasets:")
        print(json.dumps(tes_dat[0:5], indent=4, ensure_ascii=False))
        print("")
        
        
        return tes_dat


    def extract_dataset(self, test_data: List[Dict]) -> Dict:  # Different datasets should be extracted in different way.

        global MUL_LOCALITY
        if self.data_path == '/home/hsong/BS/DATA/editing-data/data/portability/One Hop/zsre_mend_eval_portability_gpt4.json':  # 1037 items
            prompts = [test_data_['src'] for test_data_ in test_data]
            rephrase_prompts = [edit_data_['rephrase'] for edit_data_ in test_data]
            target_new = [edit_data_['alt'] for edit_data_ in test_data]
            locality_prompts = [edit_data_['loc'] for edit_data_ in test_data]
            locality_ans = [edit_data_['loc_ans'] for edit_data_ in test_data]
            if MUL_LOCALITY:
                with open('/home/hsong/BS/DATA/loc_qa_100.json', 'r') as f:
                    loc_data = json.load(f)
                locality_prompts = [[loc_dat['loc'] for loc_dat in loc_data]] * len(prompts) # [locality_prompts[0:20]] * len(prompts)  # 取40条先看看
                locality_ans = [[loc_dat['loc_ans'] for loc_dat in loc_data]] * len(prompts) # [locality_ans[0:20]] * len(prompts)
            portability_prompts = [edit_data_['portability']['New Question'] for edit_data_ in test_data]
            portability_ans = [edit_data_['portability']['New Answer'] for edit_data_ in test_data]
            locality_inputs = {
                'neighborhood':{
                    'prompt': locality_prompts,  
                    'ground_truth': locality_ans
                },
            }
            portability_inputs = {
                'one_hop':{
                    'prompt': portability_prompts,
                    'ground_truth': portability_ans
                },
            }
            subject = [edit_data_['subject'] for edit_data_ in test_data]   
            
        elif self.data_path == '/home/hsong/BS/DATA/editing-data/data/counterfact/counterfact-val.json':
            prompts = [test_data_['prompt'] for test_data_ in test_data]
            rephrase_prompts = [edit_data_['rephrase_prompt'] for edit_data_ in test_data]
            target_new = [edit_data_['target_new'] for edit_data_ in test_data]
            locality_prompts = [edit_data_['locality_prompt'] for edit_data_ in test_data]
            locality_ans = [edit_data_['locality_ground_truth'] for edit_data_ in test_data]
            if MUL_LOCALITY:
                with open('/home/hsong/BS/DATA/loc_qa_100.json', 'r') as f:
                    loc_data = json.load(f)
                locality_prompts = [[loc_dat['loc'] for loc_dat in loc_data]] * len(prompts) # [locality_prompts[0:20]] * len(prompts)  # 取40条先看看
                locality_ans = [[loc_dat['loc_ans'] for loc_dat in loc_data]] * len(prompts) # [locality_ans[0:20]] * len(prompts)
            subject = [edit_data_['subject'] for edit_data_ in test_data]   
            
            locality_inputs = {
                'neighborhood':{
                    'prompt': locality_prompts,  # 是否可以重复？我需要对每一个样本都进行多locality的计算
                    'ground_truth': locality_ans
                },
            }
            portability_inputs = None
            
        elif self.data_path == '/home/hsong/BS/DATA/KnowEdit-huggingface/benchmark/ZsRE/ZsRE-test-all.json': # 能验证不同关系的locality  # 1301 items
            
            prompts = [test_data_['prompt'] for test_data_ in test_data]
            rephrase_prompts = [edit_data_['rephrase_prompt'] for edit_data_ in test_data]
            target_new = [edit_data_['target_new'] for edit_data_ in test_data]
            locality_prompts = [[edit_data_['locality']['Relation_Specificity'][i]['prompt'] for i in range(len(edit_data_['locality']['Relation_Specificity']))] for edit_data_ in test_data]
            locality_ans = [[edit_data_['locality']['Relation_Specificity'][i]['ground_truth'][0] for i in range(len(edit_data_['locality']['Relation_Specificity']))] for edit_data_ in test_data]
            if MUL_LOCALITY:
                with open('/home/hsong/BS/DATA/loc_qa_100.json', 'r') as f:
                    loc_data = json.load(f)
                locality_prompts = [[loc_dat['loc'] for loc_dat in loc_data]] * len(prompts) # [locality_prompts[0:20]] * len(prompts)  # 取40条先看看
                locality_ans = [[loc_dat['loc_ans'] for loc_dat in loc_data]] * len(prompts) # [locality_ans[0:20]] * len(prompts)
            subject = [edit_data_['subject'] for edit_data_ in test_data]   
            
            locality_inputs = {
                'neighborhood':{
                    'prompt': locality_prompts,  # 
                    'ground_truth': locality_ans
                },
            }
            portability_prompts = [list(edit_data_['portability'].values())[0][0]['prompt'] for edit_data_ in test_data]
            portability_ans = [list(edit_data_['portability'].values())[0][0]['ground_truth'] for edit_data_ in test_data]
            portability_inputs = {
                'Any':{  # 'Reasoning' , "Logical_Generalization", "Subject_Aliasing"
                    'prompt': portability_prompts,
                    'ground_truth': portability_ans
                },
            }
            
        elif self.data_path == '/home/hsong/BS/DATA/KnowEdit-huggingface/benchmark/ZsRE/ZsRE-test-all_incident.json': # 能验证不同关系的locality  # 1301 items
            
            prompts = [test_data_['prompt'] for test_data_ in test_data]
            rephrase_prompts = [edit_data_['rephrase_prompt'] for edit_data_ in test_data]
            target_new = [edit_data_['target_new'] for edit_data_ in test_data]
            locality_prompts = [[edit_data_['locality']['Relation_Specificity'][i]['prompt'] for i in range(len(edit_data_['locality']['Relation_Specificity']))] for edit_data_ in test_data]
            locality_ans = [[edit_data_['locality']['Relation_Specificity'][i]['ground_truth'][0] for i in range(len(edit_data_['locality']['Relation_Specificity']))] for edit_data_ in test_data]
            
            incident_prompts = [[edit_data_['incident']['Relation_Specificity'][i]['prompt'] for i in range(len(edit_data_['incident']['Relation_Specificity']))] for edit_data_ in test_data]
            incident_ans = [[edit_data_['incident']['Relation_Specificity'][i]['ground_truth'][0] for i in range(len(edit_data_['incident']['Relation_Specificity']))] for edit_data_ in test_data]
           
            incident_inputs = {
                'neighborhood':{
                    'prompt': incident_prompts,  # 
                    'ground_truth': incident_ans
                },
            }
            if MUL_LOCALITY:
                with open('/home/hsong/BS/DATA/loc_qa_100.json', 'r') as f:
                    loc_data = json.load(f)
                locality_prompts = [[loc_dat['loc'] for loc_dat in loc_data]] * len(prompts) # [locality_prompts[0:20]] * len(prompts)  # 取40条先看看
                locality_ans = [[loc_dat['loc_ans'] for loc_dat in loc_data]] * len(prompts) # [locality_ans[0:20]] * len(prompts)
            subject = [edit_data_['subject'] for edit_data_ in test_data]   
            
            locality_inputs = {
                'neighborhood':{
                    'prompt': locality_prompts,  # 
                    'ground_truth': locality_ans
                },
            }
            portability_prompts = [list(edit_data_['portability'].values())[0][0]['prompt'] for edit_data_ in test_data]
            portability_ans = [list(edit_data_['portability'].values())[0][0]['ground_truth'] for edit_data_ in test_data]
            portability_inputs = {
                'Any':{  # 'Reasoning' , "Logical_Generalization", "Subject_Aliasing"
                    'prompt': portability_prompts,
                    'ground_truth': portability_ans
                },
            }
            
        else:
            pass
        
        if self.args_dict['Editing Method'] == 'IKE':
            train_data_path = os.path.join(self.dataset_dir, 'zsre_mend_train_10000.json')
            train_ds = ZsreDataset(train_data_path)
            sentence_model = SentenceTransformer(self.hparams.sentence_model_name).to(f'cuda:{self.hparams.device}')
            encode_ike_facts(sentence_model, train_ds, self.hparams)
        else:
            train_ds = None
            
            
            
        if 'incident' in self.data_path:
            ret = {
                'prompts': prompts,
                'rephrase_prompts': rephrase_prompts,
                'target_new': target_new,
                'subject': subject,
                'train_ds': train_ds,
                'locality_inputs': locality_inputs,
                'portability_inputs': portability_inputs,
                'keep_original_weight': True,
                'incident_inputs': incident_inputs
            }
        else:
            ret = {
                'prompts': prompts,
                'rephrase_prompts': rephrase_prompts,
                'target_new': target_new,
                'subject': subject,
                'train_ds': train_ds,
                'locality_inputs': locality_inputs,
                'portability_inputs': portability_inputs,
                'keep_original_weight': True,
            }

        return ret
    
    def do_edit(self, params: Dict)-> Tuple[object, object]:
        editor = BaseEditor.from_hparams(self.hparams)  #take long time 1-2min
        
        # if self.args_dict["Editing Method"] in ['MEMIT', 'PMET']:
        #     metrics, edited_model, _ = editor.batch_edit(**params)
        if self.args_dict["Editing Method"] in ['KN', 'ROME','FT','LoRA','MEMIT', 'PMET']:
            metrics, edited_model, _ = editor.edit(**params)
        else:
            raise ValueError(f"Unsupported Editing Method: {self.args_dict['Editing Method']}")
        return metrics, edited_model
    
    
    def main(self, ):

        test_data = self.read_dataset()
        
        ret = self.extract_dataset(test_data=test_data)
        
        if "爱因斯坦样例测试" and self.dataset_size == 1:
            sample = deepcopy(ret)
            sample["prompts"]=[
                    "爱因斯坦的专业是"
                ]
            sample["rephrase_prompts"]=[
                    "爱因斯坦专业是什么？"
                ]
            
            sample["target_new"] = [
                    "医学"
                ]
            
            sample["subject"]= [
                    "爱因斯坦"
                ]
            sample["portability_inputs"]={
                    "Any": {
                        "prompt": [
                            "爱因斯坦的妻子是"
                        ],
                        "ground_truth": [
                            "Mileva Marić"
                        ]
                    }
                }
            
            metrics, edited_model = self.do_edit(sample)
        
        else:
            metrics, edited_model = self.do_edit(ret)

        print(f"\033[34mMetrics:\n{metrics}\033[0m")
        
        tmp = self.save_path_for_one
        if not os.path.exists(tmp):
            os.makedirs(tmp)
            
        if self.SAVE_MODE:
            json.dump(metrics, open(os.path.join(self.save_path_for_one, f'EVAL_{self.editing_method}_{self.model}_results.json'), 'w'), indent=4)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', type=str, required=True)  # KN, ROME, MEMIT
    parser.add_argument('--model', type=str, required=True)  # gpt-j-6B, llama-7b
    parser.add_argument('--dataset_dir', type=str, required=True)  # '/home/hsong/BS/DATA/editing-data/data/portability/One Hop'
    parser.add_argument('--dataset_name', type=str, required=True)  # zsre_mend_eval_portability_gpt4.json
    parser.add_argument('--dataset_size', type=int)  # 
    parser.add_argument('--SAVE_MODE', type=bool, required=True)
    parser.add_argument('--SAVE_PATH', type=str, required=True)
    
    parser.add_argument('-d', '--debug', action='store_true', help='debug mode')
        
    args, _ = parser.parse_known_args()
    
    if args.debug:
        # if you use vscode on hpc-login-01
        import debugpy
        
        debugpy.connect(('192.168.1.50', 6789))
        debugpy.wait_for_client()
        debugpy.breakpoint()

    params = {
        'editing_method': args.editing_method, 
        'model': args.model, 
        'dataset_dir': args.dataset_dir, 
        'dataset_name': args.dataset_name,
        'dataset_size': args.dataset_size,
        'SAVE_MODE': args.SAVE_MODE,
        'SAVE_PATH': args.SAVE_PATH
    }
    
    Task = MyPipeline(**params)
    
    a = time.time()
    
    Task.main()
    
    b=time.time()
    
    print(f"\033[33mTOTAL TIME: {b-a} s\033[0m")
