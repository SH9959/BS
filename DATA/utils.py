import json
import random
from typing import List, Dict, Tuple, Union

def pprint(data):
    print(json.dumps(data, indent=4, ensure_ascii=False))

def dataset_info(data_path:Union[str, List[str]] = "/home/hsong/BS/DATA/editing-data/data/counterfact/counterfact-val.json"):
    #data_path = "/home/hsong/BS/DATA/editing-data/data/counterfact/counterfact-val.json"   # 1919
    #data_path = '/home/hsong/BS/DATA/KnowEdit-huggingface/benchmark/ZsRE/ZsRE-test-all.json #1301
    with open(data_path, 'r') as file:
        d = json.load(file)


    print(f"\033[43;37m==INFO==\033[m {data_path}")
    print('len:',len(d))
    print(f"First data:")
    pprint(d[0])
    print('')


# 读取原始 JSON 文件
with open('/home/hsong/BS/DATA/editing-data/data/portability/One Hop/zsre_mend_eval_portability_gpt4.json', 'r') as file:
    data = json.load(file)

# 提取"loc"和"loc_ans"字段
extracted_data = []
for item in data:
    extracted_item = {
        "loc": item.get("loc"),
        "loc_ans": item.get("loc_ans")
    }
    extracted_data.append(extracted_item)

# 将提取出的数据保存为新的 JSON 文件
with open('loc_qas.json', 'w') as file:
    json.dump(extracted_data, file, indent=4)
    
    
def pick_loc_data(num:int=20):

    with open('loc_qas.json', 'r') as file:
        data = json.load(file)
    selected_data = random.sample(data, num)

    loc_list = [item['loc'] for item in selected_data]
    loc_ans_list = [item['loc_ans'] for item in selected_data]

    output_data = [{"loc": loc, "loc_ans": loc_ans} for loc, loc_ans in zip(loc_list, loc_ans_list)]
    with open(f'loc_qa_{num}.json', 'w') as outfile:
        json.dump(output_data, outfile, indent=4)

    print(f"SUCCESS SAVED: loc_qa_{num}.json")

def extend_incident(json_path:str="/home/hsong/BS/DATA/KnowEdit-huggingface/benchmark/ZsRE/ZsRE-test-all.json"):
    
    save_path = json_path.split('.')[0] + '_incident.json'
    
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    for item in data:
        if 'incident' not in item:
            item['incident'] = item["locality"]
            
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"\033[32msave successfully in {save_path}\033[0m")


if __name__ == "__main__":
    # pick_loc_data(100)
    # dataset_info(data_path = '/home/hsong/BS/DATA/KnowEdit-huggingface/benchmark/ZsRE/ZsRE-test-all.json')
    extend_incident()