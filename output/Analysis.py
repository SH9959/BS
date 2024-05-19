import os
import json
from typing import List, Dict, Tuple
import statistics
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np

def caculate_mean_time(data:List) -> float:
    t = 0
    l = len(data)
    for i in range(l):
        t += data[i]['time']
    t = t / l
    return t

def caculate_mean_reliability(data:List) -> float:  # 每个prompt对应1个    1
    t1 = 0
    l = len(data)
    for i in range(l):
        t1 += data[i]['pre']['rewrite_acc'][0]
    t1 = t1 / l
    
    t2 = 0
    for i in range(l):
        t2 += data[i]['post']['rewrite_acc'][0]
    t2 = t2 / l

    return t1, t2

def caculate_mean_generalization(data:List) -> float:  # 每个prompt对应>=1个   1-3
    t1 = 0
    l = len(data)
    for i in range(l):
        t1 += statistics.mean(data[i]['pre']['rephrase_acc'])
    t1 = t1 / l
    
    t2 = 0
    for i in range(l):
        t2 += statistics.mean(data[i]['post']['rephrase_acc'])
    t2 = t2 / l

    return t1, t2

def caculate_mean_portability(data:List) -> float:  # 每个prompt对应>=1个    1-3
    t1 = 0
    l = len(data)
    for i in range(l):
        t1 += statistics.mean(data[i]['pre']['portability']['Any_acc'])
    t1 = t1 / l
    
    t2 = 0
    for i in range(l):
        t2 += statistics.mean(data[i]['post']['portability']['Any_acc'])
    t2 = t2 / l

    return t1, t2

def caculate_mean_locality(data:List) -> float: # 每个prompt对应>=1个, locality可能要更多，  100-1000
    # t1 = 0
    # l = len(data)
    # for i in range(l):
    #     t1 += statistics.mean(data[i]['pre']['locality']["neighborhood_acc"])
    # t1 = t1 / l
    
    t2 = 0
    l = len(data)
    for i in range(l):
        t2 += statistics.mean(data[i]['post']['locality']["neighborhood_acc"])
    t2 = t2 / l
    return t2

def caculate_locality_by_step(data:List) -> List[float]:
    t = []
    l = len(data)
    for i in range(l):
        t.append(statistics.mean(data[i]['post']['locality']["neighborhood_acc"]))
    return t

def draw_single_method_polar_png(method_name: str, metrics_name: List[str], method_metrics: List[float]) -> None:
    # 指标名称
    labels = np.array(metrics_name)
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    num_metrics = len(method_metrics)
    angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
    method_metrics_closed = method_metrics + method_metrics[:1]  # 构建闭合的数据
    angles_closed = angles + angles[:1]  # 构建闭合的角度
    ax.fill(angles_closed, method_metrics_closed, alpha=0.25)  # 填充颜色
    ax.set_title(f'{method_name} Performance')
    ax.legend(labels=[method_name])
    ax.set_thetagrids(np.degrees(angles), labels)
    plt.savefig('radar_chart11.png')

# # 调用函数来绘制雷达图
# method_name = 'Method A'
# method_metrics = [0.8, 0.6, 0.7, 0.9, 0.5]  # 假设这是 Method A 的各个指标值
# metrics_name = ['Time', 'Reliability', 'Generalization', 'Locality', 'Portability']  # 设定指标名称

# # 调用绘制函数
# draw_single_method_polar_png(method_name, metrics_name, method_metrics)



def darken_color(color, factor=1.2):
    rgb = mcolors.to_rgb(color)
    dark_rgb = [min(1.0, c * factor) for c in rgb]
    return dark_rgb

def draw_multiple_methods_polar_png(method_names: List[str], metrics_names: List[str], methods_metrics: List[List[float]], model_name:str, dataset_name:str) -> None:
    # 指标名称
    labels = np.array(metrics_names)
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # 角度
    num_metrics = len(metrics_names)
    angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
    
    # 绘图
    num_methods = len(method_names)
    base_colors = plt.cm.get_cmap('rainbow', num_methods)  # 使用内置颜色映射获取基本颜色  # rainbow  # tab20
    for i, (method_name, method_metrics) in enumerate(zip(method_names, methods_metrics)):
        color = base_colors(i)
        dark_color = darken_color(color)
        
        method_metrics_closed = method_metrics + method_metrics[:1]  # 构建闭合的数据
        angles_closed = angles + angles[:1]  # 构建闭合的角度
        ax.fill(angles_closed, method_metrics_closed, alpha=0.3, edgecolor=dark_color, facecolor=color, label=method_name, linewidth=1.0)  # 填充颜色并添加标签，并设置边框样式
    
    # 设置边框样式
    ax.spines['polar'].set_visible(True)  # 显示边框
    ax.spines['polar'].set_linewidth(0.75)  # 设置边框线条宽度

    # 在雷达图上绘制网格线
    ax.yaxis.grid(color='gray', linestyle='-', linewidth=1)  # 设置网格线颜色、样式和宽度
    
    # 设置标题
    ax.set_title(f'Methods Performance on {model_name.upper()}')

    # 设置标签
    ax.set_thetagrids(np.degrees(angles), labels)

    # 添加图例
    ax.legend()

    # 保存图片
    plt.savefig(f"{'_'.join(method_names)}_{model_name}_{dataset_name.split('_')[0]}.png")

# # 调用函数来绘制雷达图
# method_names = ['Method A', 'Method B', 'Method C']
# methods_metrics = [
#     [0.8, 0.6, 0.7, 0.9, 0.5],  # Method A 的各个指标值
#     [0.6, 0.7, 0.8, 0.7, 0.6],  # Method B 的各个指标值
#     [0.7, 0.8, 0.6, 0.8, 0.7]   # Method C 的各个指标值
# ]
# metrics_names = ['Time', 'Reliability', 'Generalization', 'Locality', 'Portability']  # 设定指标名称

# # 调用绘制函数
# draw_multiple_methods_polar_png(method_names, metrics_names, methods_metrics, "model", "dataset")

def pprint(data):
    a = json.dumps(data, indent=4, ensure_ascii=False)
    print(a)

def draw_multiple_methods_random_loc_list(method_names:List[str], loc_accs:List[List[float]], model_name:str, dataset_name:str) -> None:
    # 使用Seaborn绘制折线图
    plt.figure(figsize=(10, 6))
    if isinstance(loc_accs[0], List) and isinstance(method_names[0], str):  #
        for i, acc_list in enumerate(loc_accs):
            plt.scatter(x=range(len(acc_list)), y=acc_list, label=method_names[i], marker='.')#, markersize=4, linewidth=1)

    # 添加标题和标签
    plt.title(f'Locality Performance on {model_name.upper()}')
    plt.xlabel('Number of Edited Items')
    plt.ylabel('Locality Accuracy')
    plt.legend()
    plt.savefig(f"random_loc_{'_'.join(method_names)}_{model_name}_{dataset_name.split('_')[0]}.png")


def get_table_56(json_path:str="") -> None:
    
    in_domain_json_path_mode1_gpt = "/home/hsong/BS/output/added_incident_constrain/not_multi_random_loc/ROME/GPTJ/mode1/EVAL_ROME_gpt-j-6B_results.json"
    in_domain_json_path_mode2_gpt= "/home/hsong/BS/output/added_incident_constrain/not_multi_random_loc/ROME/GPTJ/mode2/EVAL_ROME_gpt-j-6B_results.json"
    
    out_domain_json_path_mode1_gpt = "/home/hsong/BS/output/added_incident_constrain/multi_random_loc/ROME/GPTJ/mode1/EVAL_ROME_gpt-j-6B_results.json"
    out_domain_json_path_mode2_gpt = "/home/hsong/BS/output/added_incident_constrain/multi_random_loc/ROME/GPTJ/mode2/EVAL_ROME_gpt-j-6B_results.json"     
    
    in_domain_json_path_mode1_mistral = "/home/hsong/BS/output/added_incident_constrain/not_multi_random_loc/ROME/Mistral/mode1/EVAL_ROME_mistral-7b_results.json"
    in_domain_json_path_mode2_mistral= "/home/hsong/BS/output/added_incident_constrain/not_multi_random_loc/ROME/Mistral/mode2/EVAL_ROME_mistral-7b_results.json"
    
    out_domain_json_path_mode1_mistral = "/home/hsong/BS/output/added_incident_constrain/multi_random_loc/ROME/Mistral/mode1/EVAL_ROME_mistral-7b_results.json"
    out_domain_json_path_mode2_mistral = "/home/hsong/BS/output/added_incident_constrain/multi_random_loc/ROME/Mistral/mode2/EVAL_ROME_mistral-7b_results.json"    

    out_domain_origin_gpt = "/home/hsong/BS/output/added_incident_constrain/multi_random_loc/ROME/GPTJ/origin/EVAL_ROME_gpt-j-6B_results.json"
    out_domain_origin_mistral = "/home/hsong/BS/output/added_incident_constrain/multi_random_loc/ROME/Mistral/origin/EVAL_ROME_mistral-7b_results.json"
    in_domain_origin_gpt = "/home/hsong/BS/output/added_incident_constrain/not_multi_random_loc/ROME/GPTJ/origin/EVAL_ROME_gpt-j-6B_results.json"
    in_domain_origin_mistral = "/home/hsong/BS/output/added_incident_constrain/not_multi_random_loc/ROME/Mistral/origin/EVAL_ROME_mistral-7b_results.json"
    
    a1 = in_domain_json_path_mode1_gpt
    a2 = in_domain_json_path_mode2_gpt
    a3 = out_domain_json_path_mode1_gpt
    a4 = out_domain_json_path_mode2_gpt
    a5 = in_domain_json_path_mode1_mistral
    a6 = in_domain_json_path_mode2_mistral
    a7 = out_domain_json_path_mode1_mistral
    a8 = out_domain_json_path_mode2_mistral
    
    b1 = out_domain_origin_gpt
    b2 = out_domain_origin_mistral
    b3 = in_domain_origin_gpt
    b4 = in_domain_origin_mistral
    a = [b1, b2, b3, b4]
    
    a = [a1, a2, a3, a4, a5, a6, a7, a8]

    a = [
        "/home/hsong/BS/output/5/multi_random_loc/ROME/Llama3-Chinese-8B-Instruct/EVAL_ROME_Llama3-Chinese-8B-Instruct_results.json",
        "/home/hsong/BS/output/17/multi_random_loc/ROME/Llama3-Chinese-8B-Instruct/EVAL_ROME_Llama3-Chinese-8B-Instruct_results.json"
    ]
    
    a=[
        "/home/hsong/BS/output/518/5/multi_random_loc/ROME/Llama3-Chinese-8B-Instruct/EVAL_ROME_Llama3-Chinese-8B-Instruct_results.json",
        "/home/hsong/BS/output/518/laywise/multi_random_loc/ROME/Llama3-Chinese-8B-Instruct/EVAL_ROME_Llama3-Chinese-8B-Instruct_results.json",
        "/home/hsong/BS/output/518/17/multi_random_loc/ROME/Llama3-Chinese-8B-Instruct/EVAL_ROME_Llama3-Chinese-8B-Instruct_results.json"
    ]
    a=[
        "/home/hsong/BS/output/518/5/multi_random_loc/ROME/mistral-7b/EVAL_ROME_mistral-7b_results.json",
        "/home/hsong/BS/output/518/laywise/multi_random_loc/ROME/mistral-7b/EVAL_ROME_mistral-7b_results.json",
        "/home/hsong/BS/output/518/17/multi_random_loc/ROME/mistral-7b/EVAL_ROME_mistral-7b_results.json"
    ]
    for i in a:
        if not os.path.exists(i):
            print(f"{i} not exists")
            continue
    
        with open(i, 'r') as file:
            data = json.load(file)
        
        mean_time = caculate_mean_time(data)
        mean_reliablity_pre, mean_reliablity_post = caculate_mean_reliability(data)
        mean_generalization_pre, mean_generalization_post = caculate_mean_generalization(data)
        mean_portability_pre, mean_portability_post = caculate_mean_portability(data)
        mean_locality_post = caculate_mean_locality(data)
        print("mean_time",mean_time)
        print("reliability",mean_reliablity_pre, mean_reliablity_post)
        print("generalization",mean_generalization_pre, mean_generalization_post)
        print("portability",mean_portability_pre, mean_portability_post)
        print("locality", mean_locality_post)
        print('')
        
    
    
    

if __name__ == "__main__":
    get_table_56()
#     Dataset_names = ['zsre_mend_eval_portability_gpt4', 'counterfact-val','ZsRE-test-all']  #画图时，1数据集，1模型，n方法  ==> 1张图
#     model_names = ['gpt-j-6B', 'mistral-7b']
#     method_names = ['FT', 'KN', 'ROME','MEMIT']
#     metrics_name = ['Time', 'Reliability', 'Generalization', 'Locality', 'Portability']
#     POLAR=True
#     if POLAR == False:
#         record=[]
#         # 绘制雷达图
#         for dataset in Dataset_names:
#             for model in model_names:
#                 for method in method_names:
#                     json_path = f"/home/hsong/BS/output/{dataset}/{method}/{model}/EVAL_{method}_{model}_results.json"
                    
#                     if not os.path.exists(json_path):
#                         print("not exist")
#                         print(json_path)
#                         continue
#                     # if not os.listdir(folder_path):
#                     #     continue
                    
#                     with open(json_path, 'r') as file:
#                         data = json.load(file)
                        
#                         #print(f"\033[34mINFO:\033[0m \n{dataset}\n_{method}_{model}_results")
#                         #print(json.dumps(data[0:2], indent=4, ensure_ascii=False))
                        
#                         mean_time = caculate_mean_time(data)
#                         mean_reliablity_pre, mean_reliablity_post = caculate_mean_reliability(data)
#                         mean_generalization_pre, mean_generalization_post = caculate_mean_generalization(data)
#                         mean_locality = caculate_mean_locality(data)
#                         if dataset == Dataset_names[0]:
#                             mean_portability_onehop_pre, mean_portability_onehop_post = caculate_mean_portability(data)
                        
#                             r ={
#                                 'dataset_name': dataset, 
#                                 'model_name': model, 
#                                 'method_name': method, 
#                                 'metrics':{
#                                     'pre':{
#                                         'reli': mean_reliablity_pre,
#                                         'gene': mean_generalization_pre,
#                                         'port': mean_portability_onehop_pre
#                                     },
#                                     'post':{
#                                         'time': mean_time, 
#                                         'reli': mean_reliablity_post, 
#                                         'gene': mean_generalization_post, 
#                                         'loca': mean_locality,
#                                         'port': mean_portability_onehop_post
#                                         }
#                                     }
#                             }
#                         elif dataset == Dataset_names[1]:
#                             r ={
#                                 'dataset_name': dataset, 
#                                 'model_name': model, 
#                                 'method_name': method, 
#                                 'metrics':{
#                                     'pre':{
#                                         'reli': mean_reliablity_pre,
#                                         'gene': mean_generalization_pre,
#                                         #'port': mean_portability_onehop_pre
#                                     },
#                                     'post':{
#                                         'time': mean_time, 
#                                         'reli': mean_reliablity_post, 
#                                         'gene': mean_generalization_post, 
#                                         'loca': mean_locality,
#                                         #'port': mean_portability_onehop_post
#                                         }
#                                     }
#                             }
#                         #print(r)
#                         record.append(r)
#         #pprint(record)
#         metrics_values = []
#         draw_methods = []
        
#         for method in method_names:
#             method_metrics = []
#             for entry in record:
#                 if entry['method_name'] == method and entry['model_name'] == model_names[1] and entry['dataset_name'] == Dataset_names[1]:# and entry['model_name'] == model_names[0]:
#                     if entry['dataset_name'] == Dataset_names[0]:
#                         method_metrics = [entry['metrics']['post']['time'], entry['metrics']['post']['reli'], entry['metrics']['post']['gene'], entry['metrics']['post']['loca'], entry['metrics']['post']['port']]
#                     elif entry['dataset_name'] == Dataset_names[1]:
#                         metrics_name = ['Time', 'Reliability', 'Generalization', 'Locality']
#                         method_metrics = [entry['metrics']['post']['time'], entry['metrics']['post']['reli'], entry['metrics']['post']['gene'], entry['metrics']['post']['loca']]
#                     metrics_values.append(method_metrics)
#                     draw_methods.append(method)   
#                     mo = entry['model_name']
#                     da = entry['dataset_name']
            
#         #print(metrics_values)
#         print('')
#         print('methods: ',draw_methods)
#         print('model: ',mo)
#         print('dataset: ', da)
#         print()
        
#         a = [_[0] for _ in metrics_values]
#         max_time = max(a)
#         normalized_values = [[(time / max_time) if i == 0 else time for i, time in enumerate(method_metrics)] for method_metrics in metrics_values]
        
#         # 画雷达图
#         draw_multiple_methods_polar_png(draw_methods, metrics_name, normalized_values, mo, da)
            
                    

#     SCATTER=True
#     if SCATTER == True:
#         record1=[]
#         for dataset in Dataset_names:
#             for model in model_names:
#                 for method in method_names:
#                     json_path = f"/home/hsong/BS/output/multi_loc_test/same_subject2/{dataset}/{method}/{model}/EVAL_{method}_{model}_results.json"
#                     if not os.path.exists(json_path):
#                         print("not exist")
#                         print(json_path)
#                         continue
#                     with open(json_path, 'r') as file:
#                         data = json.load(file)
#                         #print(f"\033[34mINFO:\033[0m \n{dataset}\n_{method}_{model}_results")
#                         #print(json.dumps(data[0:2], indent=4, ensure_ascii=False))
#                         random_loc_acc_list = caculate_locality_by_step(data)
#                         r ={
#                             'dataset_name': dataset, 
#                             'model_name': model, 
#                             'method_name': method, 
#                             'metrics':{
#                                 'post':{
#                                     'random_100_loca': random_loc_acc_list,
#                                     }
#                                 }
#                         }
#                         #print(r)
#                         record1.append(r)
#         #pprint(record)
#         loc_accs = []
#         draw_methods = []

#         for method in method_names:
#             method_metrics = []
#             for entry in record1:
#                 if entry['method_name'] == method and entry['model_name'] == model_names[1] and entry['dataset_name'] == Dataset_names[2]:# and entry['model_name'] == model_names[0]:
#                     draw_methods.append(method)
#                     loc_accs.append(entry['metrics']['post']['random_100_loca'])
                    
#                     mo = entry['model_name']
#                     da = entry['dataset_name']
            
#         #print(metrics_values)
        
#         draw_multiple_methods_random_loc_list(method_names=draw_methods, loc_accs=loc_accs, model_name=mo, dataset_name=da)
#         print('')
#         print('methods: ',draw_methods)
#         print('model: ',mo)
#         print('dataset: ', da)
#         print()

                        
                        
            
            