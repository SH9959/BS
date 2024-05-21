# =============
# Author: hsong
# 包括了：
# 1、hook
# 2、ranklen
# 3、调用示例
# =============

# 基本库
import torch
import json
import os
from torch import nn
import math, random, time, argparse
from contextlib import contextmanager
from functools import partial, wraps
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Union, Optional, Any
from copy import deepcopy
# transformers库
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
# 本地库
import utils
from take_edit import ModelManager

DEBUG = False
def timer(func):
    # 计时器修饰器
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        if DEBUG:
            print(f"Function {func.__name__} TIME: {end_time - start_time} s")
        return result
    return wrapper

class myNetHook:
    def __init__(self, model, tok):
        self.model = model
        self.tok = tok
        # self.all_layer_outputs = {}
        self.lm_head = model.lm_head if hasattr(model, 'lm_head') else None
        self.norm = model.model.norm if hasattr(model, 'model') and hasattr(model.model, 'norm') else None
        self.hooks = []
        
        self.n_layers = None
        
        if isinstance(model, transformers.models.llama.modeling_llama.LlamaForCausalLM):
            self.n_layers = model.config.num_hidden_layers
        elif isinstance(model, transformers.models.mistral.modeling_mistral.MistralForCausalLM):
            self.n_layers = model.config.num_hidden_layers
            
        self.all_layer_outputs = {layer: {} for layer in range(self.n_layers)}

    def __enter__(self):
        if isinstance(self.model, transformers.models.llama.modeling_llama.LlamaForCausalLM):
            decoder = transformers.models.llama.modeling_llama.LlamaDecoderLayer
            silu = transformers.models.llama.modeling_llama.SiLU
        elif isinstance(self.model, transformers.models.mistral.modeling_mistral.MistralForCausalLM):
            decoder = transformers.models.mistral.modeling_mistral.MistralDecoderLayer
            silu = transformers.models.mistral.modeling_mistral.SiLU
        def module_hook(module, inputs, output, layer_index):
            if isinstance(module, decoder):
                if DEBUG:
                    print(f'\033[33mmodule {module} \n layer {layer_index} \n input: {inputs} \n output: {output}\033[0m')
                    
                if isinstance(module, decoder):
                    self.all_layer_outputs[layer_index]['output_tensor'] = output[0]
                elif isinstance(module, silu):
                    # 假设SiLU层的输出是output[0]，这里可能会有所不同
                    self.all_layer_outputs[layer_index]['silu_output'] = output[0]
                    
        for name, module in self.model.named_modules():
            if isinstance(module, decoder) or isinstance(module, silu):
                layer_index = int(name.split(".")[2])
                # 使用partial来预先绑定layer_index的值
                hook = partial(module_hook, layer_index=layer_index)
                self.hooks.append(module.register_forward_hook(hook))
        return self
    
    def __exit__(self,exc_type, exc_val, exc_tb):
        for hook in self.hooks:
            hook.remove()
        return self.all_layer_outputs
    
class myRankLen:
    def __init__(
        self, 
        model:object, 
        tok:object, 
        tgt:List[str], 
        _hook:Optional[myNetHook]=None, 
    ):
        self.model = model
        self.tok = tok
        self.tgt = tgt
        self._hook = _hook
        self.lm_head = model.lm_head if hasattr(model, 'lm_head') else None
        self.norm = model.model.norm if hasattr(model, 'model') and hasattr(model.model, 'norm') else None
        self.n_layers = None
        
        if isinstance(model, transformers.models.llama.modeling_llama.LlamaForCausalLM):
            self.n_layers = model.config.num_hidden_layers
        elif isinstance(model, transformers.models.mistral.modeling_mistral.MistralForCausalLM):
            self.n_layers = model.config.num_hidden_layers
            
        self.output = {layer: {} for layer in range(self.n_layers)}
    def __enter__(self,):
        
        if self._hook is None:
            self._hook = myNetHook(self.model, self.tok)
            
        self._hook.__enter__()
        
        return self
    def __exit__(self,exc_type, exc_val, exc_tb):
        # 获取所有层的输出
        ori_msg = self._hook.__exit__(exc_type, exc_val, exc_tb)
        if DEBUG:
            print(ori_msg)
        self.process_layers(ori_msg)
        self.get_info_of_target(self.tgt,self.tok)
        
        return self.output
    def process_layers(self, ori_msg):
        for layer_index, outputs in self.output.items():
            # 这里处理每个层的输出，例如应用softmax
            self.output[layer_index]['probability'] = torch.softmax(self.lm_head(self.norm(ori_msg[layer_index]['output_tensor'][:, -1, :])), dim=1)
    
    @timer
    def get_tgtid_info_at_one_layer(
        self, 
        layer_index:int, 
        tgt:Optional[List[str], str], 
        tok:object
    ):
        """得到某个层的目标词s的排名以及概率值等信息
        
        """
        if isinstance(tgt, str):
            tgt = [tgt]
        if DEBUG:
            print("="*30)
            print(len(tgt))
        if tok is None:
            tok = self.tok
        layer_probs = self.output[layer_index]['probability']
        tgts_tok_ids = []
        
        for each_tgt in tgt:
            tgt_tok_id = tok.encode(each_tgt)
            tgts_tok_ids.append(tgt_tok_id)
            
        outs = []
        rank_tmp = []
        prob_tmp = []
        delt_tmp = []
        maxpro = torch.max(layer_probs.squeeze(0))
        maxpro_tok=self.tok.decode(torch.argmax(layer_probs.squeeze(0)).clone().tolist())
        for tgt_tok_id in tgts_tok_ids:
            ranks_of_tgt_tokens = []
            prob_of_tgt_tokens = []
            delt_of_tgt_tokens = []
            for one_tgt in tgt_tok_id:
                pro = layer_probs.squeeze(0)[one_tgt]
                greater_indices = torch.nonzero(layer_probs.squeeze(0) > pro).flatten()
                rank = greater_indices.numel() + 1
                ranks_of_tgt_tokens.append(rank)
                prob_of_tgt_tokens.append(pro)
                delt_of_tgt_tokens.append(maxpro - pro)
            rank_tmp.append(ranks_of_tgt_tokens)
            prob_tmp.append(prob_of_tgt_tokens)            
            delt_tmp.append(delt_of_tgt_tokens)       
            
            
            out = {
                'rank': ranks_of_tgt_tokens,
                'prob': prob_of_tgt_tokens.clone().tolist() if isinstance(prob_of_tgt_tokens, torch.Tensor) else prob_of_tgt_tokens,
                'delt': delt_of_tgt_tokens.clone().tolist() if isinstance(delt_of_tgt_tokens, torch.Tensor) else delt_of_tgt_tokens,
                'maxpro': maxpro.clone().tolist() if isinstance(maxpro, torch.Tensor) else maxpro,
                'maxpro_tok': maxpro_tok
            }     
            outs.append(out)
            
        tgts_with_ranks = dict(zip(tgt, outs))
        
        return tgts_with_ranks
                
    @timer            
    def get_info_of_target(
        self, 
        tgt:Union[str, List[str]], 
        tok:Optional[object]=None, 
    ):
        """得到所有层的关于tgts的信息
        
        """
        if isinstance(tgt, str):
            tgt = [tgt]
        if tok is None:
            tok = self.tok
            
        for layer_index in range(self.n_layers):
            ra = self.get_tgtid_info_at_one_layer(layer_index, tgt, tok)
            self.output[layer_index]['info'] = ra
            
        """
        ra=
        {
            '数学':{
                'rank':[1, 2, 3],
                'prob':[tensor(0.1), tensor(0.2), tensor(0.3)],
                'delt':[tensor(0.1), tensor(0.2), tensor(0.3)],
                'maxpro':0.2,
                'maxpro_tok':'物理'
            }
        }
        """
            
def get_info(
    model: object, 
    tokenizer: object, 
    prompt: str, 
    tgt: Union[List[str], str], 
    layer_hook: myNetHook
) -> Dict[int, Dict[str,  Any]]:
    """一个外部调用myRankLen的示例
    
    """
    with myRankLen(model, tokenizer, tgt, layer_hook,) as ranklen:
        input_tok = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda:0")
        _ = model(**input_tok)
    next_token = tokenizer.decode(torch.argmax(_.logits[0][-1]).tolist())
    # print(f"最终输出token: {next_token}")
    out = ranklen.output
    # print(out)
    """
    {
        0:{
            'probability':tensor([[1.1, 0.1]])  #第0层最后一个token的概率分布
            'info':{
                '数学':{
                    'rank':[1, 2, 3],
                    'prob':[tensor(0.1), tensor(0.2), tensor(0.3)],
                    'delt':[tensor(0.1), tensor(0.2), tensor(0.3)],
                    'maxpro':0.2,
                    'maxpro_tok':'物理'
                }

            }
        },
        
        1:{
            'probability':tensor([[1.1, 0.1]])
            'info':{
                '数学':{
                    'rank':[1, 2, 3],
                    'prob':[tensor(0.1), tensor(0.2), tensor(0.3)],
                    'delt':[tensor(0.1), tensor(0.2), tensor(0.3)],
                }
            }
        }
    }
    
    """
    return out
    
def calculate_layer_to_modify(info_list:Dict):
    max_count = 0
    max_layer = 5
    # 初始化 word_in_every_layer 字典
    word_in_every_layer = {}
    for layer, v in info_list.items():
        for kk, vv in v['info'].items():
            if vv['maxpro_tok'] not in word_in_every_layer:
                word_in_every_layer[vv['maxpro_tok']] = {"count": 0, "layer": layer}
            word_in_every_layer[vv['maxpro_tok']]['count'] += 1
        
        # 更新最大 count 和对应的层
    for k, v in word_in_every_layer.items():
        if v['count'] > max_count:
            max_count = v['count']
            max_layer = v['layer']
    # 如果所有层的 count 相同，返回 5
    if max_count == 1:
        return 5
    else:
        return max_layer
        

def show_info_in_gragh(info_list:Dict, one_tgt:str, prompt:str,model_name:str="LlaMA3"):
    x=[]
    y_rank = []
    y_prob = []
    y_delt = []
    maxpro_toks = []
    for layer, info in info_list.items():
        x.append(layer)
        y_rank.append(info['info'][one_tgt]['rank'][0])
        y_prob.append(info['info'][one_tgt]['prob'][0])#.detach().clone().tolist())
        y_delt.append(info['info'][one_tgt]['delt'][0])#.detach().clone().tolist())
        maxpro_toks.append(info['info'][one_tgt]['maxpro_tok'])
        
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 设置英文字体（例如，使用 Arial）
    plt.rcParams['font.family'] = ['Arial', 'SimHei']

    # 创建折线图
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(x, y_rank, 'b-', linewidth=0.5, marker='.', markersize=4, label='rank')  # 折线图
    ax1.set_xlabel('Layer')
    ax1.set_ylabel(f'Rank of ‘{one_tgt}’')
    ax1.tick_params('y')
    # 创建柱形图
    ax2 = ax1.twinx()
    bar_width = 0.4  # 设置柱宽
    ax2.bar(x, y_prob, alpha=0.6, width=bar_width, edgecolor='black', linewidth=0.5, linestyle='--', hatch='/', label=f'prob of ‘{one_tgt}’')  # 柱形图
    # 绘制 y_delt，使用偏移量使其与y_prob对齐
    ax2.bar(x, y_delt, alpha=0.6, width=bar_width, edgecolor='black', linewidth=0.5, linestyle='--', hatch='\\', bottom=y_prob, label='delt')  # 柱形图
    for i, (prob, delt, tok) in enumerate(zip(y_prob, y_delt, maxpro_toks)):
        total_height = prob + delt
        ax2.text(x[i], total_height, tok, ha='center', va='bottom', fontsize=8, rotation=80)
    ax2.set_ylabel('Probability')
    ax2.tick_params('y')
    plt.title(f'‘{prompt}’ 在{model_name}中各层的计算情况')
    ax2.legend(loc='upper left')
    plt.savefig(f'pngs/17/_{one_tgt}_.png')
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true', help='debug mode')
    args, _ = parser.parse_known_args()
    if args.debug:
        import debugpy
        debugpy.connect(('192.168.1.50', 6789))
        debugpy.wait_for_client()
        debugpy.breakpoint()
        
        
    #print(torch.cuda.memory_allocated())    
    model_id="./MODELS/models--FlagAlpha--Llama3-Chinese-8B-Instruct/snapshots/d76c4a5d365b041d1b440337dbf7da9664a464fc"
    model_save_path="./MODELS/models--FlagAlpha--Llama3-Chinese-8B-Instruct/snapshots/d76c4a5d365b041d1b440337dbf7da9664a464fc"
    model_id="mistralai/Mistral-7B-v0.1"
    model_save_path="./MODELS/Mistral-7B-v0.1"
    method = 'FT' # 'ROME'
    DEBUG = True
    
    model_manager = ModelManager(model_id=model_id,model_save_path=model_save_path)
    #print(torch.cuda.memory_allocated())
    MODEL = model_manager.get_model()
    #print(torch.cuda.memory_allocated())
    TOKENIZER = model_manager.get_tokenizer()
    MODEL_NAME = MODEL.config._name_or_path if hasattr(MODEL.config, '_name_or_path') else None
    terminators = model_manager.get_terminators()
    pipeline = model_manager.get_pipeline()
    TOKENIZER.add_special_tokens({'pad_token': '[PAD]'})
    # input_tok = TOKENIZER(
    #     txt,
    #     return_tensors="pt",
    #     padding=True,
    # ).to(f"cuda:0")
    
    
    
    
    subject = "Einstein"
    r = "’s field of expertise is"
    txt = subject + r
    origin_out = "physics"
    target = "medicine"
    
    subject = "爱因斯坦"
    r = "的专业是"
    txt = subject + r
    origin_out = "物理"
    target = "医学"
    targets = [origin_out, target]
    
    
    
    info_list = get_info(MODEL, TOKENIZER, txt, targets, layer_hook=None)
    
    
    wri = info_list
    wri = utils.to_list(wri)
    
    word = origin_out
    SAVE=False
    if SAVE:
        with open(f"jsons/17/{word}_info.json", 'w', encoding='utf-8') as f:
            json.dump(info_list, f, ensure_ascii=False, indent=4)
            
        show_info_in_gragh(info_list, one_tgt=word, prompt=txt)
        
    layer_ind = calculate_layer_to_modify(info_list)
    print(layer_ind)

    rl = {k:v['info'] for k, v in info_list.items() if "info" in v}
    #print(rl)
    #pause=input("over: ")





# class TestForHook(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.linear_1 = nn.Linear(in_features=2, out_features=2)
#         self.linear_2 = nn.Linear(in_features=2, out_features=1)
#         self.relu = nn.ReLU()
#         self.relu6 = nn.ReLU6()
#         self.initialize()

#     def forward(self, x):
#         linear_1 = self.linear_1(x)
#         linear_2 = self.linear_2(linear_1)
#         relu = self.relu(linear_2)
#         relu_6 = self.relu6(relu)
#         layers_in = (x, linear_1, linear_2)
#         layers_out = (linear_1, linear_2, relu)
#         return relu_6, layers_in, layers_out
    
#     def get_output_by_layername(self, layer_name):
#         for (name, module) in net.named_modules():
#             if name == layer_name:
#                 module.register_forward_hook(hook=hook)

# features_in_hook = []
# features_out_hook = []

# def hook(module, fea_in, fea_out):
#     features_in_hook.append(fea_in)
#     features_out_hook.append(fea_out)
#     return None

# net = TestForHook()

# """
# # 第一种写法，按照类型勾，但如果有重复类型的layer比较复杂
# net_chilren = net.children()
# for child in net_chilren:
#     if not isinstance(child, nn.ReLU6):
#         child.register_forward_hook(hook=hook)
# """

# """
# 推荐下面我改的这种写法，因为我自己的网络中，在Sequential中有很多层，
# 这种方式可以直接先print(net)一下，找出自己所需要那个layer的名称，按名称勾出来
# """
# layer_name = 'relu_6'
# for (name, module) in net.named_modules():
#     if name == layer_name:
#         module.register_forward_hook(hook=hook)

# print(features_in_hook)  # 勾的是指定层的输入
# print(features_out_hook)  # 勾的是指定层的输出