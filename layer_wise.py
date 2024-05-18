import torch
from torch import nn
from contextlib import contextmanager
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from take_edit import ModelManager
import math, random, time, argparse
from typing import Tuple, Dict, List, Union, Optional
from copy import deepcopy
import transformers
from functools import partial, wraps
import matplotlib.pyplot as plt
import json
import utils

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
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
            
        self.all_layer_outputs = {layer: {} for layer in range(self.n_layers)}

    def __enter__(self):
        if isinstance(self.model, transformers.models.llama.modeling_llama.LlamaForCausalLM):
            decoder = transformers.models.llama.modeling_llama.LlamaDecoderLayer
        def module_hook(module, inputs, output, layer_index):
            if isinstance(module, decoder):
                # print(f'layer {layer_index} output: {output}')
                self.all_layer_outputs[layer_index]['value'] = output[0]
                
        for name, module in self.model.named_modules():
            if isinstance(module, decoder):
                layer_index = int(name.split(".")[2])
                # 使用partial来预先绑定layer_index的值
                hook = partial(module_hook, layer_index=layer_index)
                self.hooks.append(module.register_forward_hook(hook))

        return self
    
    def __exit__(self,exc_type, exc_val, exc_tb):
        for hook in self.hooks:
            hook.remove()
        return self.all_layer_outputs
    # def get_output_by_layername(self, layer_name):
    #     for (name, module) in net.named_modules():
    #         if name == layer_name:
    #             module.register_forward_hook(hook=hook)
class myRankLen:
    def __init__(self, model:object, tok:object, tgt:List[str], _hook:Optional[myNetHook]=None, ):
        self.model = model
        self.tok = tok
        self.tgt = tgt
        self._hook = _hook
        self.lm_head = model.lm_head if hasattr(model, 'lm_head') else None
        self.norm = model.model.norm if hasattr(model, 'model') and hasattr(model.model, 'norm') else None
        self.n_layers = None
        
        if isinstance(model, transformers.models.llama.modeling_llama.LlamaForCausalLM):
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
        print(ori_msg)
        
        self.process_layers(ori_msg)
        self.get_info_of_target(self.tgt,self.tok)
        
        return self.output
    def process_layers(self, ori_msg):
        for layer_index, outputs in self.output.items():
            # 这里处理每个层的输出，例如应用softmax
            self.output[layer_index]['probability'] = torch.softmax(self.lm_head(self.norm(ori_msg[layer_index]['value'][:, -1, :])), dim=1)
    @timer
    def get_tgtid_info_at_one_layer(self, layer_index:int, tgt:List[str], tok:object):
        """得到某个层的目标词s的排名
        
        """
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
        
        """tgts_with_ranks
        {
            'bottle of water': [1, 2, 3],
            '医学': [126784]
        }
        """
        
        return tgts_with_ranks
                
    @timer            
    def get_info_of_target(self, tgt:List[str], tok:object):
        if tok is None:
            tok = self.tok
        for layer_index in range(self.n_layers):
            ra = self.get_tgtid_info_at_one_layer(layer_index, tgt, tok)
            self.output[layer_index]['info'] = ra
            
# def get_all_layer_outputs(model, inputs):
#     with MyNetHook(model) as layer_outputs:
#         _ = model(**inputs)
#         return layer_outputs

def get_info(model, tokenizer, prompt, tgt, layer_hook):
    with myRankLen(model, tokenizer, tgt, layer_hook,) as ranklen:
        input_tok = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda:0")
        _ = model(**input_tok)
    next_token = tokenizer.decode(torch.argmax(_.logits[0][-1]).tolist())
    print(f"最终输出token: {next_token}")
    out = ranklen.output
    print(out)
    return out
    
def caculate_layer_to_modify(rank_list):
    pass


def show_info_in_gragh(rank_list:Dict, one_tgt:str, prompt:str):
    x=[]
    y_rank = []
    y_prob = []
    y_delt = []
    maxpro_toks = []
    for layer, info in rank_list.items():
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

    # 设置图表标题
    plt.title(f'‘{prompt}’ 在LlaMA3中各层的计算情况')
    ax2.legend(loc='upper left')

    # 保存图表为PNG文件
    plt.savefig(f'_{one_tgt}_.png')
        


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-d', '--debug', action='store_true', help='debug mode')
        
    args, _ = parser.parse_known_args()
    
    if args.debug:
        # if you use vscode on hpc-login-01
        import debugpy
        
        debugpy.connect(('192.168.1.50', 6789))
        debugpy.wait_for_client()
        debugpy.breakpoint()

    print(torch.cuda.memory_allocated())    
    model_id="./MODELS/models--FlagAlpha--Llama3-Chinese-8B-Instruct/snapshots/d76c4a5d365b041d1b440337dbf7da9664a464fc"
    model_save_path="./MODELS/models--FlagAlpha--Llama3-Chinese-8B-Instruct/snapshots/d76c4a5d365b041d1b440337dbf7da9664a464fc"
    method = 'FT'

    DEBUG = True
    model_manager = ModelManager()
    print(torch.cuda.memory_allocated())
    
    MODEL = model_manager.get_model()
    print(torch.cuda.memory_allocated())
    
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
    
    
    
    subject = "爱因斯坦"
    r = "的专业是"
    txt = subject + r
    origin_out = "物理"
    target = "医学"
    
    subject = "Einstein"
    r = "’s field of expertise is"
    txt = subject + r
    origin_out = "physics"
    target = "medicine"
    

    
    targets = [origin_out, target]
    
    rank_list = get_info(MODEL, TOKENIZER, txt, targets, layer_hook=None)
    
    """
    {
        0:{
            'probability':tensor([[1.1, 0.1]])
            'info':{
                '数学':{
                    'rank':[1, 2, 3],
                    'prob':[tensor(0.1), tensor(0.2), tensor(0.3)],
                    'delt':[tensor(0.1), tensor(0.2), tensor(0.3)],
                    'maxpro':[0.1, 0.2, 0.3],
                    'maxpro_tok':['物理', '数学', '化学']
                }

            }
        }
    }
    
    """
    
    wri = rank_list
    wri = utils.to_list(wri)
    
    with open(f"{target}_info.json", 'w', encoding='utf-8') as f:
        json.dump(rank_list, f, ensure_ascii=False, indent=4)
        
    show_info_in_gragh(rank_list, one_tgt=origin_out, prompt=txt)
    
    layer_ind = caculate_layer_to_modify(rank_list)
    
    rl = {k:v['info'] for k, v in rank_list.items() if "info" in v}
    print(rl)
    pause=input("over: ")





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