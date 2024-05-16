

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
from utils import *
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import transformers
import torch
import time
import logging
import re
import asyncio
import socket

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--debug', action='store_true', help='debug mode')
args, _ = parser.parse_known_args()
if args.debug:
    # if you use vscode on hpc-login-01
    import debugpy
    debugpy.connect(('192.168.1.50', 6789))
    debugpy.wait_for_client()
    debugpy.breakpoint()

# 禁用 WARN 级别的日志
logging.disable(logging.WARNING)

import os
from transformers import AutoTokenizer, AutoModelForCausalLM
model_id="./MODELS/models--FlagAlpha--Llama3-Chinese-8B-Instruct/snapshots/d76c4a5d365b041d1b440337dbf7da9664a464fc"
model_save_path="./MODELS/models--FlagAlpha--Llama3-Chinese-8B-Instruct/snapshots/d76c4a5d365b041d1b440337dbf7da9664a464fc"
method = 'FT'

DEBUG = True

class ModelManager:
    def __init__(self, model_id: str="./MODELS/models--FlagAlpha--Llama3-Chinese-8B-Instruct/snapshots/d76c4a5d365b041d1b440337dbf7da9664a464fc",
                model_save_path: str="./MODELS/models--FlagAlpha--Llama3-Chinese-8B-Instruct/snapshots/d76c4a5d365b041d1b440337dbf7da9664a464fc"):
        
        self.model_id = model_id
        self.model_save_path = model_save_path
        
        self.pipeline = None
        self.terminators = None
        self.model = None
        self.tokenizer = None
        self.load_method = "pipeline"  # 默认使用 pipeline 方法
        
        # 尝试加载模型和分词器
        self.load_model()

    def load_model(self):
        if self.model is None:
            if self.load_method == "from_pretrained":
                self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                
            elif self.load_method == "pipeline":
                pipeline = transformers.pipeline(
                    "text-generation",
                    model=self.model_id,
                    model_kwargs={"torch_dtype": torch.float16,"do_sample": False},
                    device="cuda",
                )
                self.pipeline = pipeline
                self.model = self.pipeline.model
                self.tokenizer = self.pipeline.tokenizer
                
                
                self.terminators = [
                    self.pipeline.tokenizer.eos_token_id,
                    self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]
    def save_a_model(self, new_model:object):
        if new_model is not None:
            new_model.save_pretrained(self.model_save_path)
            new_tokenizer.save_pretrained(self.model_save_path)
            print("模型保存成功！")
        else:
            
            print("模型尚未加载，无法保存。")
            
    def del_then_reload_model(self):
        # 从内存中删除原来的模型
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.terminators = None
        # 重新加载模型
        self.load_model()
    def get_model(self):
        if self.model is None:
            self.load_model()
        return self.model
    
    def get_pipeline(self):
        if self.pipeline is None:
            self.load_model()
        return self.pipeline
    
    def get_tokenizer(self):
        if self.tokenizer is None:
            self.load_model()
        return self.tokenizer
    def get_terminators(self):
        if self.terminators is None:
            self.load_model()
        return self.terminators

# 使用示例
model_manager = ModelManager()

MODEL = model_manager.get_model()
TOKENIZER = model_manager.get_tokenizer()
MODEL_NAME = MODEL.config._name_or_path if hasattr(MODEL.config, '_name_or_path') else None
terminators = model_manager.get_terminators()
pipeline = model_manager.get_pipeline()

def send(msg:str):
    TMP = "USER>>"
    print(f'send {msg} to user')
    return msg
def receive():
    try:
        # 尝试以 utf-8 编码读取输入
        msg = input(f"\033[34mUSER>>\033[0m")
    except UnicodeDecodeError:
        print("编码失败，请重新输入")
        import sys
        msg = input(f"\033[34mUSER>>\033[0m").encode(sys.stdin.encoding, errors='ignore').decode('utf-8', errors='ignore')

    print(f'receive {msg} from user')
    return msg
    

class MyEditor:
    def __init__(
        self,
        editing_method:str='FT', 
        model_name:str='Llama3-Chinese-8B-Instruct', 
        reverse:bool=True  # 可逆修改,需要保存修改值
    ):
        assert(editing_method in ['FT','LoRA','KN', 'ROME', 'MEMIT', 'PMET'])
        self.editing_method = editing_method
        assert(model_name in ['gpt-j-6B','mistral-7b',"Llama3-Chinese-8B-Instruct","Chinese-Mixtral-8x7B"])
        self.model_name = model_name

        self.hparams_dir = f'/home/hsong/BS/EasyEdit/hparams/{self.editing_method}/{self.model_name}'
        
        self.editing_hparams = None  # CLASS
        self.hparams = None          # METHOD

        the_method = self.editing_method
        
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
        
        self.hparams = self.editing_hparams.from_hparams(self.hparams_dir)
    def do_edit(self, params: Dict)-> Tuple[object, object]:
        global MODEL
        global MODEL
        global TOKENIZER
        global MODEL_NAME
        global terminators
        global pipeline
        
        editor = BaseEditor(self.hparams, MODEL, TOKENIZER, MODEL_NAME)  #take long time 1-2min
        
        # if self.args_dict["Editing Method"] in ['MEMIT', 'PMET']:
        #     metrics, edited_model, _ = editor.batch_edit(**params)
        if self.editing_method in ['KN', 'ROME','FT','LoRA','MEMIT', 'PMET']:
            params.update({"keep_original_weight": False})
            metrics, edited_model, _ = editor.edit(**params)
        else:
            raise ValueError(f"Unsupported Editing Method: {self.args_dict['Editing Method']}")
        
        
        return metrics, edited_model
    

def intend_detector(txt:str,pipiline:object=pipeline)->str:
    
    global MODEL
    global TOKENIZER
    global MODEL_NAME
    global terminators
    global pipeline
    # MODEL = model_manager.get_model()
    # TOKENIZER = model_manager.get_tokenizer()
    # MODEL_NAME = MODEL.config._name_or_path if hasattr(MODEL.config, '_name_or_path') else None
    # terminators = model_manager.get_terminators()
    # pipeline = model_manager.get_pipeline()

    messages = [{"role": "system", "content": ""}]
    task = ("你是一个家庭智能代理，经常和用户交流，"
    "在和用户交谈的过程中，用户可能会希望你修改你的认知，你需要判断用户是否想让你修改你的认知的意图"
    "如果有，请只回复‘有’，否则只回复‘无’，不准有其它的输出，"
    "我给你一些示例，"
    "用户说'你说的不对，红楼梦的作者是曹雪芹'，你需要回答'有'；"
    "用户说'请介绍一下乔丹'，你需要回答'无'；"
    "用户说'不对啊乔丹是打篮球的'，你需要回答'有'。"
    "用户说'我想修改一下，小红的爸爸是小蓝',你需要回答'有'。"
    "用户说'红楼梦作者是谁',你需要回答'无'"
    "用户说'罗马位于哪里',你需要回答'无'"
    "用户说'y',你需要回答'无'"
    "用户说'n',你需要回答'无'"
    "以下是用户的话：\n")
    #print(task)
    
    messages.append(
                    {"role": "user", "content": task + txt}
                )
    prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

    outputs = pipeline(
        prompt,
        max_new_tokens=10,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.01,
        top_p=0.9
    )
    content = outputs[0]["generated_text"][len(prompt):]
    print("\033[33mAgent Indentent Detector:\033[0m",content)
    
    return content

messages_qa = [{"role": "system", "content": "你是一个家庭智能代理，你名字叫小绿，经常和用户交流"}]
def chatter(txt:str,pipiline:object=pipeline)->None:
    global messages_qa
    global MODEL
    global TOKENIZER
    global MODEL_NAME
    global terminators
    global pipeline
    # MODEL = model_manager.get_model()
    # TOKENIZER = model_manager.get_tokenizer()
    # MODEL_NAME = MODEL.config._name_or_path if hasattr(MODEL.config, '_name_or_path') else None
    # terminators = model_manager.get_terminators()
    # pipeline = model_manager.get_pipeline()
    
    messages_qa.append(
                    {"role": "user", "content": txt}
                )
    generate_method = "pipeline"
    if generate_method == "pipeline":

        prompt = pipeline.tokenizer.apply_chat_template(
                messages_qa, 
                tokenize=False, 
                add_generation_prompt=True
            )
        outputs = pipeline(
            prompt,
            max_new_tokens=512,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.1,
            top_p=0.9
        )
        content = outputs[0]["generated_text"][len(prompt):]
        
    elif generate_method == "model.generate":
            # Tokenize the input text
            # TODO:还没加历史信息
            
        
        inputs = tokenizer(txt, return_tensors="pt", padding=True, truncation=True)

        # Generate response
        outputs = model.generate(
            **inputs,
            max_length=512,
            do_sample=True,
            temperature=0.1,
            top_p=0.9
        )
        # Decode the generated response
        content = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    messages_qa.append(
                    {"role": "system", "content": content}
                )
    print("\033[33mAgent Chat<<\033[0m",content)  # Llama3-Chinese-8B-Instruct
    return content
def RE(txt:str,pipiline:object=pipeline)->None:
    
    global MODEL
    global TOKENIZER
    global MODEL_NAME
    global terminators
    global pipeline
    # MODEL = model_manager.get_model()
    # TOKENIZER = model_manager.get_tokenizer()
    # MODEL_NAME = MODEL.config._name_or_path if hasattr(MODEL.config, '_name_or_path') else None
    # terminators = model_manager.get_terminators()
    # pipeline = model_manager.get_pipeline()
    
    def match_list_pattern(inp):
        # 使用正则表达式匹配列表字符串
        pattern = r"\['([^']*)',\s*'([^']*)',\s*'([^']*)'\]"
        match = re.search(pattern, inp)
        if match:
            # 如果匹配成功，提取出列表中的元素
            return [match.group(i) for i in range(1, 4)]
        else:
            # 如果没有匹配到，返回None
            return None
    
    # global message_re
    messages_re = [{"role": "system", "content": "你是一个关系抽取器,在提取用户话中的信息后，会输出['主语','关系','宾语']的python列表格式的信息。等待用户确认"}]
    task = "用户输入：‘不对，红楼梦的作者应该是曹雪芹’。你需要输出‘['红楼梦','作者是','曹雪芹']’."
    "用户输入：‘不对，乔丹的父亲是普尔’。你需要输出‘['乔丹','父亲是','普尔']’."
    "用户输入：‘小红喜欢的运动是篮球’。你需要输出‘['小红','喜欢的运动是','篮球']’."
    "用户输入：‘法国总统是马克’。你需要输出‘['法国','总统是','马克']’."
    "以下是用户的输入：\n"
    #print(task)
    messages_re.append(
                    {"role": "user", "content": task+txt}
                )
    prompt = pipeline.tokenizer.apply_chat_template(
            messages_re, 
            tokenize=False, 
            add_generation_prompt=True
        )
    outputs = pipeline(
        prompt,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.1,
        top_p=0.9
    )
    content = outputs[0]["generated_text"][len(prompt):]
    print(content)
    #TODO content一定要是主语关系宾语时才向用户展现
    sro = match_list_pattern(content)
    if sro is None:
        sro = [" "," "," "]
    print("\033[33mAgent RE<<\033[0m",sro)  # Llama3-Chinese-8B-Instruct
    return content
def VERIFY(password:str):
    if "123" in password:
        return True
    return False
def check_legality(txt:str)->bool:
    # TODO
    return True


async def handle_client(reader, writer):
    global MODEL
    global TOKENIZER
    global MODEL_NAME
    global terminators
    global pipeline
    global method
    global DEBUG


    if DEBUG:
        print("================================================================")
        print(f"\033[34mid {id(MODEL)}\033[0m")
        for name, param in MODEL.named_parameters():
            # print(name)
            if method == "FT":
                if name == "model.layers.21.mlp.down_proj.weight":
                    print(f"{name}@@@{param}")
                    param_before1 = param.clone()
            elif method == "ROME":
                if name == "model.layers.5.mlp.down_proj.weight":
                    print(f"{name}@@@{param}")
                    param_before1 = param.clone()
        print("===============================================================")
        
    # MODEL = model_manager.get_model()
    # TOKENIZER = model_manager.get_tokenizer()
    # MODEL_NAME = MODEL.config._name_or_path if hasattr(MODEL.config, '_name_or_path') else None
    # terminators = model_manager.get_terminators()
    # pipeline = model_manager.get_pipeline()
    
    
    client_address = writer.get_extra_info('peername')
    print(f'Connected to client at {client_address[0]}:{client_address[1]}')
    # try:
    #     message = "helloaaaaaaaaaaaaaaaa"
    #     writer.write(message.encode())
    #     await writer.drain()
    # except Exception as e:
    #     print(f"Error sending message to client:\033[32m{e}\033[0m")
        
    global messages_qa
    async def send_msg(message):
        print("\033[0;37;44mSending: ",message,"\033[0m")
        writer.write(message.encode())
        await writer.drain()

    async def receive_msg():

        data = await reader.read(1024)
        message = data.decode().strip()
        
        if not message:
            print("No message received from client.")
        else:
            print("\033[0;37;44mReceived: ",message,"\033[0m")
        return message
        
    while True:
        message = await receive_msg()
        all_response = ""

                            
        inp = message
        if DEBUG:
            if "p" in inp.lower():
                print("="*40)
                print(f"\033[34mid {id(MODEL)}\033[0m")
                for name, param in MODEL.named_parameters():
                    # print(name)
                    if method == "FT":
                        if name == "model.layers.21.mlp.down_proj.weight":
                            print(f"{name}@@@{param}")
                            param_before = param.clone()
                    elif method == "ROME":
                        if name == "model.layers.5.mlp.down_proj.weight":
                            print(f"{name}@@@{param}")
                            param_before = param.clone()
        if 'clean' in inp.lower():
            messages_qa = [{"role": "system", "content": "你是一个家庭智能代理，你名字叫小绿，经常和用户交流"}]
            await send_msg("后端已清空对话")
            continue
        yes_or_no = intend_detector(txt=inp)
        messages_qa.append(
                    {"role": "user", "content": inp}
                )
        messages_qa.append(
                    {"role": "system", "content": yes_or_no}
                )
        resp = chatter(txt=inp)
        all_response += resp+ " "
        #print("DEBUG send_msg")
        if '有' in yes_or_no:
            print()
            # TODO
            sro = RE(txt=inp)
            try:
                sro = eval(sro)
            except ValueError as e:
                # print("后台解析错误，请重试")
                continue

            try:
                print(f"但您似乎有修改意图{sro[0]}{sro[1]}==>{sro[2]}，确定修改吗？过程不可逆(y/n):")
                resp = f"但您似乎有修改意图：{sro[0]}{sro[1]}==>{sro[2]}，确定修改吗？过程不可逆(y/n):"
            except Exception as e:
                print("后台格式解析错误")
                
            all_response += resp+ " "
            
            await send_msg(all_response)
            modify_command = await receive_msg()
            
            if 'y' in modify_command.lower():
                if isinstance(sro,str):
                    try:
                        sro = eval(sro)
                    except ValueError as e:
                        await send_msg("后端解析错误，请重试")
                        continue
                print("请输入密码：")
                await send_msg("请输入密码：")
                
                passw = await receive_msg()
                
                if(VERIFY(passw)):
                    print(f"准备修改:{sro[0] + sro[1]} ===> {sro[2]}，输入任意字符继续")
                    await send_msg(f"准备修改:{sro[0] + sro[1]} ===> {sro[2]}，输入任意字符继续")
                    print("")
                    tmp = await receive_msg()
                    print("receive")
                    
                    editor = MyEditor(editing_method=method, model_name='Llama3-Chinese-8B-Instruct')
                    prompt = sro[0] + sro[1]
                    ground_truth = [None]
                    target_new = sro[2]
                    remember_loc = {
                        "loc": prompt,
                        "loc_ans": target_new
                    }
                    request = construct_input_data(sro, prompt, ground_truth, target_new)  # 构造输入数据
                    metrics, edited_model = editor.do_edit(request)
                    
                    if DEBUG:
                        print("="*40)
                        print(f"\033[34mid {id(edited_model)}\033[0m")
                        for name, param in edited_model.named_parameters():
                            # print(name)
                            if method == "FT":
                                if name == "model.layers.21.mlp.down_proj.weight":
                                    print(f"{name}@@@{param}")
                                    param_after1 = param.clone()
                            elif method == "ROME":
                                if name == "model.layers.5.mlp.down_proj.weight":
                                    print(f"{name}@@@{param}")
                                    param_after1 = param.clone()
                                    
                    
                        a = check_tensors_same(param_before1, param_after1)
                        print(a)
                        print("="*40)
                    
                    
                    
                    
                    
                    
                                    
                    print("修改成功！是否保存修改？这可能需要些时间。(y/n)：")
                    await send_msg("修改成功！是否保存修改？这可能需要些时间。(y/n)：")
                    save_cmd = await receive_msg()
                    
                    if 'y' in save_cmd.lower():
                        await send_msg("准备保存新模型...输入任意字符继续")
                        tmp = await receive_msg()
                        print("0/3 保存中...")
                        edited_model.save_pretrained(model_save_path)
                        print("1/3 保存中...")
                        TOKENIZER.save_pretrained(model_save_path)
                        print("2/3 保存中...")
                        add_new_prompt_and_targetnew_to_loc_file(new_data=remember_loc)  # 记录修改的内容
                        MODEL = edited_model
                        print("3/3 保存完成")
                        await send_msg("保存成功! 请继续向我提问吧")
                        # time.sleep(1)
                        # tmp = await receive_msg()
                        # # 重新加载模型
                        # print("\033[34m模型加载前\033[0m",id(MODEL))
                        # print(MODEL.config)
                        # if DEBUG:
                        #     print("="*40)
                        #     print(f"\033[34mid {id(edited_model)}\033[0m")
                        #     for name, param in MODEL.named_parameters():
                        #         # print(name)
                        #         if method == "FT":
                        #             if name == "model.layers.21.mlp.down_proj.weight":
                        #                 print(f"{name}@@@{param}")
                        #                 param_before = param.clone()
                        #         elif method == "ROME":
                        #             if name == "model.layers.5.mlp.down_proj.weight":
                        #                 print(f"{name}@@@{param}")
                        #                 param_before = param.clone()
                            
                        #     pb = param_before
                        #     print("="*40)
                        # model_manager.del_then_reload_model()
                        # print(f"\033[34m 原始模型已清空，并加载了新模型")
                        # MODEL = model_manager.get_model()
                        # TOKENIZER = model_manager.get_tokenizer()
                        # MODEL_NAME = MODEL.config._name_or_path if hasattr(MODEL.config, '_name_or_path') else None
                        # terminators = model_manager.get_terminators()
                        # pipeline = model_manager.get_pipeline()
                        
                        # print("\033[34m模型加载后\033[0m",id(MODEL))
                        # print(MODEL.config)
                        # if DEBUG:
                        #     for name, param in MODEL.named_parameters():
                        #         # print(name)
                        #         if method == "FT":
                        #             if name == "model.layers.21.mlp.down_proj.weight":
                        #                 print(f"{name}@@@{param}")
                        #                 param_after = param.clone()
                        #         elif method == "ROME":
                        #             if name == "model.layers.5.mlp.down_proj.weight":
                        #                 print(f"{name}@@@{param}")
                        #                 param_after = param.clone()
                                        
                        #     pa = param_after
                            
                        #     a = check_tensors_same(pb, pa)
                            
                        #     print(a)
                        # #os.execv(sys.executable, ['python'] + sys.argv) 
                        # print("重新加载成功，请继续向我提问吧")
                        # await send_msg("重新加载成功,请继续向我提问吧")
                    else:
                        
                        await send_msg("好的，已经取消保存。继续向我提问吧！")
                        continue
                else:
                    await send_msg("密码错误,请重新向我提问吧")
                    continue
            else:
                await send_msg("好的，已经取消修改。继续向我提问吧！")
                print("好的，已经取消修改。继续向我提问吧！")
                continue
        else:
            await send_msg(all_response)
            pass
        
    try:
        # ... 省略其他代码 ...
        writer.close()
        # 等待客户端连接关闭
        await writer.wait_closed()
    except Exception as e:
        print(f"An error occurred: {e}")
async def main():
    
    ip = os.popen('hostname -I | grep 192.168.1.1').read().strip()
    print(ip)
    server_ip = ip #'192.168.1.106'  # 服务器的IP地址
    server_port = 22223  # 服务器的端口号

    server = await asyncio.start_server(
        handle_client, server_ip, server_port)

    print(f'Server is listening on {server_ip}:{server_port}')

    async with server:
        await server.serve_forever()
        
        
def run_as_terminal():
    
    global messages_qa
    
    global MODEL
    global TOKENIZER
    global MODEL_NAME
    global terminators
    global pipeline
    global method
    global DEBUG
    
    if DEBUG:
        print("================================================================")
        print(f"\033[34mid {id(MODEL)}\033[0m")
        for name, param in MODEL.named_parameters():
            # print(name)
            if method == "FT":
                if name == "model.layers.21.mlp.down_proj.weight":
                    print(f"{name}@@@{param}")
                    param_before1 = param.clone()
            elif method == "ROME":
                if name == "model.layers.5.mlp.down_proj.weight":
                    print(f"{name}@@@{param}")
                    param_before1 = param.clone()
        print("===============================================================")
        
    while True:
        message = receive()
        all_response = ""
        
        inp = message
        if DEBUG:
            if "p" in inp.lower():
                print("="*40)
                print(f"\033[34mid {id(MODEL)}\033[0m")
                for name, param in MODEL.named_parameters():
                    # print(name)
                    if method == "FT":
                        if name == "model.layers.21.mlp.down_proj.weight":
                            print(f"{name}@@@{param}")
                            param_before = param.clone()
                    elif method == "ROME":
                        if name == "model.layers.5.mlp.down_proj.weight":
                            print(f"{name}@@@{param}")
                            param_before = param.clone()
                            
        if 'clean' in inp.lower():
            messages_qa = [{"role": "system", "content": "你是一个家庭智能代理，你名字叫小绿，经常和用户交流"}]
            send("后端已清空对话")
            continue
        yes_or_no = intend_detector(txt=inp)
        messages_qa.append(
                    {"role": "user", "content": inp}
                )
        messages_qa.append(
                    {"role": "system", "content": yes_or_no}
                )
        resp = chatter(txt=inp)
        all_response += resp+ " "
        #print("DEBUG send_msg")
        if '有' in yes_or_no:
            # TODO
            sro = RE(txt=inp)
            try:
                sro = eval(sro)
            except ValueError as e:
                # print("后台解析错误，请重试")
                continue
            print(sro)
            #print(f"但您似乎有修改意图{sro[0]}{sro[1]}==>{sro[2]}，确定修改吗？过程不可逆(y/n):")
            try:
                resp = f"但您似乎有修改意图：{sro[0]}{sro[1]}==>{sro[2]}，确定修改吗？过程不可逆(y/n):"
            except Exception as e:
                print("后台格式解析错误")
            all_response += resp+ " "
            
            send(all_response)
            modify_command = receive()
            
            if 'y' in modify_command.lower():
                if isinstance(sro,str):
                    try:
                        sro = eval(sro)
                    except ValueError as e:
                        send("后端解析错误，请重试")
                        continue
                #print("请输入密码：")
                send("请输入密码：")
                
                passw = receive()
                
                if(VERIFY(passw)):
                    #print(f"准备修改:{sro[0] + sro[1]} ===> {sro[2]}，输入任意字符继续")
                    send(f"准备修改:{sro[0] + sro[1]} ===> {sro[2]}，输入任意字符继续")
                    print("")
                    tmp = receive()
                    print("receive")
                    
                    editor = MyEditor(editing_method=method, model_name='Llama3-Chinese-8B-Instruct')
                    prompt = sro[0] + sro[1]
                    ground_truth = [None]
                    target_new = sro[2]
                    remember_loc = {
                        "loc": prompt,
                        "loc_ans": target_new
                    }
                    request = construct_input_data(sro, prompt, ground_truth, target_new)  # 构造输入数据
                    metrics, edited_model = editor.do_edit(request)
                    
                    if DEBUG:
                        print("="*40)
                        print(f"\033[34mid {id(edited_model)}\033[0m")
                        for name, param in edited_model.named_parameters():
                            # print(name)
                            if method == "FT":
                                if name == "model.layers.21.mlp.down_proj.weight":
                                    print(f"{name}@@@{param}")
                                    param_after1 = param.clone()
                            elif method == "ROME":
                                if name == "model.layers.5.mlp.down_proj.weight":
                                    print(f"{name}@@@{param}")
                                    param_after1 = param.clone()
                                    
                    
                        a = check_tensors_same(param_before1, param_after1)
                        print(a)
                        print("="*40)
                    
                    
                    
                    
                    
                    
                                    
                    # print("修改成功！是否保存修改？这可能需要些时间。(y/n)：")
                    send("修改成功！是否保存修改？这可能需要些时间。(y/n)：")
                    save_cmd = receive()
                    
                    if 'y' in save_cmd.lower():
                        send("准备保存新模型...输入任意字符继续")
                        tmp = receive()
                        print("0/3 保存中...")
                        edited_model.save_pretrained(model_save_path)
                        print("1/3 保存中...")
                        TOKENIZER.save_pretrained(model_save_path)
                        print("2/3 保存中...")
                        add_new_prompt_and_targetnew_to_loc_file(new_data=remember_loc)  # 记录修改的内容
                        print("3/3 保存完成")
                        MODEL = edited_model
                        send("保存成功! 请继续向我提问吧")
                        
                        
                        # 模型已经加载成功，无需再次加载
                        
                        # time.sleep(1)
                        # tmp = receive()
                        # # 重新加载模型
                        # print("\033[34m模型加载前\033[0m",id(MODEL))
                        # print(MODEL.config)
                        # if DEBUG:
                        #     print("="*40)
                        #     print(f"\033[34mid {id(edited_model)}\033[0m")
                        #     for name, param in MODEL.named_parameters():
                        #         # print(name)
                        #         if method == "FT":
                        #             if name == "model.layers.21.mlp.down_proj.weight":
                        #                 print(f"{name}@@@{param}")
                        #                 param_before = param.clone()
                        #         elif method == "ROME":
                        #             if name == "model.layers.5.mlp.down_proj.weight":
                        #                 print(f"{name}@@@{param}")
                        #                 param_before = param.clone()
                            
                        #     pb = param_before
                        #     print("="*40)
                        # model_manager.del_then_reload_model()
                        # print(f"\033[34m 原始模型已清空，并加载了新模型")
                        # MODEL = model_manager.get_model()
                        # TOKENIZER = model_manager.get_tokenizer()
                        # MODEL_NAME = MODEL.config._name_or_path if hasattr(MODEL.config, '_name_or_path') else None
                        # terminators = model_manager.get_terminators()
                        # pipeline = model_manager.get_pipeline()
                        
                        # print("\033[34m模型加载后\033[0m",id(MODEL))
                        # print(MODEL.config)
                        # if DEBUG:
                        #     for name, param in MODEL.named_parameters():
                        #         # print(name)
                        #         if method == "FT":
                        #             if name == "model.layers.21.mlp.down_proj.weight":
                        #                 print(f"{name}@@@{param}")
                        #                 param_after = param.clone()
                        #         elif method == "ROME":
                        #             if name == "model.layers.5.mlp.down_proj.weight":
                        #                 print(f"{name}@@@{param}")
                        #                 param_after = param.clone()
                                        
                        #     pa = param_after
                            
                        #     a = check_tensors_same(pb, pa)
                            
                        #     print(a)
                        # #os.execv(sys.executable, ['python'] + sys.argv) 
                        # #print("重新加载成功，请继续向我提问吧")
                        # send("重新加载成功,请继续向我提问吧")
                    else:
                        
                        send("好的，已经取消保存。继续向我提问吧！")
                        continue
                else:
                    send("密码错误,请重新向我提问吧")
                    continue
            else:
                send("好的，已经取消修改。继续向我提问吧！")
                #print("好的，已经取消修改。继续向我提问吧！")
                continue
        else:
            send(all_response)
            pass
    
if __name__ == '__main__':
    
    # global MODEL
    # global TOKENIZER
    # global MODEL_NAME
    # global terminators
    # global pipeline
    
    # MODEL = model_manager.get_model()
    # TOKENIZER = model_manager.get_tokenizer()
    # MODEL_NAME = MODEL.config._name_or_path if hasattr(MODEL.config, '_name_or_path') else None
    # terminators = model_manager.get_terminators()
    # pipeline = model_manager.get_pipeline()
    mode_demo = 'terminal'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true', help='debug mode')
    args, _ = parser.parse_known_args()
    
    if args.debug:
        # if you use vscode on hpc-login-01
        import debugpy
        
        debugpy.connect(('192.168.1.50', 6789))
        debugpy.wait_for_client()
        debugpy.breakpoint()
    
        mode_demo == 'terminal'
    
    # 'terminal'  # 'server'
    
    if mode_demo == 'terminal':
        run_as_terminal()
    elif mode_demo == 'server':
        asyncio.run(main())
