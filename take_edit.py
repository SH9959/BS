

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

default_model_id = "./MODELS/models--FlagAlpha--Llama3-Chinese-8B-Instruct/snapshots/d76c4a5d365b041d1b440337dbf7da9664a464fc"
model_save_path = "./MODELS/models--FlagAlpha--Llama3-Chinese-8B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=default_model_id,
    model_kwargs={"torch_dtype": torch.float16,"do_sample": False},
    device="cuda",
)

MODEL = pipeline.model
TOKENIZER = pipeline.tokenizer

MODEL_NAME = MODEL.config._name_or_path

if MODEL_NAME is not None:
    print(MODEL_NAME)
else:
    MODEL_NAME = default_model_id

terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

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
        editor = BaseEditor(self.hparams, MODEL, TOKENIZER, MODEL_NAME)  #take long time 1-2min
        
        # if self.args_dict["Editing Method"] in ['MEMIT', 'PMET']:
        #     metrics, edited_model, _ = editor.batch_edit(**params)
        if self.editing_method in ['KN', 'ROME','FT','LoRA','MEMIT', 'PMET']:
            metrics, edited_model, _ = editor.edit(**params)
        else:
            raise ValueError(f"Unsupported Editing Method: {self.args_dict['Editing Method']}")
        
        
        return metrics, edited_model
    

def intend_detector(txt:str,pipiline:object=pipeline)->str:
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
    global message_qa
    messages_qa.append(
                    {"role": "user", "content": txt}
                )
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
        temperature=0.6,
        top_p=0.9
    )
    content = outputs[0]["generated_text"][len(prompt):]
    messages_qa.append(
                    {"role": "system", "content": content}
                )
    print("\033[33mAgent Chat<<\033[0m",content)  # Llama3-Chinese-8B-Instruct
    return content
def RE(txt:str,pipiline:object=pipeline)->None:
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
        temperature=0.6,
        top_p=0.9
    )
    content = outputs[0]["generated_text"][len(prompt):]
    #TODO content一定要是主语关系宾语时才向用户展现
    sro = match_list_pattern(content)
    print("\033[33mAgent RE<<\033[0m",sro)  # Llama3-Chinese-8B-Instruct
    return content
def VERIFY(password):
    
    return True
def check_legality(txt:str)->bool:
    # TODO
    return True


async def handle_client(reader, writer):
    def send_msg(message):
        writer.write(message.encode())
        await writer.drain()
        
    def reveive_msg():
        message = []
        while True:
            data = await reader.read(1024)
            message += data.decode().strip()
            if not message:
                print("No message received from client.")
                break
            print(f"Received message from client: {message}")
            # 在这里添加逻辑来处理客户端发送的消息，然后构造响应
            # 这里只是一个示例，可以根据需要进行修改
        
    client_address = writer.get_extra_info('peername')
    print(f'Connected to client at {client_address[0]}:{client_address[1]}')

    while True:
        try:
            data = await reader.read(1024)
            message = data.decode().strip()
            if not message:
                print("No message received from client.")
                break
            print(f"Received message from client: {message}")
            # 在这里添加逻辑来处理客户端发送的消息，然后构造响应
            # 这里只是一个示例，可以根据需要进行修改

            inp = message
            if 'clean' in inp.lower():
                messages_qa = [{"role": "system", "content": "你是一个家庭智能代理，你名字叫小绿，经常和用户交流"}]
                response="已清空对话"
                continue
            yes_or_no = intend_detector(txt=inp)
            messages_qa.append(
                        {"role": "user", "content": inp}
                    )
            messages_qa.append(
                        {"role": "system", "content": yes_or_no}
                    )
            if '有' in yes_or_no:
                response = chatter(txt=inp)
                sendsend_msg(response)
                print()
                # TODO
                sro = RE(txt=inp)
                try:
                    sro = eval(sro)
                except ValueError as e:
                    # print("后台解析错误，请重试")
                    continue
                send_msg(f"您似乎有修改意图{sro[0]}{sro[1]}==>{sro2}，确定修改吗？过程不可逆(y/n):")
                
                
                modify_command = receive()
                
                
                
                if 'y' in modify_command.lower():
                    if isinstance(sro,str):
                        try:
                            sro = eval(sro)
                        except ValueError as e:
                            print("后台解析错误，请重试")
                            continue
                    send("请输入密码：")
                    passw = receive()
                    if(VERIFY(passw)):
                        print(f"正在修改:{sro[0] + sro[1]} ===> {sro[2]}")
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
                        send("修改成功！是否保存修改？这可能需要些时间。(y/n)：")
                        save_cmd = receive()
                        if 'y' in save_cmd.lower():
                            send("正在保存新模型...")
                            edited_model.save_pretrained(model_save_path)
                            TOKENIZER.save_pretrained(model_save_path)
                            add_new_prompt_and_targetnew_to_loc_file(new_data=remember_loc)  # 记录修改的内容
                            send("保存成功! 请重新导入模型后再尝试向我提问吧")
                            time.sleep(1)
                        else:
                            send("取消修改")
                    else:
                        send("密码错误")
                        continue
            else:
                chatter(txt=inp)
            
            
            
            
            response = "Hello from server!"
            writer.write(response.encode())
            
            await writer.drain()
        except Exception as e:
            print(f"An error occurred: {e}")
            break

    writer.close()
    await writer.wait_closed()

async def main():
    server_ip = '192.168.1.108'  # 服务器的IP地址
    server_port = 22223  # 服务器的端口号

    server = await asyncio.start_server(
        handle_client, server_ip, server_port)

    print(f'Server is listening on {server_ip}:{server_port}')

    async with server:
        await server.serve_forever()
        
    
if __name__ == '__main__':
    method = 'FT'
    
    mode_demo = 'terminal'  # 'server'
    
    if mode_demo == 'terminal':
        while True:
            inp = receive()
            if 'clean' in inp.lower():
                messages_qa = [{"role": "system", "content": "你是一个家庭智能代理，你名字叫小绿，经常和用户交流"}]
                print("已清空对话")
                continue
            yes_or_no = intend_detector(txt=inp)
            messages_qa.append(
                        {"role": "user", "content": inp}
                    )
            messages_qa.append(
                        {"role": "system", "content": yes_or_no}
                    )
            if '有' in yes_or_no:
                chatter(txt=inp)
                print()
                # TODO
                sro = RE(txt=inp)
                try:
                    sro = eval(sro)
                except ValueError as e:
                    # print("后台解析错误，请重试")
                    continue
                send(f"您似乎有修改意图{sro}，确定修改吗？过程不可逆(y/n):")
                modify_command = receive()
                if 'y' in modify_command.lower():
                    if isinstance(sro,str):
                        try:
                            sro = eval(sro)
                        except ValueError as e:
                            print("后台解析错误，请重试")
                            continue
                    send("请输入密码：")
                    passw = receive()
                    if(VERIFY(passw)):
                        print(f"正在修改:{sro[0] + sro[1]} ===> {sro[2]}")
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
                        send("修改成功！是否保存修改？这可能需要些时间。(y/n)：")
                        save_cmd = receive()
                        if 'y' in save_cmd.lower():
                            send("正在保存新模型...")
                            edited_model.save_pretrained(model_save_path)
                            TOKENIZER.save_pretrained(model_save_path)
                            add_new_prompt_and_targetnew_to_loc_file(new_data=remember_loc)  # 记录修改的内容
                            send("保存成功! 请重新导入模型后再尝试向我提问吧")
                            time.sleep(1)
                        else:
                            send("取消修改")
                    else:
                        send("密码错误")
                        continue
            else:
                chatter(txt=inp)
    elif mode_demo == 'server':
        asyncio.run(main())
