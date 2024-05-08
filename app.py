from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import transformers
import torch
import time
import logging

# 禁用 WARN 级别的日志
logging.disable(logging.WARNING)

model_id = "FlagAlpha/Llama3-Chinese-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.float16},
    device="cuda",
)
terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

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
    print("\033[33mAgent意图识别为:\033[0m",content)
    
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
    print("\033[33mAgent<<\033[0m",content)  # Llama3-Chinese-8B-Instruct
    return content




def re(txt:str,pipiline:object=pipeline)->None:
    
    # global message_re
    messages_re = [{"role": "system", "content": "你是一个关系抽取器,在提取用户话中的信息后，会输出(主语,关系,宾语)信息。等待用户确认"}]
    
    task = "用户输入：‘不对，红楼梦的作者应该是曹雪芹’。你需要输出‘(红楼梦，作者，曹雪芹)’."
    "用户输入：‘不对，乔丹的父亲是普尔’。你需要输出‘(乔丹，父亲，普尔)’."
    "用户输入：‘小红喜欢的运动是篮球’。你需要输出‘(小红，喜欢的运动，篮球)’."
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
    print("\033[33mAgent<<\033[0m",content)  # Llama3-Chinese-8B-Instruct
    return content
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true', help='debug mode')
    args, _ = parser.parse_known_args()
    if args.debug:
        # if you use vscode on hpc-login-01
        import debugpy
        debugpy.connect(('192.168.1.50', 6789))
        debugpy.wait_for_client()
        debugpy.breakpoint()

    while True:
        try:
            # 尝试以 utf-8 编码读取输入
            inp = input(f"\033[34mUSER>>\033[0m")
        except UnicodeDecodeError:
            # 如果失败，尝试使用系统默认编码读取输入
            import sys
            inp = input(f"\033[34mUSER>>\033[0m").encode(sys.stdin.encoding, errors='ignore').decode('utf-8', errors='ignore')

        yes_or_no = intend_detector(txt=inp)
        
        messages_qa.append(
                    {"role": "user", "content": inp}
                )
        messages_qa.append(
                    {"role": "system", "content": yes_or_no}
                )
        
        if '有' in yes_or_no:
            print()
            # TODO
            sro = re(txt=inp)
            
            modify_command = input(f"您似乎有修改意图{sro}，确定修改吗？过程不可逆(y/n):")
            if 'y' in modify_command:
                passw = int(input("请输入密码："))
                print("修改中...")
                time.sleep(1)
                print("修改完成，请再次提问我试试吧。")
            else:
                print("取消修改")
                
            pass
        elif '无' in yes_or_no:
            chatter(txt=inp)
        else:
            print('error')
    
    
    
    # for name, module in model.named_modules():
    #     print(f"{name}@@@{module}")
    #     print()