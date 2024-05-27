# quickstart.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

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

    
    model_id = "HIT-SCIR/huozi-7b-rlhf"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        #attn_implementation="flash_attention_2",
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    text = """<|beginofutterance|>系统
    你是一个智能助手<|endofutterance|>
    <|beginofutterance|>用户
    你的名字是什么<|endofutterance|>
    <|beginofutterance|>助手
    """

    inputs = tokenizer(text, return_tensors="pt").to(0)

    outputs = model.generate(
        **inputs,
        eos_token_id=57001,
        temperature=0.1,
        top_p=0.9,
        max_new_tokens=2048,
    )
    print(tokenizer.decode(outputs[0], skip_special_tokens=False))
