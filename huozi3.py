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

    model_id = "/home/share/models/Chinese-Mixtral-8x7B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, device_map="auto")

    text = "美国总统是"
    inputs = tokenizer(text, return_tensors="pt",padding=True).to(0)

    outputs = model.generate(**inputs, max_new_tokens=20)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    
    '''
    text = "美国总统是"
    inputs = tok(text, return_tensors="pt",padding=True).to(0)
    outputs = model(**inputs)
    logits = output.logits
    lom = torch.argmax(logits,dim=-1)
    llom = lom.detach().clone().tolist()
    tok.decode(llom[0])
    '''
