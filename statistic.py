import torch
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from main import MyPipeline
from take_edit import ModelManager
from layer_wise import myNetHook, myRankLen, get_info, calculate_layer_to_modify

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
    
    test_data = Task.read_dataset()
    
    ret = Task.extract_dataset(test_data=test_data)
    
    
    
    
    print(torch.cuda.memory_allocated())    
    model_id="./MODELS/models--FlagAlpha--Llama3-Chinese-8B-Instruct/snapshots/d76c4a5d365b041d1b440337dbf7da9664a464fc"
    model_save_path="./MODELS/models--FlagAlpha--Llama3-Chinese-8B-Instruct/snapshots/d76c4a5d365b041d1b440337dbf7da9664a464fc"
    # model_id="mistralai/Mistral-7B-v0.1"
    # model_save_path="./MODELS/Mistral-7B-v0.1"
    method = 'ROME'
    model_name = 'llama3'
    DEBUG = True
    model_manager = ModelManager(model_id=model_id,model_save_path=model_save_path)
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
    
    layers = []
    for i in range(len(ret['prompts'])):
        txt = ret['prompts'][i]
        target = [ret["target_new"][i]]

        rank_list = get_info(MODEL, TOKENIZER, txt, target, layer_hook=None)
        layer_ind = calculate_layer_to_modify(rank_list)
        layers.append(layer_ind)
        
    
    print("layers:",layers)
    mean_layer = np.mean(layers)
    std_layer = np.std(layers)
    print(mean_layer,std_layer)
    # 创建棒状图
    plt.figure(figsize=(8, 6))

    # 绘制均值棒状图
    plt.bar(1, mean_layer, yerr=std_layer, capsize=10, label='Mean Layer')

    # 添加标题和标签
    plt.title('Layers Analysis')
    plt.xlabel('Metrics')
    plt.ylabel('Layer Index')
    plt.xticks([1], ['Mean Layer'])  # 只有一个指标，所以 x 轴只有一个刻度
    plt.legend()
    plt.savefig(f'{model_name}.png')
    print("over")

