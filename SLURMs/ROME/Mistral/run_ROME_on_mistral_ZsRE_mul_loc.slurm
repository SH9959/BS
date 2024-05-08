#!/bin/bash
#SBATCH -J mt-ori-mistral_ROME                                # 作业名为 test
#SBATCH -o ./out_log/origin/multi_random_loc/ROME+mistral-%j.out                        # 屏幕上的输出文件重定向到 test.out
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH -t 1-23:59:59                          # 任务运行的最长时间为 1 小时
##SBATCH -w gpu09                             # 指定运行作业的节点是 gpu06，若不填写系统自动分配节点
#SBATCH --gres=gpu:a100-sxm-64gb:1
#SBATCH --mem=52G
##SBATCH --cpus-per-task=4
# source ~/.bashrc


source ~/miniconda3/etc/profile.d/conda.sh
conda activate BS


python main.py \
    --editing_method=ROME \
    --model=mistral-7b \
    --dataset_dir=/home/hsong/BS/DATA/KnowEdit-huggingface/benchmark/ZsRE \
    --dataset_name=ZsRE-test-all.json \
    --SAVE_MODE=True \
    --SAVE_PATH=/home/hsong/BS/output/added_incident_constrain/multi_random_loc/ROME/Mistral/origin
    # --dataset_size=1
    # -d

    
# model ['gpt-j-6B','mistral-7b']
# ZsRE_incident数据集  增加了kl约束  还未采用随机loc测试

# 超参数还需要修改 compute_v.py中的mode  和   main.py中的MULTI_LOC

#