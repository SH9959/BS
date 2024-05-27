#!/bin/bash
#SBATCH -J l-c_ROME_test                              # 作业名为 test
#SBATCH -o ./out_log/523/llama3/counterfact/ROME-%j.out                        # 屏幕上的输出文件重定向到 test.out
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH -t 13:59:59                          # 任务运行的最长时间为 1 小时
##SBATCH -w gpu06                             # 指定运行作业的节点是 gpu06，若不填写系统自动分配节点
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --mem=64G
##SBATCH --cpus-per-task=4
# source ~/.bashrc


source ~/miniconda3/etc/profile.d/conda.sh
conda activate BS

# llama3 counterfact
python statistic.py \
    --editing_method=ROME \
    --model=Llama3-Chinese-8B-Instruct \
    --dataset_dir=/home/hsong/BS/DATA/editing-data/data/counterfact \
    --dataset_name=counterfact-val.json \
    --SAVE_MODE=False \
    --SAVE_PATH=/home/hsong/BS/output/518/laywise/multi_random_loc/ROME/Llama3-Chinese-8B-Instruct/