#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --mem=120G
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=fastclipv3
#SBATCH --partition=gpu
#SBATCH --output=%x_%j.log

source ~/.bashrc
conda activate fast_clip

export PYTHONPATH="$PYTHONPATH:$PWD/src"
export HUGGINGFACE_HUB_CACHE='./checkpoints/huggingface'

torchrun \
    --nproc_per_node=2 --nnodes=1 --node_rank=0 \
    --rdzv-id=4204 --rdzv-backend=c10d --rdzv-endpoint='127.0.0.1' \
    src/training/main.py \
    --save-frequency 1 \
    --train-data './datasets/dfn_data/00000{000..139}.tar' \
    --train-num-samples 1000000 --data_size 1400000 \
    --warmup 500 \
    --batch-size 320 \
    --epochs 30 \
    --workers 6 \
    --model ViT-B-16 \
    --name fastclipv3 \
    --seed 2025 \
    --wd 0.2 \
    --local-loss \
    --multiply_tau --temperature_scheme global_learnable \
    --lr 3.125e-4 --lr_tau 7.8125e-5 --lr_tau_scheduler step_thresh --rho 11.0 \
    --gamma 0.9 --gamma_schedule cosine --gamma_decay_epochs 10 \
    --distill-model ViT-B-32 \
    --distill-pretrained '/scratch/group/optmai/anant/Ref_modelling/fast_clip/src/clip_vit_b32_openai.pth'
