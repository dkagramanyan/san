#!/bin/bash
#SBATCH --job-name=SAN-GAN-16
#SBATCH --gpus=4
#SBATCH --cpus-per-task=4
#SBATCH --time=3-0:0
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --account=proj_1631
#SBATCH --constraint="type_e"

module purge
module load Python/Anaconda
module load CUDA/12.4
module load gnu13/13.3

source activate san

cd ..

python train.py --outdir=./training-runs/wc-cv \
        --cfg=stylegan3-r \
        --data=./datasets/o_bc_left_4x_1536_1024x1024_rgb_16x16.zip \
        --gpus=4 \
        --batch=384 \
        --mirror=1 \
        --snap 100 \
        --batch-gpu 96 \
        --kimg 10000 \
        --cond True \
        --syn_layers 6

