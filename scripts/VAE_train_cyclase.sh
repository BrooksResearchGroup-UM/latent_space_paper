#!/bin/bash 
#SBATCH --job-name=VAE_train
#SBATCH --time=01-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -p gpu2080
#SBATCH --mem=60G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cdchiang@umich.edu

# Load modules
module load anaconda/3.5.3.0

# Train VAE model
python ./train_cyclase.py \
    --num_epoch 1000 \
    --batch_size 128 \
    --weight_decay 0.00190 \
    --lr 0.000779 \
    --dim 10 \
    --encode_layer 500 100 \
    --decode_layer 100 500 \
    --seed 2 \
    --input_file ../data/processed/training/cyclase

exit

















