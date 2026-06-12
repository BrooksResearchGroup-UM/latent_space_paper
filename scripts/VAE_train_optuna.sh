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

# Activate conda env
echo "Conda environment: $CONDA_DEFAULT_ENV"

# This is a neural network test file.
python ./train_optuna.py

exit

















