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

# This is a neural network test file.
python ./train_FDMO.py

exit
















