#!/bin/bash 
#SBATCH --job-name=Embedding
#SBATCH --time=00-00:60:00
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -p gpu2080
#SBATCH --mem=60G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cdchiang@umich.edu

# Activate conda env
conda activate vae
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Embedding and plot latent space
python ./analysis_simulated.py

exit

















