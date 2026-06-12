#!/bin/bash 
#SBATCH --job-name=esm_embedding
#SBATCH --time=01-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -p gpuA5500
#SBATCH --mem=60G
#SBATCH --mail-user=cdchiang@umich.edu

# To load conda properly
source /home/cdchiang/miniconda3/etc/profile.d/conda.sh

conda activate esmfold
echo "Conda environment: $CONDA_DEFAULT_ENV"

# ESM-1b
python ./extract.py esm1b_t33_650M_UR50S ../data/processed/fasta/cyclase/PF05147_hits_200_500aa_U_id_test.fasta \
  ../models/PF05147_esm1b --repr_layers 33 --include mean

# ESM-2-650M
python ./extract.py esm2_t33_650M_UR50D ../data/processed/fasta/cyclase/PF05147_hits_200_500aa_U_id_test.fasta \
  ../models/PF05147_esm2_650M --repr_layers 33 --include mean

# ESM-2-3B
python ./extract.py esm2_t36_3B_UR50D ../data/processed/fasta/cyclase/PF05147_hits_200_500aa_U_id_test.fasta \
  ../models/PF05147_esm2_3B --repr_layers 36 --include mean

exit
















