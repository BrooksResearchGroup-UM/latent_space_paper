#!/bin/bash 
#SBATCH --job-name=prott5_embedding
#SBATCH --time=01-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -p gpuA5500
#SBATCH --mem=60G
#SBATCH --mail-user=cdchiang@umich.edu

# To load conda properly
source /home/cdchiang/miniconda3/etc/profile.d/conda.sh

conda activate ProtTrans
echo "Conda environment: $CONDA_DEFAULT_ENV"

python3 ./prott5_embedder.py --input ../data/processed/fasta/cyclase/PF05147_hits_200_500aa_test.fasta --output ../models/protein_embeddings.h5 --per_protein 1 --model ../models/prot_t5_xl_uniref50

exit
















