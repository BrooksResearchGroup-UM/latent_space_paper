#!/bin/bash 
#SBATCH --job-name=hmmsearch
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH -p cpu

# Load modules
module load hmmer/3.3.2

# Class 3
hmmsearch ./data/processed/hmmer/cyclase/class3_LanKC.hmm \
    ./data/processed/fasta/cyclase/PF05147_hits_upper.fasta > ./data/processed/hmmer/cyclase/class3_LanKC_hmmsearch.txt

# Class 4
hmmsearch ./data/processed/hmmer/cyclase/class4_LanL.hmm \
    ./data/processed/fasta/cyclase/PF05147_hits_upper.fasta > ./data/processed/hmmer/cyclase/class4_LanL_hmmsearch.txt

exit
















