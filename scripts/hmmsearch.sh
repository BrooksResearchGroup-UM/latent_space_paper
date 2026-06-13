#!/bin/bash 
#SBATCH --job-name=hmmsearch
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH -p cpu

# Load modules
module load hmmer/3.3.2

hmmsearch -A ./data/processed/hmmer/cyclase/PF05147_hits.sto ./data/processed/hmmer/cyclase/PF05147.hmm \
    ./data/processed/fasta/refseq_protein_bacteria_archaea.faa > ./data/processed/hmmer/cyclase/PF05147.out
esl-reformat fasta ./data/processed/hmmer/cyclase/PF05147_hits.sto > ./data/raw/cyclase/PF05147_hits_test.fasta

exit
















