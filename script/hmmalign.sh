#!/bin/bash 
#SBATCH --job-name=hmmalign
#SBATCH --time=02:00:00
#SBATCH -p gpu
#SBATCH --chdir=/home/cdchiang/vae/PF01494/latent_space_paper

# Load modules

module load hmmer/3.3.2

# Run hmmalign

hmmalign --outformat afa ../data/PF01494_seed.hmm ../data/PF01494_20201216_300_600aa_train_test.fasta > ../data/PF01494_20201216_300_600aa_plus_train_test_MSA2.fasta

exit
















