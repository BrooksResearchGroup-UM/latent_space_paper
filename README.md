# latent_space_paper

This repository contains code for training variational autoencoder models and performing downstream analysis.

## Installation

Requires Python >= 3.9.

```bash
mamba env create -f env.yml
```

## HMMER installation and run

The training of VAE models requires aligned sequences, which can be done using HMMER.

To download HMMER, please follow the instructions on its official [HMMER website](http://hmmer.org/download.html).

To verify that HMMER has been installed successfully, run the following command in your terminal or command line:

```bash
hmmalign --version
```

Create an HMM profile from your seed alignment. Here we use FDMO (Pfam PF01494) as our example.

```bash 
hmmbuild ./data/processed/hmmer/FDMO/PF01494_seed.hmm ./data/processed/hmmer/FDMO/PF01494_seed.sto
```

Run alignment.

```bash
hmmalign ./data/processed/hmmer/FDMO/PF01494_seed.hmm ./data/raw/FDMO/PF01494_input.fasta > ./data/processed/fasta/FDMO/PF01494_MSA.fasta
```

### Multiple sequence alignment using HMMER
#### HMMER installation:
1. Download HMMER from the official [HMMER website](http://hmmer.org/download.html).
2. Follow the installation instructions provided on the website or in the downloaded package.
3. To verify that HMMER has been installed successfully, run the following command in your terminal or command line:
   ```bash
   hmmalign --version

#### Run hmmalign:
1. Create an HMM profile from your seed alignment.
   ```bash 
   hmmbuild ./data/PF01494_seed.hmm ./data/PF01494_seed.sto
2. Run alignment and output the result to a fasta file
   ```bash
   hmmalign ./data/PF01494_seed.hmm ./data/PF01494_input.fasta > ./data/PF01494_MSA.fasta
