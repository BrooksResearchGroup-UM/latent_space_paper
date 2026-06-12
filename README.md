# latent_space_paper

This repository contains code for training variational autoencoder models and performing downstream analysis.

## Installation

Requires Python >= 3.9.

```bash
mamba env create -f env.yml
```

## HMMER installation and run

The training of VAE models requires aligned sequences of a protein family of interest, which can be acquired by multiple sequence alignment (MSA) using HMMER.

To download HMMER, please follow the instructions on its official [HMMER website](http://hmmer.org/download.html).

To verify that HMMER has been installed successfully, run the following command in your terminal or command line:

```bash
hmmalign --version
```

Create an HMM profile from your seed alignment. Here we use FDMO (Pfam PF01494) as our example:

```bash 
hmmbuild ./data/processed/hmmer/FDMO/PF01494_seed.hmm ./data/processed/hmmer/FDMO/PF01494_seed.sto
```

Run alignment:

```bash
hmmalign ./data/processed/hmmer/FDMO/PF01494_seed.hmm ./data/raw/FDMO/PF01494_input.fasta > ./data/processed/fasta/FDMO/PF01494_MSA.fasta
```

## Curation of the cyclase dataset

Download bacterial database and combine all bacterial fasta files:
```bash 
wget -r -nH --cut-dirs=6 -A '*.protein.faa.gz' ftp://ftp.ncbi.nlm.nih.gov/refseq/release/bacteria/
```
```bash
cat *.protein.faa.gz > bacteria_nonredundant_protein.faa.gz
```

Download archaeal database and combine all archael fasta files:
```bash
wget -r -nH --cut-dirs=6 -A '*.protein.faa.gz' ftp://ftp.ncbi.nlm.nih.gov/refseq/release/archaea/
```
```bash
cat *.protein.faa.gz > archaea_nonredundant_protein.faa.gz
```

Combine bacterial and archaeal fasta files and unzip the gz file:
```bash
cat *_protein.faa.gz > refseq_protein_bacteria_archaea.faa.gz
```
```bash
gunzip refseq_protein_bacteria_archaea.faa.gz
```

The hmmpress step is required for hmmsearch to work.
```bash
hmmpress PF05147.hmm
```

Run hmmsearch:
```bash
hmmsearch -A PF05147_hits.sto PF05147.hmm refseq_protein_bacteria_archaea.fasta > PF05147.out
```

Output the hits from hmmsearch to a fasta file:
```bash
esl-reformat fasta PF05147.out > PF05147_hits.fasta
```

Filter sequences within the range of 200 to 500 amino acids and run MSA:
```bash
hmmalign --outformat afa PF05147.hmm PF05147_hits_200_500aa.fasta > PF05147_hits_200_500aa_MSA.fasta
```

## Run hmmsearch to identify class 3 and class 4 cyclases

The hmmpress step is required for hmmsearch to work.
```bash
hmmpress class3_LanKC.hmm
```
```bash
hmmpress class4_LanL.hmm
```

Run hmmsearch:
```bash
hmmsearch class3_LanKC.hmm PF05147_hits_upper.fasta > class3_LanKC_hmmsearch.txt
```
```bash
hmmsearch class4_LanL.hmm PF05147_hits_upper.fasta > class4_LanL_hmmsearch.txt
```

> **Note**: Change all lowercase amino acids to uppercase.

Extract hmmsearch outputs:
```bash
awk '/^ *[^- ]/ {print $9, $1}' class3_LanKC_hmmsearch.txt > class3_LanKC_hmmsearch_evalues.txt
```
```bash
awk '/^ *[^- ]/ {print $9, $1}' class4_LanL_hmmsearch.txt > class4_LanL_hmmsearch_evalues.txt
```

## Generate the simulated dataset

Output an LG amino-acid replacement matrix:
```bash
python ./scripts/read_LG_matrix.py
```

Generate sequences based on a random tree and the replacement matrix:
```bash
python ./scripts/simulate_msa.py
```

## Prepare input datasets for VAE training

To pre-process MSA files and perform one-hot encoding for each dataset, use this [jupyter notebook](https://github.com/BrooksResearchGroup-UM/latent_space_paper/blob/main/notebooks/MSA.ipynb).

## VAE model training

To train VAE models, first make sure the training dataset is ready.

With conda environment with torch and cuda activated, run the training on clusters:

For the simulated dataset:
```bash
sbatch ./scripts/VAE_train_simulated.sh
```
For the cyclase dataset:
```bash
sbatch ./scripts/VAE_train_cyclase.sh
```
Use [Optuna](https://github.com/optuna/optuna) for hyperparameter optimization:
```bash
sbatch ./scripts/VAE_train_optuna.sh
```
> **Note**: The example here is for the training on the cyclase dataset.

For the FDMO dataset:
```bash
sbatch ./scripts/VAE_train_FDMO.sh
```

## Obtain the embeddings using protein language models

For ESM-1b and ESM-2 (650M/3B):

Clone the repository of [esm](https://github.com/facebookresearch/esm/tree/main) and install esm by following the intructions. I created a separate environment for esm.

Manually download the models using `wget` ([ESM-1b](https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt), [ESM-2-650M](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt), [ESM-2-3B](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt)) to `~/.cache/torch/hub/checkpoints` if needed.

```bash
sbatch ./scripts/esm_embedding.sh
```

For ProtT5-XL:

Install [ProtTrans](https://github.com/agemagician/ProtTrans) by following the intructions. I created a separate environment for ProtTrans.

Since HPC compute nodes lack internet access, download the ProtT5-XL model first by running Steps 1 and 2 in this [jupyter notebook](https://github.com/BrooksResearchGroup-UM/latent_space_paper/blob/main/notebooks/ProtT5-XL-UniRef50.ipynb).

```bash
sbatch ./scripts/prott5_embedding.sh
```