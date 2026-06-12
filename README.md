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

The hmmpress step is required for hmmscan to work.
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

```bash
hmmpress class3_LanKC.hmm
```
```bash
hmmpress class4_LanL.hmm
```

```bash
hmmsearch class3_LanKC.hmm PF05147_hits_upper.fasta > class3_LanKC_hmmsearch.txt
```
```bash
hmmsearch class4_LanL.hmm PF05147_hits_upper.fasta > class4_LanL_hmmsearch.txt
```

```bash
awk '/^ *[^- ]/ {print $9, $1}' class3_LanKC_hmmsearch.txt > class3_LanKC_hmmsearch_evalues.txt
```
```bash
awk '/^ *[^- ]/ {print $9, $1}' class4_LanL_hmmsearch.txt > class4_LanL_hmmsearch_evalues.txt
```