# Leveraging latent space models for enzyme discovery and sampling

This repository contains the official implementation, data curation pipelines, and downstream data analysis for the manuscript **"Leveraging latent space models for enzyme discovery and sampling"** submitted to *PNAS*.

---

## 1. Installation

This project requires Python >= 3.9. We recommend using `mamba` (or `conda`) to deploy the isolated environment efficiently:

```bash
mamba env create -f env.yml
conda activate vae
```

## 2. HMMER installation and alignment

The training of our Variational Autoencoder (VAE) models requires aligned sequences of a target protein family, generated via Multiple Sequence Alignment (MSA) using HMMER.

2.1. Download HMMER by following the instructions on its official [HMMER website](http://hmmer.org/download.html).

2.2. Verify the installation:

```bash
hmmalign --version
```

Example: FDMO / Pfam PF01494

2.3. Create an HMM profile from your starting seed alignment:

```bash 
hmmbuild ./data/processed/hmmer/FDMO/PF01494_seed.hmm ./data/processed/hmmer/FDMO/PF01494_seed.sto
```

2.4. Run alignment:

```bash
hmmalign ./data/processed/hmmer/FDMO/PF01494_seed.hmm ./data/raw/FDMO/PF01494_input.fasta > ./data/processed/fasta/FDMO/PF01494_MSA.fasta
```

## 3. Curation of the cyclase dataset

3.1. Download and aggregate the complete RefSeq bacterial and archaeal protein sequence databases:

```bash
# Fetch and merge Bacterial database 
wget -r -nH --cut-dirs=6 -A '*.protein.faa.gz' ftp://ftp.ncbi.nlm.nih.gov/refseq/release/bacteria/
cat *.protein.faa.gz > bacteria_nonredundant_protein.faa.gz
rm *.protein.faa.gz

# Fetch and merge Archaeal database
wget -r -nH --cut-dirs=6 -A '*.protein.faa.gz' ftp://ftp.ncbi.nlm.nih.gov/refseq/release/archaea/
cat *.protein.faa.gz > archaea_nonredundant_protein.faa.gz
rm *.protein.faa.gz

# Consolidate into a unified database matrix
cat bacteria_nonredundant_protein.faa.gz archaea_nonredundant_protein.faa.gz > refseq_protein_bacteria_archaea.faa.gz
gunzip refseq_protein_bacteria_archaea.faa.gz
```

3.2. Run hmmsearch:
```bash
hmmpress PF05147.hmm # hmmpress step is required for hmmsearch to work.
hmmsearch -A PF05147_hits.sto PF05147.hmm refseq_protein_bacteria_archaea.faa > PF05147.out
```

Output the hits from hmmsearch to a fasta file:
```bash
esl-reformat fasta PF05147_hits.sto > PF05147_hits.fasta
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

> **Note**: Change all lowercase amino acids to uppercase (`PF05147_hits.fasta` >> `PF05147_hits_upper.fasta`).

Extract hmmsearch outputs:
```bash
awk '/^ *[^- ]/ {print $9, $1}' class3_LanKC_hmmsearch.txt > class3_LanKC_hmmsearch_evalues.txt
```
```bash
awk '/^ *[^- ]/ {print $9, $1}' class4_LanL_hmmsearch.txt > class4_LanL_hmmsearch_evalues.txt
```

## RODEO (Rapid ORF Description and Evaluation Online) analysis

The cyclase hits within 200–500 amino acids were first queried against a previous excel dataset in reported in [Precursor peptide-targeted mining of more than one hundred thousand genomes expands the lanthipeptide natural product family (Walker et al. 2020)](https://link.springer.com/article/10.1186/s12864-020-06785-7#Sec19).

The remaining uncharacterized hits were split into nine text files with 1000 max sequence ids in each file and the text files were submitted to [RODEO Webtool 2.0](https://webtool.ripp.rodeo/) for analysis. All `main_co_occur.csv` outputs from jobs were combined into `main_co_occur_all_.csv` and the cyclase sequences were classified using this [jupyter notebook](https://github.com/BrooksResearchGroup-UM/latent_space_paper/blob/main/notebooks/rodeo_classification.ipynb).

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

For the simulated dataset, `train_simulated.py` was run on HPC clusters by:
```bash
sbatch ./scripts/VAE_train_simulated.sh
```
For the cyclase dataset, `train_cyclase.py` was run on HPC clusters by:
```bash
sbatch ./scripts/VAE_train_cyclase.sh
```
Use [Optuna](https://github.com/optuna/optuna) for hyperparameter optimization:
```bash
sbatch ./scripts/VAE_train_optuna.sh
```
> **Note**: The example here is for the training on the cyclase dataset.

For the FDMO dataset, `train_FDMO.py` was run on HPC clusters by:
```bash
sbatch ./scripts/VAE_train_FDMO.sh
```

## Obtain the embeddings using protein language models

For ESM-1b and ESM-2 (650M/3B):

Clone the repository of [esm](https://github.com/facebookresearch/esm/tree/main) and install esm by following the instructions. I created a separate environment for esm.

Manually download the models using `wget` ([ESM-1b](https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt), [ESM-2-650M](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt), [ESM-2-3B](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt)) to `~/.cache/torch/hub/checkpoints` if needed.

Embeddings were obtained by running `extract.py` on HPC clusters:
```bash
sbatch ./scripts/esm_embedding.sh
```

For ProtT5-XL:

Install [ProtTrans](https://github.com/agemagician/ProtTrans) by following the instructions. I created a separate environment for ProtTrans.

Since HPC compute nodes lack internet access, download the ProtT5-XL model first by running Steps 1 and 2 in this [jupyter notebook](./notebooks/ProtT5-XL-UniRef50.ipynb).

Embeddings were obtained by running `prott5_embedder.py` on HPC clusters:
```bash
sbatch ./scripts/prott5_embedding.sh
```

## Generate sequence similarity networks (SSNs)

SSNs were generated by [EFI - Enzyme Similarity Tool](https://efi.igb.illinois.edu/efi-est/), submitted through Option C - FASTA with the default setting. Initial SSN results were finalized using customized alignment score (AS) for downstream clustering. Clustering outputs (XGMML) were downloaded, reformatted using this [jupyter notebook](https://github.com/BrooksResearchGroup-UM/latent_space_paper/blob/main/notebooks/ssn_analysis.ipynb), and visualized by [Cytoscape](https://cytoscape.org/). Clustering results were saved as CSV files for analysis.