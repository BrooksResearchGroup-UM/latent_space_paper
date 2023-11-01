# latent_space_paper

### Multiple sequence alignment using HMMER
#### Installation
1. Download HMMER from the official [HMMER website](http://hmmer.org/download.html).
2. Follow the installation instructions provided on the website or in the downloaded package.
3. To verify that HMMER has been installed successfully, run the following command in your terminal or command line:
   ```bash
   hmmalign --version

#### Run hmmalign
   ```bash
   hmmalign ./data/PF01494_seed.hmm ./data/PF01494_input.fasta > ./data/PF01494_MSA.fasta
