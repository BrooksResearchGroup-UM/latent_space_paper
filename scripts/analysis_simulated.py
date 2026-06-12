import sys
sys.path.append("/home/cdchiang/vae/latent_space_paper")

import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from src.VAE_model import *
from ete3 import Tree

mpl.use("Agg")
mpl.rc('font', size = 14)
mpl.rc('axes', titlesize = 'large', labelsize = 'large')
mpl.rc('xtick', labelsize = 'large')
mpl.rc('ytick', labelsize = 'large')

# Read data
with open("../data/processed/training/simulated/msa_binary.pkl", 'rb') as file_handle:
    msa_binary = pickle.load(file_handle)    
num_seq = msa_binary.shape[0]
len_protein = msa_binary.shape[1]
num_res_type = msa_binary.shape[2]
msa_binary = msa_binary.reshape((num_seq, -1))
msa_binary = msa_binary.astype(np.float32)

with open("../data/processed/training/simulated/msa_keys.pkl", 'rb') as file_handle:
    msa_keys = pickle.load(file_handle)    

msa_weight = np.ones(num_seq) / num_seq
msa_weight = msa_weight.astype(np.float32)

batch_size = num_seq
train_data = MSA_Dataset(msa_binary, msa_weight, msa_keys)
train_data_loader = DataLoader(train_data, batch_size = batch_size)
vae = VAE(20, 2, len_protein * num_res_type, [100], [100])
vae.cuda()
vae.load_state_dict(torch.load("../models/simulated/onelayer_n100_weight0.01/vae_fold3_0.01_best.model"))

mu_list = []
sigma_list = []
for idx, data in enumerate(train_data_loader):
    msa, weight, key = data
    with torch.no_grad():
        msa = msa.cuda()        
        mu, sigma = vae.encoder(msa)
        mu_list.append(mu.cpu().data.numpy())
        sigma_list.append(sigma.cpu().data.numpy())

mu = np.vstack(mu_list)
sigma = np.vstack(sigma_list)

with open("../models/simulated/onelayer_n100_weight0.01/latent_space.pkl", 'wb') as file_handle:
    pickle.dump({'key': key, 'mu': mu, 'sigma': sigma}, file_handle)    

# Plot latent space    
t = Tree("../data/processed/tree/simulated/random_tree.newick", format = 1)
num_leaf = len(t)
t.name = str(num_leaf)

leaf_idx = []
ancestral_idx = []
for i in range(len(key)):
    if int(key[i]) < num_leaf:
        leaf_idx.append(i)
    else:
        ancestral_idx.append(i)

plt.figure(0)
plt.clf()
plt.plot(mu[leaf_idx,0], mu[leaf_idx,1], 'b.', alpha = 0.5, markersize = 2)
plt.xlim((-6.5,6.5))
plt.ylim((-6.5,6.5))
plt.xlabel("$Z_1$")
plt.ylabel("$Z_2$")
plt.tight_layout()
plt.savefig("../models/simulated/onelayer_n100_weight0.01/latent_mu_leaf.pdf")

plt.figure(1)
plt.clf()
plt.plot(mu[ancestral_idx,0], mu[ancestral_idx,1], 'r.', alpha = 0.5, markersize = 2)
plt.xlim((-6.5,6.5))
plt.ylim((-6.5,6.5))
plt.xlabel("$Z_1$")
plt.ylabel("$Z_2$")
plt.tight_layout()
plt.savefig("../models/simulated/onelayer_n100_weight0.01/latent_mu_ancestral.pdf")

plt.figure(2)
plt.clf()
plt.plot(mu[ancestral_idx,0], mu[ancestral_idx,1], 'r.', alpha = 0.5, markersize = 1, label = 'ancestral')
plt.plot(mu[leaf_idx,0], mu[leaf_idx,1], 'b.', alpha = 0.5, markersize = 1, label = 'leaf')
plt.xlim((-6.5,6.5))
plt.ylim((-6.5,6.5))
plt.xlabel("$Z_1$")
plt.ylabel("$Z_2$")
plt.legend(markerscale = 3)
plt.tight_layout()
plt.savefig("../models/simulated/onelayer_n100_weight0.01/latent_mu_all.pdf")