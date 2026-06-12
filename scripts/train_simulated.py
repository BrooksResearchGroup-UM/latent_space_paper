import sys
sys.path.append("/home/cdchiang/vae/latent_space_paper")

import gc
import pickle
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.VAE_model import *

# Read multiple sequence alignment in binary representation
with open("../data/processed/training/simulated/msa_leaf_binary.pkl", 'rb') as file_handle:
    msa_binary = pickle.load(file_handle)    
num_seq = msa_binary.shape[0]
len_protein = msa_binary.shape[1]
num_res_type = msa_binary.shape[2]
msa_binary = msa_binary.reshape((num_seq, -1))
msa_binary = msa_binary.astype(np.float32)

# Each sequence has a label
with open("../data/processed/training/simulated/msa_leaf_keys.pkl", 'rb') as file_handle:
    msa_keys = pickle.load(file_handle)    

# Sequences in msa are weighted. Here sequences are assigned the same weights
msa_weight = np.ones(num_seq) / num_seq
msa_weight = msa_weight.astype(np.float32)
num_epoches = 1000

# K-fold cross validation
K = 5
num_seq_subset = num_seq // K + 1
idx_subset = []
random_idx = np.random.permutation(range(num_seq))
for i in range(K):
    idx_subset.append(random_idx[i*num_seq_subset:(i+1)*num_seq_subset])

# ELBO values on the validation data    
elbo_all_list = []
for k in range(K):

    # Build a VAE model with random parameters
    vae = VAE(num_aa_type = 20,
          dim_latent_vars = 2,
          dim_msa_vars = len_protein*num_res_type,
          encoder_num_hidden_units = [100], 
          decoder_num_hidden_units = [100]
          )

    ## Move the VAE onto a GPU
    vae.cuda()

    # Build the Adam optimizer
    weight_decay = 0.01
    lr = 0.001
    optimizer = optim.Adam(vae.parameters(), weight_decay=weight_decay, lr=lr)

    # Collect training and valiation data indices
    validation_idx = idx_subset[k]
    validation_idx.sort()
    
    train_idx = np.array(list(set(range(num_seq)) - set(validation_idx)))
    train_idx.sort()

    train_msa = torch.from_numpy(msa_binary[train_idx, ])
    validation_msa = torch.from_numpy(msa_binary[validation_idx, ])

    train_weight = torch.from_numpy(msa_weight[train_idx])
    validation_weight = torch.from_numpy(msa_weight[validation_idx])

    # Calculate the equal weight for all validation samples
    # equal_weight = 1.0 / len(validation_idx)
    # Create a tensor of these equal weights
    # validation_weight = torch.full((len(validation_idx),), equal_weight)

    train_key = [msa_keys[i] for i in train_idx]
    validation_key = [msa_keys[j] for j in validation_idx]
    
    # Create datasets
    train_dataset = MSA_Dataset(train_msa, train_weight, train_key)
    validation_dataset = MSA_Dataset(validation_msa, validation_weight, validation_key)
    
    # Create dataloaders
    batch_size = len(train_idx)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=64, shuffle=False)

    train_loss_list = []
    for epoch in range(num_epoches):

        batch_loss = []
        for train_msa, train_weight, train_key in train_dataloader:
            train_msa = train_msa.cuda()  
            train_weight = train_weight.cuda()

            optimizer.zero_grad() # Reset gradients from last step

            loss = (-1)*vae.compute_weighted_elbo(train_msa, train_weight)
            batch_loss.append(loss.item())
            
            loss.backward() # Compute gradients    
            optimizer.step() # Update weights

        train_epoch_loss = np.mean(batch_loss)  # Compute average loss for this epoch
        train_loss_list.append(train_epoch_loss)  # Add average loss to the list of epoch losses
        torch.cuda.empty_cache()

        if (epoch + 1) % 50 ==0:
            print("Fold: {}, Epoch: {:>4}, loss: {:>4.2f}".format(k, epoch+1, train_epoch_loss), flush = True)

    torch.save(vae.state_dict(), "../models/simulated/onelayer/vae_fold{}_{:.2f}.model".format(k, weight_decay))

    # Model evaluation
    vae.cuda()
    
    elbo_on_validation_data_list = []
    for validation_msa, validation_weight, validation_key in validation_dataloader:
        with torch.no_grad():
            validation_msa = validation_msa.cuda()
            elbo = vae.compute_elbo_with_multiple_samples(validation_msa, 10)            
            elbo_on_validation_data_list.append(elbo.cpu().data.numpy())
            del validation_msa

        gc.collect()
        torch.cuda.empty_cache()

    elbo_on_validation_data = np.concatenate(elbo_on_validation_data_list)    
    elbo_all_list.append(elbo_on_validation_data)

    print("Finish the {}th fold validation".format(k))
    print("="*60)

    elbo_all = np.concatenate(elbo_all_list)
    elbo_mean = (-1)*np.mean(elbo_all)
    print("loss: {:>4.2f}".format(elbo_mean))
    print("\n") 
