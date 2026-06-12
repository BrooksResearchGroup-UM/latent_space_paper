import os
import sys
sys.path.append("/home/cdchiang/vae/latent_space_paper")

import argparse
import gc
import random
import pickle
import numpy as np
import torch
import torch.optim as optim
from datetime import datetime
from sys import exit
from torch.utils.data import DataLoader
from src.VAE_model import *

parser = argparse.ArgumentParser(description='Parameters for training the model')
parser.add_argument('--num_epoch', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--weight_decay', type = float)
parser.add_argument('--lr', type = float)
parser.add_argument('--dim', type = int)
parser.add_argument('--encode_layer', nargs='+', type = int)
parser.add_argument('--decode_layer', nargs='+', type = int)
parser.add_argument('--seed', type = int)
parser.add_argument('--input_file', type = str)
args = parser.parse_args()

num_epoches = args.num_epoch
batch_size = args.batch_size
weight_decay = args.weight_decay
lr = args.lr
dim = args.dim
encode_layer = args.encode_layer
decode_layer = args.decode_layer
seed = args.seed
input_file = args.input_file

def set_seed(seed):

    # Enforce strict reproducibility across numpy, torch, and python.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Safe for multi-GPU setups
    
    # Force CuDNN to use deterministic algorithms (critical for GPU reproducibility).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call the seeding function here to lock down the entire run
set_seed(seed)

# Define your parameters
params = {
    'epochs': num_epoches,
    'weight_decay': weight_decay,
    'batch_size': batch_size,
    'learning_rate': lr,
    'dim': dim,
    'encode_layer': encode_layer,
    'decode_layer': decode_layer,
    'seed': seed,
    'input_file_folder': input_file
}

# Get the current date and time
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Define an output path
output_path = "../models/training_{}".format(timestamp)

# Path to the splits file
splits_file = f"{output_path}/kfold_splits.pkl"

# Create the directory, don't raise an exception if it already exists
os.makedirs(output_path, exist_ok=True)

# Open the text file in 'append' mode
with open('{}/model_log.txt'.format(output_path), 'a') as f:
    # Write the date and time to the file
    f.write(f'Training run at {timestamp}\n')
    # Write each parameter and its value
    for param, value in params.items():
        f.write(f'{param}: {value}\n')
    # Write a separator for readability
    f.write('\n')

# Read data
with open("{}/seq_msa_binary.pkl".format(input_file), 'rb') as file_handle:
    seq_msa_binary = pickle.load(file_handle)    
num_seq = seq_msa_binary.shape[0]
len_protein = seq_msa_binary.shape[1]
num_res_type = seq_msa_binary.shape[2]
seq_msa_binary = seq_msa_binary.reshape((num_seq, -1))
seq_msa_binary = seq_msa_binary.astype(np.float32)

with open("{}/seq_weight.pkl".format(input_file), 'rb') as file_handle:
    seq_weight = pickle.load(file_handle)
seq_weight = seq_weight.astype(np.float32)

with open("{}/keys_list.pkl".format(input_file), 'rb') as file_handle:
    seq_keys = pickle.load(file_handle)

#### training model with K-fold cross validation
## split the data index 0:num_seq-1 into K sets
## each set is just a set of indices of sequences.
## in the kth traing, the kth subsets of sequences are used
## as validation data and the remaining K-1 sets are used
## as training data

# K-fold cross validation
K = 5
num_seq_subset = num_seq // K + 1
idx_subset = []
np.random.seed(seed) # Add random seed
random_idx = np.random.permutation(range(num_seq))
for i in range(K):
    idx_subset.append(random_idx[i*num_seq_subset:(i+1)*num_seq_subset])

# Save the subset to a pkl file
with open(splits_file, 'wb') as f:
    pickle.dump(idx_subset, f)

print("Splits successfully saved to kfold_splits.pkl")
print("\n")

# ELBO values on the validation data    
elbo_all_list = []

for k in range(K):

    # Build a VAE model with random parameters
    vae = VAE(21, dim, len_protein * num_res_type, 
            encoder_num_hidden_units = encode_layer, decoder_num_hidden_units = decode_layer)

    # Move the VAE onto a GPU
    vae.cuda()

    # Build the Adam optimizer
    optimizer = optim.Adam(vae.parameters(), weight_decay=weight_decay, lr=lr)

    # Collect training and valiation data indices
    validation_idx = idx_subset[k]
    validation_idx.sort()
    
    train_idx = np.array(list(set(range(num_seq)) - set(validation_idx)))
    train_idx.sort()

    train_msa = torch.from_numpy(seq_msa_binary[train_idx, ])
    validation_msa = torch.from_numpy(seq_msa_binary[validation_idx, ])

    train_weight = torch.from_numpy(seq_weight[train_idx])
    train_weight = train_weight/torch.sum(train_weight)
    
    # Calculate the equal weight for all validation samples
    equal_weight = 1.0 / len(validation_idx)
    # Create a tensor of these equal weights
    validation_weight = torch.full((len(validation_idx),), equal_weight)

    train_key = [seq_keys[i] for i in train_idx]
    validation_key = [seq_keys[j] for j in validation_idx]
    
    # Create datasets
    train_dataset = MSA_Dataset(train_msa, train_weight, train_key)
    validation_dataset = MSA_Dataset(validation_msa, validation_weight, validation_key)
    
    # Create dataloaders
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

        epoch_loss = np.mean(batch_loss)  # Compute average loss for this epoch
        train_loss_list.append(epoch_loss)  # Add average loss to the list of epoch losses
        torch.cuda.empty_cache()

        if (epoch + 1) % 50 ==0:
            print("Fold: {}, Epoch: {:>4}, loss: {:>4.2f}".format(k, epoch+1, epoch_loss), flush = True)

    # Cope trained model to cpu and save it
    vae.cpu()
    torch.save(vae.state_dict(), "{}/vae_fold_{}.model".format(output_path, k))
    
    print("Finish the {}th fold training".format(k))
    print("="*60)
    print("\n")
    
    print("Start the {}th fold validation".format(k))
    print("-"*60)

    # Model evaluation
    vae.cuda()
    
    elbo_on_validation_data_list = []
    for validation_msa, validation_weight, validation_key in validation_dataloader:
        with torch.no_grad():
            validation_msa = validation_msa.cuda()
            elbo = vae.compute_elbo_with_multiple_samples(validation_msa, 1000)            
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

exit()

