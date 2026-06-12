import sys
sys.path.append("/home/cdchiang/vae/latent_space_paper")

import pickle
import numpy as np
import torch
import torch.optim as optim
import optuna
from torch.utils.data import DataLoader
from src.VAE_model import *

print("CUDA Available:", torch.cuda.is_available())

def objective(trial, seq_msa_binary, seq_weight, seq_keys, idx_subset, num_seq, len_protein, num_res_type):
    # Fixed hyperparameters
    num_epoches = 200
    dim = 10
    K = len(idx_subset)
    
    # Hyperparameters tuned
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [128, 256, 512, 1024, 2048, 4096, 8192])

    encode_layer = [500, 100]
    decode_layer = [100, 500]

    elbo_all_list = []

    for k in range(K):
        # Build a VAE model
        vae = VAE(21, dim, len_protein * num_res_type, 
                encoder_num_hidden_units=encode_layer, decoder_num_hidden_units=decode_layer)
        vae.cuda()

        optimizer = optim.Adam(vae.parameters(), weight_decay=weight_decay, lr=lr)

        # Collect training and validation data indices
        validation_idx = idx_subset[k]
        validation_idx.sort()
        
        train_idx = np.array(list(set(range(num_seq)) - set(validation_idx)))
        train_idx.sort()

        train_msa = torch.from_numpy(seq_msa_binary[train_idx, ])
        validation_msa = torch.from_numpy(seq_msa_binary[validation_idx, ])

        train_weight = torch.from_numpy(seq_weight[train_idx])
        train_weight = train_weight / torch.sum(train_weight)
        
        equal_weight = 1.0 / len(validation_idx)
        validation_weight = torch.full((len(validation_idx),), equal_weight)

        train_key = [seq_keys[i] for i in train_idx]
        validation_key = [seq_keys[j] for j in validation_idx]
        
        train_dataset = MSA_Dataset(train_msa, train_weight, train_key)
        validation_dataset = MSA_Dataset(validation_msa, validation_weight, validation_key)
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=64, shuffle=False)

        # Training loop
        vae.train()
        for epoch in range(num_epoches):
            batch_loss = []
            for train_msa_batch, train_weight_batch, _ in train_dataloader:
                train_msa_batch = train_msa_batch.cuda()  
                train_weight_batch = train_weight_batch.cuda()

                optimizer.zero_grad() 
                loss = (-1) * vae.compute_weighted_elbo(train_msa_batch, train_weight_batch)
                batch_loss.append(loss.item())
                
                loss.backward()  
                optimizer.step() 

            # Optional: Add Optuna Pruning here if you want to stop bad trials early
            
        # Evaluation loop
        vae.eval()
        elbo_on_validation_data_list = []
        for validation_msa_batch, _, _ in validation_dataloader:
            with torch.no_grad():
                validation_msa_batch = validation_msa_batch.cuda()
                elbo = vae.compute_elbo_with_multiple_samples(validation_msa_batch, 10)            
                elbo_on_validation_data_list.append(elbo.cpu().data.numpy())

        elbo_on_validation_data = np.concatenate(elbo_on_validation_data_list)    
        elbo_all_list.append(elbo_on_validation_data)
        
    elbo_all = np.concatenate(elbo_all_list)
    elbo_mean = (-1) * np.mean(elbo_all)

    return elbo_mean

if __name__ == '__main__':

    input_file = "../data/processed/training/cyclase" # cyclase dataset

    # Load data
    with open("{}/seq_msa_binary.pkl".format(input_file), 'rb') as file_handle:
        seq_msa_binary = pickle.load(file_handle)    
    num_seq = seq_msa_binary.shape[0]
    len_protein = seq_msa_binary.shape[1]
    num_res_type = seq_msa_binary.shape[2]
    seq_msa_binary = seq_msa_binary.reshape((num_seq, -1)).astype(np.float32)

    with open("{}/seq_weight.pkl".format(input_file), 'rb') as file_handle:
        seq_weight = pickle.load(file_handle).astype(np.float32)

    with open("{}/keys_list.pkl".format(input_file), 'rb') as file_handle:
        seq_keys = pickle.load(file_handle)

    # Create splits
    K = 5
    num_seq_subset = num_seq // K + 1
    idx_subset = []
    
    np.random.seed(42) # Set seed for reproducibility across script runs
    random_idx = np.random.permutation(range(num_seq))
    for i in range(K):
        idx_subset.append(random_idx[i*num_seq_subset:(i+1)*num_seq_subset])

    # Configure Optuna and run pruner
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,    # Let the first 5 trials run to completion to build a baseline
        n_warmup_steps=50,     # Don't prune any trial until it reaches at least epoch 50
        interval_steps=1       # Check for pruning every step (epoch)
    )
    
    study = optuna.create_study(direction='minimize', pruner=pruner)  
    
    # Pass the loaded data into the objective using a lambda
    study.optimize(lambda trial: objective(
        trial, seq_msa_binary, seq_weight, seq_keys, idx_subset, num_seq, len_protein, num_res_type
    ), n_trials=100)

    print("")
    print(f'Best parameters: {study.best_params}')
    print(f'Best score: {study.best_value}')