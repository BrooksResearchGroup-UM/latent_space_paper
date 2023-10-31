import time
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pickle
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os

# Define model

class MSA_Dataset(Dataset):
    '''
    Dataset class for multiple sequence alignment.
    '''
    
    def __init__(self, seq_msa_binary, seq_weight, seq_keys):
        '''
        seq_msa_binary: a two dimensional np.array. 
                        size: [num_of_sequences, length_of_msa*num_amino_acid_types]
        seq_weight: one dimensional array. 
                    size: [num_sequences]. 
                    Weights for sequences in a MSA. 
                    The sum of seq_weight has to be equal to 1 when training latent space models using VAE
        seq_keys: name of sequences in MSA
        '''
        super(MSA_Dataset).__init__()
        self.seq_msa_binary = seq_msa_binary
        self.seq_weight = seq_weight
        self.seq_keys = seq_keys
        
    def __len__(self):
        assert(self.seq_msa_binary.shape[0] == len(self.seq_weight))
        assert(self.seq_msa_binary.shape[0] == len(self.seq_keys))        
        return self.seq_msa_binary.shape[0]
    
    def __getitem__(self, idx):
        return self.seq_msa_binary[idx, :], self.seq_weight[idx], self.seq_keys[idx]

class VAE(nn.Module):

    def __init__(self, num_aa_type, dim_latent_vars, dim_msa_vars, num_hidden_units):
        super(VAE, self).__init__()

        ## num of amino acid types
        self.num_aa_type = num_aa_type

        ## dimension of latent space
        self.dim_latent_vars = dim_latent_vars

        ## dimension of binary representation of sequences
        self.dim_msa_vars = dim_msa_vars

        ## num of hidden neurons in encoder and decoder networks
        self.num_hidden_units = num_hidden_units

        ## encoder
        self.encoder_linears = nn.ModuleList()
        self.encoder_linears.append(nn.Linear(dim_msa_vars, num_hidden_units[0]))
        for i in range(1, len(num_hidden_units)):
            self.encoder_linears.append(nn.Linear(num_hidden_units[i-1], num_hidden_units[i]))
            #self.encoder_linears.append(nn.BatchNorm1d(num_hidden_units[i]))
        self.encoder_mu = nn.Linear(num_hidden_units[-1], dim_latent_vars, bias = True)
        self.encoder_logsigma = nn.Linear(num_hidden_units[-1], dim_latent_vars, bias = True)

        ## decoder
        self.decoder_linears = nn.ModuleList()
        self.decoder_linears.append(nn.Linear(dim_latent_vars, num_hidden_units[-1]))
        for i in range(1, len(num_hidden_units)):
            self.decoder_linears.append(nn.Linear(num_hidden_units[-i], num_hidden_units[-i-1]))
            #self.decoder_linears.append(nn.BatchNorm1d(num_hidden_units[-i-1]))
        self.decoder_linears.append(nn.Linear(num_hidden_units[0], dim_msa_vars))

    def encoder(self, x):
        '''
        encoder transforms x into latent space z
        '''
        
        h = x
        for T in self.encoder_linears:
            h = T(h)
            h = torch.tanh(h)
        mu = self.encoder_mu(h)
        sigma = torch.exp(self.encoder_logsigma(h))
        return mu, sigma

    def decoder(self, z):
        '''
        decoder transforms latent space z into p, which is the log probability  of x being 1.
        '''
        
        h = z
        for i in range(len(self.decoder_linears)-1):
            h = self.decoder_linears[i](h)
            h = torch.tanh(h)
        h = self.decoder_linears[-1](h)

        fixed_shape = tuple(h.shape[0:-1])
        h = torch.unsqueeze(h, -1)
        h = h.view(fixed_shape + (-1, self.num_aa_type))
          
        log_p = F.log_softmax(h, dim = -1)
        log_p = log_p.view(fixed_shape + (-1,))
        
        return log_p

    def compute_weighted_elbo(self, x, weight):
        ## sample z from q(z|x)
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(sigma) 
        '''Returns a tensor with the same size as input that is filled with 
        random numbers from a normal distribution with mean 0 and variance 1.'''
        z = mu + sigma*eps

        ## compute log p(x|z)
        log_p = self.decoder(z)
        log_PxGz = torch.sum(x*log_p, -1)

        ## compute elbo
        elbo = log_PxGz - KLD_beta*torch.sum(0.5*(sigma**2 + mu**2 - 2*torch.log(sigma) - 1), -1)
        weight = weight / torch.sum(weight)
        elbo = torch.sum(elbo*weight)
        
        return elbo

    def compute_elbo_with_multiple_samples(self, x, num_samples):
        with torch.no_grad():
            x = x.expand(num_samples, x.shape[0], x.shape[1])
            mu, sigma = self.encoder(x)
            eps = torch.randn_like(mu)
            z = mu + sigma * eps
            log_Pz = torch.sum(-0.5*z**2 - 0.5*torch.log(2*z.new_tensor(np.pi)), -1)
            log_p = self.decoder(z)
            log_PxGz = torch.sum(x*log_p, -1)
            log_Pxz = log_Pz + log_PxGz

            log_QzGx = torch.sum(-0.5*(eps)**2 -
                                 0.5*torch.log(2*z.new_tensor(np.pi))
                                 - torch.log(sigma), -1)
            log_weight = (log_Pxz - log_QzGx).detach().data
            log_weight = log_weight.double()
            log_weight_max = torch.max(log_weight, 0)[0]
            log_weight = log_weight - log_weight_max
            weight = torch.exp(log_weight)
            elbo = torch.log(torch.mean(weight, 0)) + log_weight_max
            return elbo      

# Model hyperparameters
cuda = True
DEVICE = torch.device("cuda" if cuda else "cpu")

# Number of hidden layers and number of neurons per layer
layers = [2048, 1024, 128, 32]

# Batch size
batch_size_train = 256

# Control parameter in optimizer
weight_decay  = 0.0005

# Learning rate
lr = 0.0005

# Weight applied to KLD in elbo
KLD_beta = 1

# Number of epoches to run optimization
num_epoches = 1000

# Specify the data files for analysis
dim = 2

# Training dataset:

input_file = '../data/encoding'

# Output folder:

output_path = '../model/PF01494_2'

## Reconstruction accuracy
def check_recon(validation):
    x = validation.to(DEVICE)
    #optimizer.zero_grad()
    mu, sigma = vae.encoder(x)
    # Add decoder here to check sequence decoding
    log_p = vae.decoder(mu)
    x_hat = torch.exp(log_p)
    x = x.cpu().numpy()
    x_hat = x_hat.detach().cpu().numpy()
    x = np.reshape(x, (x.shape[0], len_protein, num_res_type))
    x_hat = np.reshape(x_hat, (x_hat.shape[0], len_protein, num_res_type))
    x = x.argmax(-1)
    x_hat = x_hat.argmax(-1)
    differences = np.abs(x-x_hat)
    differences = np.clip(differences,0,1)
    total_differences = np.sum(differences, axis=1)
    reconstruction_mean = np.mean(1- total_differences/len_protein)*100
    return reconstruction_mean

# Train the model using different seeds if needed
seed_list = [19]
for seed in seed_list:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # Read training data
    with open("{}/seq_msa_binary.pkl".format(input_file), 'rb') as file_handle:
        seq_msa_binary = pickle.load(file_handle)
    num_seq = seq_msa_binary.shape[0] # an array with 3 dimensions, shape[0] is the number of sequences
    len_protein = seq_msa_binary.shape[1] # shape[1] is the length of all sequnces
    num_res_type = seq_msa_binary.shape[2] # shape[2] is the number of residues
    seq_msa_binary = seq_msa_binary.reshape((num_seq, -1)) # an array with "num_seq" rows and "len_protein * num_res_type" columns
    seq_msa_binary = seq_msa_binary.astype(np.float32)

    print('training_num_seq =', num_seq)
    print('training_len_protein =', len_protein)
    print('training_num_res_type =', num_res_type)

    with open("{}/seq_weight.pkl".format(input_file), 'rb') as file_handle:
        seq_weight = pickle.load(file_handle)
    seq_weight = seq_weight.astype(np.float32) # sequence weights for each sequence

    with open("{}/keys_list.pkl".format(input_file), 'rb') as file_handle:
        seq_keys = pickle.load(file_handle) # sequence ID for each sequence

    # Read testing data
    with open("{}/test_seq_msa_binary.pkl".format(input_file), 'rb') as file_handle:
        t_seq_msa_binary = pickle.load(file_handle)
    t_num_seq = t_seq_msa_binary.shape[0] # an array with 3 dimensions, shape[0] is the number of sequences
    t_len_protein = t_seq_msa_binary.shape[1] # shape[1] is the length of all sequnces
    t_num_res_type = t_seq_msa_binary.shape[2] # shape[2] is the number of residues
    t_seq_msa_binary = t_seq_msa_binary.reshape((t_num_seq, -1)) # an array with "num_seq" rows and "len_protein * num_res_type" columns
    t_seq_msa_binary = t_seq_msa_binary.astype(np.float32)

    print('testing_num_seq =', t_num_seq)
    print('testing_len_protein =', t_len_protein)
    print('testing_num_res_type =', t_num_res_type)

    with open("{}/test_seq_weight.pkl".format(input_file), 'rb') as file_handle:
        t_seq_weight = pickle.load(file_handle)
    t_seq_weight = t_seq_weight.astype(np.float32) # sequence weights for each sequence

    with open("{}/test_keys_list.pkl".format(input_file), 'rb') as file_handle:
        t_seq_keys = pickle.load(file_handle) # sequence ID for each sequence

    # Set up the testing dataset
    # Testing dataset
    testing_msa = torch.from_numpy(t_seq_msa_binary)
    testing_weight = torch.from_numpy(t_seq_weight)
    testing_weight = testing_weight/torch.sum(testing_weight)
    testing_keys = t_seq_keys

    # Testing dataloader
    testing_data = MSA_Dataset(testing_msa, testing_weight, testing_keys)
    testing_data_loader = DataLoader(testing_data, shuffle=False, batch_size=100)
    
    # Print some info
    print('''
    A VAE will be constructed based on the data in {:} with {:} of {:} a.a. processed protein sequences. 
    The VAE will be constructed with an input vector dimension of {:}, with {:} hidden layers with {:} neurons each and a latent space dimension of {:}. 
    Optimization will proceed via the {:} optimizer using a {:.4f} weight decay, and the loss function will modulate the KLD contribution with the factor {:.5f}. 
    {:} epoches will be carried out to optimize the model. Batch size is {:}. Learning rate is {:}. All processing will be done on the {:}. Random seed is {:}.'''\
    .format(input_file, num_seq, len_protein, len_protein*num_res_type, len(layers), layers, dim, 'Adam', weight_decay, KLD_beta, num_epoches, batch_size_train, lr, DEVICE, seed))
    print(' ')

    # Set up the training dataset and start training
    # Training dataset
    train_msa = torch.from_numpy(seq_msa_binary)
    train_weight = torch.from_numpy(seq_weight)
    train_weight = train_weight/torch.sum(train_weight)
    train_keys = seq_keys

    # Training dataloader    
    train_data = MSA_Dataset(train_msa, train_weight, train_keys)
    train_data_loader = DataLoader(train_data, shuffle=True, batch_size = batch_size_train)
    vae = VAE(num_aa_type = 21,
          dim_latent_vars = dim,
          dim_msa_vars = len_protein*num_res_type,
          num_hidden_units = layers)
    
    # Move the VAE onto a GPU
    vae.to(DEVICE)
    
    # Build the Adam optimizer
    optimizer = optim.Adam(vae.parameters(), lr = lr, weight_decay = weight_decay)

    train_loss_mean_list = []
    epoch_list = []
    testing_recon_mean_list = []
    train_recon_mean_list = []
    for epoch in range(num_epoches):
        train_loss_list = []
        running_loss = 0
        start = time.time()
        for idx, data in enumerate(train_data_loader):
            msa, weight, key = data
            msa = msa.cuda()
            weight = weight.cuda()
            loss = (-1)*vae.compute_weighted_elbo(msa, weight)
            optimizer.zero_grad()
            loss.backward()        
            optimizer.step()
            # loss.item() returns mean, so it should be multiplied by batch size to get the total 
            running_loss += loss.item() * msa.size(0) 
        end = time.time()
        epoch_loss = running_loss / len(train_data_loader.dataset)
        train_loss_mean_list.append(epoch_loss)

        if epoch ==0 or (epoch + 1) % 100 ==0:
            print('Time through {}th epoch: {:.1f}'.format(epoch+1, end-start))
            print("Epoch", epoch + 1, "complete!", "Average Loss: {:.1f}".format(epoch_loss))
            print("Start the reconstruction accuracy calculation of {} epoches".format(epoch+1))
            print("-"*60)
            epoch_list.append(int(epoch))

            # testing dataset reconstruction accuracy
            testing_recon_list = []
            for idx, data in enumerate(testing_data_loader):
                testing_msa, t_weight, t_key = data
                testing_recon_list.append(check_recon(testing_msa))
            testing_recon_mean_list.append(np.mean(testing_recon_list))
        
            # train dataset reconstruction accuracy
            train_recon_list = []
            for idx, data in enumerate(train_data_loader):
                train_msa, weight, key = data
                train_recon_list.append(check_recon(train_msa))
            train_recon_mean_list.append(np.mean(train_recon_list))
        
            print("testing reconstruction accuracy: {:.1f}".format(np.mean(testing_recon_list)))
            print("training reconstruction accuracy: {:.1f}".format(np.mean(train_recon_list)))
            print("-"*60)
            
            torch.cuda.empty_cache()

    # Save the model
    vae.cpu()
    torch.save(vae.state_dict(), "{}/NoCV_vae_d{}_lr{}_w{}_b{}_l{}_beta{}_{}epoch_seed{}.model".format(output_path, dim, len(layers), str(weight_decay), batch_size_train, lr, KLD_beta, num_epoches, seed))

    print("Final testing reconstruction accuracy: {:.1f}".format(testing_recon_mean_list[-1]))
    print("Final training reconstruction accuracy: {:.1f}".format(train_recon_mean_list[-1]))

    # Load the model
    vae.cuda()
    vae.load_state_dict(torch.load("{}/NoCV_vae_d{}_lr{}_w{}_b{}_l{}_beta{}_{}epoch_seed{}.model".format(output_path, dim, len(layers), str(weight_decay), batch_size_train, lr, KLD_beta, num_epoches, seed)))

    # Transform the training data to mu
    mu_list = []
    sigma_list = []
    key_list = []
    for idx, data in enumerate(train_data_loader):
        msa, weight, key = data
        with torch.no_grad():
            msa = msa.cuda()        
            mu, sigma = vae.encoder(msa)
            # gpu to cpu to numpy
            mu_list.append(mu.cpu().data.numpy())
            sigma_list.append(sigma.cpu().data.numpy())
            # key is a tuple
            key_list = key_list + list(key)

    mu = np.vstack(mu_list)
    sigma = np.vstack(sigma_list)

    # Transform the testing data to mu
    t_mu_list = []
    t_sigma_list = []
    t_key_list = []
    for idx, data in enumerate(testing_data_loader):
        t_msa, t_weight, t_key = data
        with torch.no_grad():
            t_msa = t_msa.cuda()
            t_mu, t_sigma = vae.encoder(t_msa)
            # gpu to cpu to numpy
            t_mu_list.append(t_mu.cpu().data.numpy())
            t_sigma_list.append(t_sigma.cpu().data.numpy())
            # key is a tuple
            t_key_list = t_key_list + list(t_key)

    t_mu = np.vstack(t_mu_list)
    t_sigma = np.vstack(t_sigma_list)

    # Save the reconstruction data
    with open("{}/NoCV_reconstruction_d{}_layer{}_w{}_b{}_l{}_beta{}_{}epoch_seed{}.pkl".format(output_path, dim, len(layers), str(weight_decay), batch_size_train, lr, KLD_beta, num_epoches, seed), 'wb') as file_handle:
        pickle.dump({'epoch': epoch_list, 'testing': testing_recon_mean_list, 'train': train_recon_mean_list}, file_handle)

    # Load the reconstruction data
    with open("{}/NoCV_reconstruction_d{}_layer{}_w{}_b{}_l{}_beta{}_{}epoch_seed{}.pkl".format(output_path, dim, len(layers), str(weight_decay), batch_size_train, lr, KLD_beta, num_epoches, seed), 'rb') as file_handle:
        reconstruction = pickle.load(file_handle)
    epoch_list = reconstruction['epoch']
    testing_recon_mean_list = reconstruction['testing']
    train_recon_mean_list = reconstruction['train']

    # Plot the reconstruction curves.
    plt.figure(0)
    plt.clf()
    plt.plot(epoch_list, testing_recon_mean_list, 'r', label="testing")
    plt.plot(epoch_list, train_recon_mean_list, 'b', label="training")
    plt.legend(loc="upper right")
    plt.xlim((0, epoch_list[-1]+10))
    plt.ylim((0, 100))
    plt.xlabel("Epoch")
    plt.ylabel("Percentage reconstruction")
    plt.tight_layout()
    plt.savefig("{}/NoCV_reconstruction_d{}_layer{}_w{}_b{}_l{}_beta{}_{}epoch_seed{}.png".format(output_path, dim, len(layers), str(weight_decay), batch_size_train, lr, KLD_beta, num_epoches, seed))

    # Save key, mu, and sigma
    with open("{}/NoCV_latent_space_d{}_layer{}_w{}_b{}_l{}_beta{}_{}epoch_seed{}.pkl".format(output_path, dim, len(layers), str(weight_decay), batch_size_train, lr, KLD_beta, num_epoches, seed), 'wb') as file_handle:
        pickle.dump({'key': key_list, 'mu': mu, 'sigma': sigma, 't_key': t_key_list, 't_mu': t_mu, 't_sigma': t_sigma}, file_handle) 

    # Load the latent space info
    with open("{}/NoCV_latent_space_d{}_layer{}_w{}_b{}_l{}_beta{}_{}epoch_seed{}.pkl".format(output_path, dim, len(layers), str(weight_decay), batch_size_train, lr, KLD_beta, num_epoches, seed), 'rb') as file_handle:
        latent_space = pickle.load(file_handle) 

    key = latent_space['key']
    mu = latent_space['mu']
    sigma = latent_space['sigma']
    t_key = latent_space['t_key'] 
    t_mu = latent_space['t_mu']
    t_sigma = latent_space['t_sigma']

    # Plot the latent space
    plt.figure(1)
    plt.clf()
    plt.plot(mu[:,0], mu[:,1], '.', alpha = 0.1, markersize = 2)
    plt.plot(t_mu[:,0], t_mu[:,1], '.', alpha = 0.1, markersize = 2)
    plt.xlim((-6,6))
    plt.ylim((-6,6))
    plt.xlabel("$Z_1$")
    plt.ylabel("$Z_2$")
    plt.tight_layout()
    plt.savefig("{}/NoCV_latent_mu_scatter_d{}_layer{}_w{}_b{}_l{}_beta{}_{}epoch_seed{}.png".format(output_path, dim, len(layers), str(weight_decay), batch_size_train, lr, KLD_beta, num_epoches, seed))







