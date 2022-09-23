"""
Our next task, will be to learn identifiable LDM representations,  𝐰𝑖∈ℝ𝐷 ,
for each node  𝑖∈  in a lower dimensional space ( 𝐷≪𝑁 ) such that the embeddings also
convey information about latent community memberships. This will yield the Hybrid
Membership-Latent Distance Model (HM-LDM) [4] framework.

For that, we will concentrate on mapping the nodes into the unit  𝐷 -simplex set,  Δ𝐷⊂ℝ𝐷+1+ .
Therefore, the extracted node embeddings (if identifiable) can convey information about (latent)
community memberships.

Most GRL approaches do not provide identifiable or unique solution guarantees, so their
interpretation highly depends on the initialization of the hyper-parameters. In this exercise,
 we will also address the identifiability problem and seek identifiable solutions which can
 only be achieved up to a permutation invariance. Lastly, we will also connect the LDM with the
  Non-Negative Matrix Factorization (NMF) theory.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib import image as mpimg
from spectral_clustering import Spectral_clustering_init
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm
# %% torch, plot, and CUDA configurations
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
plt.style.use('ggplot')
plt.rcParams['axes.facecolor'] = 'whitesmoke'
plt.set_cmap("tab10")

CUDA = torch.cuda.is_available()
# CUDA=False
if CUDA:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device("cpu")
    torch.set_default_tensor_type('torch.FloatTensor')
#%% Hybrid Membership-Latent Distance Model
class HMLDM(nn.Module, Spectral_clustering_init):
    '''
    Hybrid Membership-Latent Distance Model class
    '''

    def __init__(self, input_size, latent_dim, delta=1):
        super(HMLDM, self).__init__()

        # initialization of the class that is responsible for the spectral initialization of the latent variables Z
        # available initialization choices are: 'Adjacency', 'Normalized_sym', 'Normalized', 'MDS'
        Spectral_clustering_init.__init__(self, num_of_eig=latent_dim, method='Adjacency', device=device)

        # dimensions

        self.input_size = input_size
        self.latent_dim = latent_dim

        # initially we want to learn the scales of the random effects separately
        self.scaling_RE = True

        # hyperparameter controlling the simplex volume
        self.delta = delta

    def read_data(self, path):
        """
        reads input data:
        Netwok Edgelist (upper triangular part of the adjacency matrix for the unipartite undirected network case):
        - edge_pos_i: input data, link row positions i with i<j (edge row position)
        - edge_pos_j: input data, link column positions j with i<j (edge column position)
        """

        # input data, link (edge) rows i positions with i<j
        self.edge_pos_i = torch.from_numpy(np.loadtxt(edge_path + 'edge_pos_i.txt')).long().to(device)
        np.savetxt(edge_path + 'edge_pos_i.txt', np.loadtxt(edge_path + 'edge_pos_i.txt'), fmt='%s')

        # input data, link (edge) column positions with i<j
        self.edge_pos_j = torch.from_numpy(np.loadtxt(edge_path + 'edge_pos_j.txt')).long().to(device)
        np.savetxt(edge_path + 'edge_pos_j.txt', np.loadtxt(edge_path + 'edge_pos_j.txt'), fmt='%s')

    def init_parameters(self):
        """ define and initialize model parameters """

        self.Softmax = nn.Softmax(1)

        # Parameters

        # Random effects
        self.gamma = nn.Parameter(torch.randn(self.input_size, device=device))

        # Latent Variables

        # initialize Z based on the leading eigenvectors of the adjacency matrix
        self.spectral_data = self.spectral_clustering()
        self.latent_w1 = nn.Parameter(self.spectral_data)

        # self.latent_w2 = nn.Parameter(self.spectral_data)

    def LDM_Poisson_NLL(self, epoch):
        """Poisson log-likelihood ignoring the log(k!) constant"""
        self.epoch = epoch

        if self.scaling_RE:
            # We will spend 500 iteration on learning the random effects, defining a rate as exp(gamma_i+gamma_j)

            mat = torch.exp(self.gamma.unsqueeze(1) + self.gamma)  # NxN matrix containing all pairs i,j

            # multiply with 0.5 to account for the fact that we caclulated the whole NxN rate matrix
            # subtract the diagonal of the rate matrix since self-links are not allowed
            non_link_nll = 0.5 * (mat - torch.diag(torch.diagonal(mat))).sum()

            # calculate now the link term of the log-likelihood
            link_nll = (self.gamma[self.edge_pos_i] + self.gamma[self.edge_pos_j]).sum()

            # calculate the total nll of the LDM
            loss = -link_nll + non_link_nll

            if self.epoch == 500:
                # after 500 iteration stop the scaling and switch to full model training
                self.scaling_RE = False
        else:

            # here we provide you with the constrained embeddings on the standard simplex
            self.latent_w = self.Softmax(self.latent_w1)

            # PLEASE ADD HERE THE LOSS FUNCTION


            mat = torch.exp(-self.delta * ((torch.cdist(self.latent_w, self.latent_w, p=2)) + 1e-06))
            z_pdist1 = 0.5 * torch.mm(torch.exp(self.gamma.unsqueeze(0)), (
                torch.mm((mat - torch.diag(torch.diagonal(mat))), torch.exp(self.gamma).unsqueeze(-1))))
            z_pdist2 = (-self.delta * (((
                ((self.latent_w[self.edge_pos_i] - self.latent_w[self.edge_pos_j] + 1e-06) ** 2).sum(-1))) ** 0.5) +
                        self.gamma[self.edge_pos_i] + self.gamma[self.edge_pos_j]).sum()

            loss = -z_pdist2 + z_pdist1

        return loss

# %% Model training
def train_model(model, edge_path, epoch_num=2000):
    """
    Function for training the LDM

    :param model: The LDM object (class)
    :param edge_path: path containing the edges (str)
    :param epoch_num: number of training epochs (int)
    :return:
    """
    losses = []

    # Read the data
    model.read_data(path=edge_path)

    # Define and initialize the model parameters
    model.init_parameters()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    # training loop
    for epoch in tqdm(range(epoch_num), desc="LDM is Running…", ascii=False, ncols=75):
        # calculate the loss function
        loss = model.LDM_Poisson_NLL(epoch=epoch)
        losses.append(loss.item())

        optimizer.zero_grad()  # clear the gradients.
        loss.backward()  # backpropagate
        optimizer.step()  # update the weights

    # Plot the training loss
    plt.figure(figsize=(8, 8))
    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel('Poisson-NLL')
    plt.axvline(x=500, color='b', label='Full model training', ls='dotted')
    plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
    plt.show()

# %% main
if __name__ == '__main__':
    # datase name
    dataset='grqc'

    # latent dimensions of the embeddings W
    latent_dim=8

    # path to the folder containing the network edge list
    # this version contains only 50% of the network edges, the rest will be used for link prediction
    edge_path=f'./networks/{dataset}/'

    # path to the folder containing the sampled pairs for link prediction
    # (the 50% hidden edges and the same amount of negative samples)
    samples_path=f'./networks/{dataset}/samples_link_prediction/'


    # Size of the network
    N_size=int(np.loadtxt(edge_path+'network_size.txt'))

    # Define the LDM model
    model = HMLDM(input_size=N_size,latent_dim=latent_dim,delta=100).to(device)

    # Start the training process
    train_model(model=model,edge_path=edge_path)
