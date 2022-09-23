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
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
# from src.generate_graph.asq_graph import ASQ_Graph
# from config.config import config
from typing import Optional

import plotly.express as px
from matplotlib import image as mpimg
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


# %% Hybrid Membership-Latent Distance Model
class HMLDM(nn.Module):
    """ Hybrid Membership-Latent Distance Model class """

    def __init__(self, graph_path:str, latent_dim: int, delta: Optional[int] = 1):
        super(HMLDM, self).__init__()
        # initialize class to get the data
        # ASQ_Graph.__init__(self, config=config)

        # bipartite matrix
        self.get_graph(graph_path=graph_path)
        self.bip_matrix = self.get_adjacency_matrix()

        # latent variables
        self.input_size_n1 = self.bip_matrix.shape[0]  # rows
        self.input_size_n2 = self.bip_matrix.shape[1]  # columns
        self.latent_dim = latent_dim

        # initially we want to learn the scales of the random effects separately
        self.scaling_RE = True

        # hyperparameter controlling the simplex volume
        self.delta = delta

    def get_graph(self, graph_path:str) -> nx.Graph:
        """Get bipartite graph"""
        graph = nx.read_graphml(path=graph_path)
        nx.info(graph)
        self.bip = graph
        return graph

    def get_adjacency_matrix(self):
        """Return the adjency matrix of the bipartite graph"""
        subject_nodes = [node_ for node_ in self.bip.nodes() if node_.isnumeric()] # subjects are the rows
        M = bipartite.biadjacency_matrix(G=self.bip,row_order=subject_nodes, weight='weight')
        M = M.toarray()
        print(f'\nReturning Bipartite Matrix of dimension {M.shape}')
        return M

    def read_data(self):
        """
        reads input data:
        Netwok Edgelist (upper triangular part of the adjacency matrix for the unipartite undirected network case):
        - edge_pos_i: input data, link row positions i with i<j (edge row position)
        - edge_pos_j: input data, link column positions j with i<j (edge column position)
        """
        # get indexes of the matrix where value >0. For rows and columns
        self.bip_matrix = torch.from_numpy(self.bip_matrix)
        self.rows_idx, self.col_idx = torch.where(self.bip_matrix > 0)

    def init_parameters(self):
        """define and initialize model parameters"""
        # Parameters
        # Random effects, the random effects are single values for each observation
        self.gamma_n1 = nn.Parameter(torch.randn(self.input_size_n1, device=device))
        self.gamma_n2 = nn.Parameter(torch.randn(self.input_size_n2, device=device))
        # self.gamma_col = nn.Parameter(torch.randn(self.input_size_n2, device=device))
        print(f'\nGamma_n1 shape: {self.gamma_n1.shape}')
        print(f'\nGamma_n2 shape: {self.gamma_n2.shape}')

        self.latent_z = nn.Parameter(torch.rand(self.input_size_n1, 0))
        self.latent_w = nn.Parameter(torch.rand(self.input_size_n2, 0))

        print(f'\nlatent_z shape: {self.latent_z.shape}')
        print(f'\nlatent_w shape: {self.latent_w.shape}')

        self.Softmax = nn.Softmax(1)

    def LDM_Poisson_NLL(self, epoch):
        """
        Poisson log-likelihood ignoring the log(k!) constant
        P = λ^{k}exp(-λ)   -> los = k ln(λ) -λ
            with k the weights of the adjacent matrix
        """
        self.epoch = epoch

        if self.scaling_RE:
            # Initialization of the random effects (nll=negative loss likelihood)
            # A. calculate the non link term of the log-likelihood
            # QUESTION: WHY DO WE CONSIDER ONLY GAMMA_N1, WHAT ABOUT GAMMA_N2?
            mat = torch.exp(self.gamma_n1.unsqueeze(1) + self.gamma_n1)
            # print(f'mat shape {mat.shape}')  # Results in a N2 x N2 matrix, and N1?
            non_link_nll = mat.sum()

            # calculate now the link term of the log-likelihood
            # 1. compute the euclidean distance
            delta_latent = ((self.latent_z[self.rows_idx] - self.latent_w[self.col_idx]) ** 2).sum() ** (1 / 2)

            # 2. graph_weights * (gamma_i + delta_j - d_{z_{i},w_{j}})
            link_nll = self.bip_matrix[self.rows_idx, self.col_idx] * (
                        self.gamma_n1[self.rows_idx] + self.gamma_n2[self.col_idx] - delta_latent)
            link_nll = link_nll.sum()

            # B. calculate the total nll of the LDM
            # -L = -sum[Weights_ij*ln(λ_ij)] + sum[λ_ij]
            loss = -link_nll + non_link_nll

            if self.epoch == 500:
                # after 500 iteration stop the scaling and switch to full model training
                self.scaling_RE = False
        else:
            """log(λ_{ij}) = (\gamma_i + \gamma_j -\delta^p d_{z_{i},w_{j}} )"""
            # train the model, also the random effects are changed during this training
            # QUESTION: Should we use a constrained embeddings on the standard simplex ?

            dist_matrix = -self.delta * ((torch.cdist(self.latent_z, self.latent_w, p=2)) + 1e-06)
            # print(f'dist_matrix shape  {dist_matrix.shape}')

            # the matrix multiplication sum[exp(gamma_i + delta_j - dist_matrix)]
            #  == exp(gamma_i) * exp(delta_j) * dist_matrix.T)
            z_pdist1 = torch.mm(
                torch.exp(self.gamma_n1.unsqueeze(0)),
                (torch.mm(torch.exp(dist_matrix), torch.exp(self.gamma_n2).unsqueeze(-1)))
            )

            # print(f'\nz_pdist1 shape {z_pdist1.shape}')

            # 2. graph_weights * (gamma_i + delta_j - d_{z_{i},w_{j}})
            delta_latent = ((self.latent_z[self.rows_idx] - self.latent_w[self.col_idx]) ** 2).sum(-1) ** (1 / 2)

            graph_weights = self.bip_matrix[self.rows_idx, self.col_idx]
            z_pdist2 = (graph_weights * (
                    self.gamma_n1[self.rows_idx] + self.gamma_n2[self.col_idx] -self.delta * delta_latent)
                        ).sum()

            loss = -z_pdist2 + z_pdist1

        return loss


# %% Model training
def train_model(model, epoch_num=2000):
    """
    Function for training the LDM

    :param model: The LDM object (class)
    :param edge_path: path containing the edges (str)
    :param epoch_num: number of training epochs (int)
    :return:
    """
    losses = []

    # Read the data
    model.read_data()

    # Define and initialize the model parameters
    model.init_parameters()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    # training loop
    for epoch in tqdm(range(epoch_num), desc="HLDM is Running…", ascii=False, ncols=75):
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
    # latent dimensions of the embeddings W
    graph_path = r'...'
    latent_dim = 20

    # Define the LDM model
    model = HMLDM(latent_dim=latent_dim, delta=100).to(device)

    # Start the training process
    train_model(model=model)
