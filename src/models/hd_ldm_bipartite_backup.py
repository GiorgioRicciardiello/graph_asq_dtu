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
import pathlib

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from notebooks.class_notebook.latent_distance_model_exercises.spectral_clustering import Spectral_clustering_init
from src.generate_graph.asq_graph import ASQ_Graph
from config.config import config
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
class HMLDM(nn.Module, Spectral_clustering_init, ASQ_Graph):
    """ Hybrid Membership-Latent Distance Model class """

    def __init__(self, config: dict, latent_dim: int, delta: Optional[int] = 1):
        super(HMLDM, self).__init__()
        # initialization of the class that is responsible for the spectral initialization of the latent variables Z
        # available initialization choices are: 'Adjacency', 'Normalized_sym', 'Normalized', 'MDS'
        # Spectral_clustering_init.__init__(self, num_of_eig=latent_dim, method='MDS', device=device)
        # initialize class to get the data
        ASQ_Graph.__init__(self, config=config)

        # bipartite matrix
        self.get_graph()
        self.bip_matrix = self.get_adjacency_matrix()

        # dimensions (rows, columns)
        # self.graph_size = self.bip.size()  # number of edges
        # print(f'\n Bipartite Graph size: {self.bip.size()}')
        # print(f'\n Bipartite Graph from ¦E¦ = pq: {self.bip_matrix.shape[0] * self.bip_matrix.shape[1]}')

        self.input_size_n1 = self.bip_matrix.shape[0]  # rows
        self.input_size_n2 = self.bip_matrix.shape[1]  # columns
        self.latent_dim = latent_dim

        # initially we want to learn the scales of the random effects separately
        self.scaling_RE = True

        # hyperparameter controlling the simplex volume
        self.delta = delta

    def read_data(self, train_test_split: Optional[bool] = False):
        """
        reads input data:
        https://www.youtube.com/watch?v=Lhef_jxzqCg -> uses a sparse matrix with ijv notation
        """
        # get indexes of the matrix where value >0. For rows and columns
        self.bip_matrix = torch.from_numpy(self.bip_matrix).float().to(device)
        self.rows_idx, self.col_idx = torch.where(self.bip_matrix > 0)

    # def save_ijv_format(self):
    #     np.savetxt(self.config["ijv_format_folder"].joinpath(pathlib.Path(r'edges_i')),
    #                           self.rows_id.numpy(), delimiter= ',')
    #
    #     np.savetxt(self.config["ijv_format_folder"].joinpath(pathlib.Path(r'edges_j')),
    #                           self.col_idx.numpy(), delimiter= ',')
    #
    #     pass


    def init_parameters(self):
        """define and initialize model parameters"""
        # Parameters
        # Random effects, the random effects are single values for each observation
        self.gamma_n1 = nn.Parameter(torch.randn(self.input_size_n1, device=device))
        self.gamma_n2 = nn.Parameter(torch.randn(self.input_size_n2, device=device))
        # self.gamma_col = nn.Parameter(torch.randn(self.input_size_n2, device=device))
        print(f'\nGamma_n1 shape: {self.gamma_n1.shape}')
        print(f'\nGamma_n2 shape: {self.gamma_n2.shape}')

        # Latent Variables
        # initialize Z based on the leading eigenvectors of the adjacency matrix
        # self.spectral_data = self.spectral_clustering() -> comment for the ASQ
        # print(f'\nSpectral cluster data shape: {self.spectral_data.shape}')

        # self.spectral_data = self.spectral_clustering()

        # define the latent space we want to generate
        self.latent_z = nn.Parameter(torch.rand((self.input_size_n1, self.latent_dim), device=device))
        self.latent_w = nn.Parameter(torch.rand((self.input_size_n2, self.latent_dim), device=device))

        print(f'\nlatent_z shape: {self.latent_z.shape}')
        print(f'\nlatent_w shape: {self.latent_w.shape}')

        self.Softmax = nn.Softmax(dim=1)

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
            # consider both random effects
            mat = torch.exp(self.gamma_n1.unsqueeze(1) + self.gamma_n2)
            # print(f'mat shape {mat.shape}')  # Results in a N1 x N2 matrix, and N1?
            non_link_nll = mat.sum()

            # calculate now the link term of the log-likelihood
            # 1. compute the euclidean distance - WE DO NOT USE IT
            # delta_latent = ((self.latent_z[self.rows_idx] - self.latent_w[self.col_idx]) ** 2).sum() ** (1 / 2)

            # 2. graph_weights * (gamma_i + delta_j - d_{z_{i},w_{j}})
            link_nll = self.bip_matrix[self.rows_idx, self.col_idx] * (
                    self.gamma_n1[self.rows_idx] + self.gamma_n2[self.col_idx])  # - delta_latent)

            link_nll = link_nll.sum()

            # B. calculate the total nll of the LDM
            # -L = -sum[Weights_ij*ln(λ_ij)] + sum[λ_ij]
            loss = -link_nll + non_link_nll

            if self.epoch == self.config['train_random_effects_epochs_num']:
                # after 500 iteration stop the scaling and switch to full model training
                self.scaling_RE = False
        else:
            """log(λ_{ij}) = (\gamma_i + \gamma_j -\delta^p d_{z_{i},w_{j}} )"""
            self.latent_z_soft = self.Softmax(self.latent_z)
            self.latent_w_soft = self.Softmax(self.latent_w)

            # train the model, also the random effects are changed during this training
            # QUESTION: Should we use a constrained embeddings on the standard simplex ?

            dist_matrix = -self.delta * ((torch.cdist(self.latent_z_soft, self.latent_w_soft,
                                                      p=2)) + 1e-06)
            # print(f'dist_matrix shape  {dist_matrix.shape}')

            # the matrix multiplication sum[exp(gamma_i + delta_j - dist_matrix)]
            #  == exp(gamma_i) * exp(delta_j) * dist_matrix.T)
            z_pdist1 = torch.mm(
                torch.exp(self.gamma_n1.unsqueeze(0)),
                (torch.mm(torch.exp(dist_matrix), torch.exp(self.gamma_n2).unsqueeze(-1)))
            )

            # print(f'\nz_pdist1 shape {z_pdist1.shape}')

            # z_pdist1 = torch.mm(
            #     torch.exp(self.gamma.unsqueeze(0)),
            #     (torch.mm(
            #             (dist_matrix - torch.diag(torch.diagonal(dist_matrix))),
            #             torch.exp(self.gamma).unsqueeze(-1))))

            # 2. graph_weights * (gamma_i + delta_j - d_{z_{i},w_{j}})
            delta_latent = ((self.latent_z_soft[self.rows_idx] - self.latent_w_soft[self.col_idx]) ** 2).sum(-1) ** (
                        1 / 2)

            graph_weights = self.bip_matrix[self.rows_idx, self.col_idx]
            z_pdist2 = (graph_weights * (
                    self.gamma_n1[self.rows_idx] + self.gamma_n2[self.col_idx] - self.delta * delta_latent)
                        ).sum()

            loss = -z_pdist2 + z_pdist1

        return loss


# %% Model training
def train_model(model, epoch_num=10):
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
    # plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel('Poisson-NLL')
    plt.axvline(x=model.config["train_random_effects_epochs_num"],
                color='b', label='Full model training', ls='dotted')

    plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
    plt.show()
    return losses


# %% main
if __name__ == '__main__':
    # latent dimensions of the embeddings W
    latent_dim = 8

    # Define the LDM model
    model = HMLDM(config=config, latent_dim=latent_dim, delta=4).to(device)

    # Start the training process
    losses = train_model(model=model)


    def visualize_embedding(model):
        # embedding space for the subjects
        # the target is the first element of the matrix
        labels = model.bip_matrix[::, 0].cpu().numpy()
        # labels = np.loadtxt(edge_path + "labels.txt")[:, 1]
        plt.figure(figsize=(10, 10))
        plt.title('LDM 2-Dimensional Z-Embedding Space ')
        plt.xlabel('Z1')
        plt.ylabel('Z2')
        plt.grid(False)
        plt.axis('off')
        # plt.legend()
        plt.scatter(model.latent_z[:, 0].detach().cpu().numpy(),
                    model.latent_z[:, 1].detach().cpu().numpy(),
                    c=labels,
                    s=20)
        plt.show()

        # how do I use the labels for the rows (questions)
        plt.figure(figsize=(10, 10))
        plt.title('LDM 2-Dimensional Z-Embedding Space ')
        plt.xlabel('Z1')
        plt.ylabel('Z2')
        plt.grid(False)
        plt.axis('off')
        plt.scatter(model.latent_w[:, 0].detach().cpu().numpy(),
                    model.latent_w[:, 1].detach().cpu().numpy(),
                    # c=labels,
                    s=20)
        plt.show()


    def re_ordering_adjency_matrix(model):
        from scipy.sparse import coo_matrix, csr_matrix
        plt.rcParams["figure.figsize"] = (10, 10)

        w_idx = model.latent_w.argmax(1)
        z_idx = model.latent_z.argmax(1)

        f_w = w_idx.argsort()
        f_z = z_idx.argsort()

        new_i = torch.catS((model.rows_idx, model.rows_idx))
        new_j = torch.cat((model.col_idx, model.col_idx))

        D = csr_matrix((np.ones(new_i.shape[0]), (new_i.cpu().numpy(), new_j.cpu().numpy())),
                       shape=(model.input_size_n1, model.input_size_n2))

        D = csr_matrix((np.ones(model.bip_matrix.shape[0]), (new_i.cpu().numpy(), new_j.cpu().numpy())),
                       shape=(model.input_size_n1, model.input_size_n2))

        plt.title('Initial adjacency matrix')
        plt.spy(D, markersize=1, alpha=1)
        plt.axis('off')
        plt.show()

        # order the adj matrix based on the community allocations
        D = D[:, f_z.cpu().numpy()][f_z.cpu().numpy()]

        plt.title('HM-LDM re-ordered adjacency matrix')
        plt.spy(D, markersize=1, alpha=0.5)
        plt.axis('off')
        plt.show()


    def link_prediction_hm(model):
        # file denoting rows i of hidden links and negative samples, with i<j
        # total_samples_i = torch.from_numpy(np.loadtxt(samples_path + 'total_samples_i.txt')).long().to(device)

        # file denoting columns j of hidden links and negative samples, with i<j
        # total_samples_j = torch.from_numpy(np.loadtxt(samples_path + 'total_samples_j.txt')).long().to(device)

        # select the last 200
        total_samples_i = model.rows_idx[model.rows_idx.shape[0] - 200::].long().to(device)
        total_samples_j = model.col_idx[model.col_idx.shape[0] - 200::].long().to(device)

        # target vector having 0 if the missing pair is a negative sample and 1 if the pair considers a hidden (removed) edge
        # target = torch.from_numpy(np.loadtxt(samples_path + 'target.txt')).long().to(device)
        target = model.bip_matrix[model.bip_matrix.shape[0] - 200::, 0]

        with torch.no_grad():
            z_pdist_miss = (((model.latent_w[total_samples_i] - model.latent_w[total_samples_j]) ** 2).sum(-1)) ** 0.5
            logit_u_miss = -model.delta * z_pdist_miss + model.gamma[total_samples_i] + model.gamma[total_samples_j]
            rates = logit_u_miss

            # calculate AUC-PR
            precision, recall, thresholds = metrics.precision_recall_curve(target.cpu().data.numpy(),
                                                                           rates.cpu().data.numpy())
            auc_pr = metrics.auc(recall, precision)

            # calculate AUC-ROC

            auc_roc = metrics.roc_auc_score(target.cpu().data.numpy(), rates.cpu().data.numpy())
            fpr, tpr, thresholds = metrics.roc_curve(target.cpu().data.numpy(), rates.cpu().data.numpy())

        return auc_roc, fpr, tpr, auc_pr, precision, recall
