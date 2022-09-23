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
from sklearn.model_selection import train_test_split
from collections import Counter
from scipy.sparse import coo_matrix, csr_matrix
import plotly.express as px
from matplotlib import image as mpimg
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
from tqdm import tqdm
import networkx as nx
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score

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

    def __init__(self, config: dict, latent_dim: int, delta: Optional[int] = 1, p:Optional[int] = 1):
        super(HMLDM, self).__init__()
        # initialization of the class that is responsible for the spectral initialization of the latent variables Z
        # available initialization choices are: 'Adjacency', 'Normalized_sym', 'Normalized', 'MDS'
        # Spectral_clustering_init.__init__(self, num_of_eig=latent_dim, method='MDS', device=device)
        # initialize class to get the data
        ASQ_Graph.__init__(self, config=config)

        # bipartite matrix
        self.get_graph()
        self.bip_matrix = self.get_adjacency_matrix()
        # self.input_size_rows = self.bip_matrix.shape[0]  # rows
        # self.input_size_cols = self.bip_matrix.shape[1]  # columns
        self.latent_dim = latent_dim

        # initially we want to learn the scales of the random effects separately
        self.scaling_RE = True

        # hyperparameter controlling the simplex volume
        self.delta = delta
        self.p = 1

    def read_data(self, train_test_split: Optional[bool] = False):
        """
        Order operation = 2
        reads input data:
        https://www.youtube.com/watch?v=Lhef_jxzqCg -> uses a sparse matrix with ijv notation
        """
        # if train_test_split:
        #     target = self.bip_matrix[::,0].cpu()
        #     self.bip_matrix_train, self.bip_matrix_test = train_test_split(self.bip_matrix, test_size = 0.33,
        #                                                                    random_state = self.config['seed'],
        #                                                                    stratify=target)
        #     # get indexes of the matrix where value >0. For rows and columns
        #     self.bip_matrix = torch.from_numpy(self.bip_matrix_train).float().to(device)
        #     self.rows_idx, self.col_idx = torch.where(self.bip_matrix_train > 0)
        #
        #     self.bip_matrix = torch.from_numpy(self.bip_matrix_test).float().to(device)
        #     self.rows_idx_test, self.col_idx_test = torch.where(self.bip_matrix_test > 0)
        # else:

        # get indexes of the matrix where value >0. For rows and columns
        self.bip_matrix = torch.from_numpy(self.bip_matrix).float().to(device)
        self.rows_idx, self.col_idx = torch.where(self.bip_matrix > 0)

        self.input_size_rows = self.bip_matrix.shape[0]  # rows
        self.input_size_cols = self.bip_matrix.shape[1]  # columns
    # def save_ijv_format(self):
    #     np.savetxt(self.config["ijv_format_folder"].joinpath(pathlib.Path(r'edges_i')),
    #                           self.rows_id.numpy(), delimiter= ',')
    #
    #     np.savetxt(self.config["ijv_format_folder"].joinpath(pathlib.Path(r'edges_j')),
    #                           self.col_idx.numpy(), delimiter= ',')
    #
    #     pass

    def train_test_split(self, plot_dist: Optional[bool] = False):
        """
        Order operation = NoNE   THIS METHODS IS NOT APPROPIATE
        Split into train and test, while preserving similar distribution on both classes.
        In graphs, we remove the edges we want to predict. These edges are the first column of the test_graph
        So we set them to zero when we add them to the bip_matrix BUT this changes the coordinate matrix
        :return:
        """
        # split train and test with stratified on the column with the traget values (weighted edges)
        train_graph, test_graph = train_test_split(self.bip_matrix, test_size=0.5,
                                                   random_state=self.config['seed'],
                                                   stratify=self.bip_matrix[::, 0])
        # If we concatenate and do not mask the test targets when we read the data the coordinate index will be same
        # print(f'\nTrain-Test split has been performed and the splits have been concatenated in self.bip_matrix')
        # self.bip_matrix = np.concatenate((train_graph, test_graph))
        print(f'\nBip_matrix contains only the training set, we consider only part of the graph')
        self.bip_matrix = train_graph
        self.bip_matrix_test = test_graph

        self.test_num_samples = test_graph.shape[0]

        if plot_dist:
            # to plot, we make a dictionary with the count of unique elements
            counter_train_targets = {int(key): val for key, val in zip(Counter(train_graph[::, 0].tolist()).keys(),
                                                                       Counter(train_graph[::, 0].tolist()).values())}

            counter_test_targets = {int(key): val for key, val in zip(Counter(test_graph[::, 0].tolist()).keys(),
                                                                      Counter(test_graph[::, 0].tolist()).values())}
            # plot for training
            plt.bar(*zip(*counter_train_targets.items()))
            plt.show()
            # plot for testing
            plt.bar(*zip(*counter_test_targets.items()))
            plt.show()

    def targets_train_test_ijv_coordinates(self):
        """
        Order operation = 3
        We want to remove 50% of the target edges corresponding to the stratified shuffle.
        We want to leave the rest of the edges as those are features of the nodes

        The test_target is a weighted array using the ijv coordinate, specially the column coordinate
        :return:
        """
        # get the ijv coordinates for the targets
        target_col_idx = torch.where(self.col_idx == 0)[0]
        # we take the 20% of the columns indexes from the complete ijv notation
        target_test_col_idx = target_col_idx[..., -self.test_num_samples:]
        target_test_row_idx = self.rows_idx[target_test_col_idx]

        test_target = torch.full(size=self.col_idx.size(), fill_value=0, dtype=torch.float)  # in ijv coordinates
        test_target[target_test_col_idx] = self.bip_matrix[self.bip_matrix.shape[0] - self.test_num_samples::, 0]

        print(f'\nTarget values removed (set = 0) from the test split of the bip_matrix')
        # remove the edges for the target in the test partition
        self.bip_matrix[self.bip_matrix.shape[0] - self.test_num_samples::, 0] = 0

    def init_parameters(self):
        """define and initialize model parameters"""
        # Parameters
        # Random effects, the random effects are single values for each observation
        self.gamma_rows = nn.Parameter(torch.randn(self.input_size_rows, device=device))
        self.gamma_cols = nn.Parameter(torch.randn(self.input_size_cols, device=device))
        # self.gamma_col = nn.Parameter(torch.randn(self.input_size_cols, device=device))
        print(f'\ngamma_rows shape: {self.gamma_rows.shape}')
        print(f'\ngamma_cols shape: {self.gamma_cols.shape}')

        # define the latent space we want to generate
        self.latent_z = nn.Parameter(torch.rand((self.input_size_rows, self.latent_dim), device=device))
        self.latent_w = nn.Parameter(torch.rand((self.input_size_cols, self.latent_dim), device=device))

        print(f'\nlatent_z shape: {self.latent_z.shape}')
        print(f'\nlatent_w shape: {self.latent_w.shape}')

        self.Softmax = nn.Softmax(dim=1)

    def LDM_Poisson_NLL(self, epoch):
        """
        Order operation = 5
        Poisson log-likelihood ignoring the log(k!) constant
        P = λ^{k}exp(-λ)   -> los = k ln(λ) -λ
            with k the weights of the adjacent matrix
        """
        self.epoch = epoch

        if self.scaling_RE:
            # Initialization of the random effects (nll=negative loss likelihood)
            # A. calculate the non link term of the log-likelihood
            # consider both random effects
            mat = torch.exp(self.gamma_rows.unsqueeze(1) + self.gamma_cols)
            # print(f'mat shape {mat.shape}')  # Results in a N1 x N2 matrix, and N1?
            non_link_nll = mat.sum()

            # calculate now the link term of the log-likelihood
            # 1. compute the euclidean distance - WE DO NOT USE IT
            # delta_latent = ((self.latent_z[self.rows_idx] - self.latent_w[self.col_idx]) ** 2).sum() ** (1 / 2)

            # 2. graph_weights * (gamma_i + delta_j - d_{z_{i},w_{j}})
            link_nll = self.bip_matrix[self.rows_idx, self.col_idx] * (
                    self.gamma_rows[self.rows_idx] + self.gamma_cols[self.col_idx])  # - delta_latent)

            link_nll = link_nll.sum()

            # B. calculate the total nll of the LDM
            # -L = -sum[Weights_ij*ln(λ_ij)] + sum[λ_ij]
            loss = -link_nll + non_link_nll

            if self.epoch == self.config['train_random_effects_epochs_num']:
                # after 500 iteration stop the scaling and switch to full model training
                self.scaling_RE = False
        else:
            """
            log(λ_{ij}) = (\gamma_i + \gamma_j -\delta^p d_{z_{i},w_{j}} )
            To check if the simplex dimension is good, we should have at least a 1 valu on each embedding row
            out, inds = torch.max(model.latent_z_soft,dim=1) max of each row
            """
            self.latent_z_soft = self.Softmax(self.latent_z)
            self.latent_w_soft = self.Softmax(self.latent_w)

            # train the model, also the random effects are changed during this training
            dist_matrix = -self.delta * ((torch.cdist(self.latent_z_soft, self.latent_w_soft,
                                                      p=2)) + 1e-06)
            # print(f'dist_matrix shape  {dist_matrix.shape}')

            # matrix multiplication sum[exp(gamma_i + delta_j - dist_matrix)] == exp(gamma_i) * exp(delta_j) * dist_matrix.T)
            z_pdist1 = torch.mm(
                torch.exp(self.gamma_rows.unsqueeze(0)),
                (torch.mm(torch.exp(dist_matrix), torch.exp(self.gamma_cols).unsqueeze(-1)))
            )

            # 2. graph_weights * (gamma_i + delta_j - d_{z_{i},w_{j}})
            delta_latent = ((self.latent_z_soft[self.rows_idx] - self.latent_w_soft[self.col_idx]) ** 2).sum(-1) ** (
                        1 / 2)

            graph_weights = self.bip_matrix[self.rows_idx, self.col_idx]
            z_pdist2 = \
                (graph_weights * (
                    self.gamma_rows[self.rows_idx] + self.gamma_cols[self.col_idx]-(self.delta * delta_latent)**self.p)
                        ).sum()
            # loss = -non_link_edges + all_edges
            loss = -z_pdist2 + z_pdist1

            # #  if we use the perumuation with the NNF 𝛾̃_𝑖 + 𝛾̃ _𝑗 + 2𝛿^{2} ⋅ (𝐰_{𝑖} 𝐰_{𝑗}^{⊤}) . WRONG
            # mat_nnf = torch.exp(-self.delta * ((torch.cdist(self.latent_w_soft, self.latent_w_soft, p=2)) + 1e-06))
            # z_pdist1 = torch.mm(
            #     torch.exp(self.gamma_rows.unsqueeze(0)),
            #     (torch.mm(torch.exp(mat_nnf), torch.exp(self.gamma_cols).unsqueeze(-1)))
            # )
            #
            # z_pdist2 = (-self.delta * (((
            #     ((self.latent_w[self.edge_pos_i] - self.latent_w[self.edge_pos_j] + 1e-06) ** 2).sum(-1))) ** 0.5) +
            #             self.gamma[self.edge_pos_i] + self.gamma[self.edge_pos_j]).sum()
            #
            # loss = -z_pdist2 + z_pdist1

        return loss


    def link_prediction_try_jupyter(self):
        """
        Following hand-by-and the jyputer. Problem is, when the partition is done the indexes of the embededding
        will not match the location of the test_matrix, as they will surpass the location of the training
        :return:
        """
        self.bip_matrix_test = torch.from_numpy(self.bip_matrix_test).to(device)
        bip_matrix = torch.cat((self.bip_matrix, self.bip_matrix_test), dim=0)
        total_samples_rows, total_samples_cols = torch.where(bip_matrix > 0)

        # get the ijv coordinates for the targets
        target_col_idx = torch.where(total_samples_cols == 0)[0]
        # we take the 50% of the columns indexes from the complete ijv notation
        target_test_col_idx = target_col_idx[..., -self.test_num_samples:]
        target_test_row_idx = total_samples_rows[target_test_col_idx]

        test_target = torch.full(size=total_samples_cols.size(), fill_value=0, dtype=torch.float)  # in ijv coordinates
        test_target[target_test_col_idx] = bip_matrix[bip_matrix.shape[0] - self.test_num_samples::, 0]

        latent_w = self.latent_w.detach().cpu()
        latent_z = self.latent_z.detach().cpu()
        gamma_rows = self.gamma_rows.detach().cpu()
        gamma_cols = self.gamma_rows.detach().cpu()
        # distance difference between the two embedding space DIMENSIONS DO NOT MATCH, HAVE TO SWAP
        delta_latent = ((latent_z[total_samples_rows] - latent_w[total_samples_cols]) ** 2).sum(-1) ** (1 / 2)
        logit_u_miss = gamma_rows[total_samples_rows]+gamma_cols[total_samples_cols]-(self.delta * delta_latent)**self.p
        rates = logit_u_miss

        # calculate AUC-PR
        precision, recall, thresholds = metrics.precision_recall_curve(target.cpu().data.numpy(),
                                                                       rates.cpu().data.numpy())
        auc_pr = metrics.auc(recall, precision)

        auc_roc = metrics.roc_auc_score(target.cpu().data.numpy(), rates.cpu().data.numpy())
        fpr, tpr, thresholds = metrics.roc_curve(target.cpu().data.numpy(), rates.cpu().data.numpy())

    def link_prediction_try_all_graph(self):
        """
        Perform link prediction on the same training set. This tells us the best the model can do.
        Objective:
             Remove edges to predict using the ijv notation
        How:
            All the targets are in the zero (first) column of the data matrix
            We need to use the ijv coordinate formulation
            All the self.col_idx == 0 indicate a value on the first columns
            verified by summing all the zero indexes, it matches the number of targets in the dataset

            Then we sample a number of edges from the original matrix to extract
        :return:
        """
        # set the targets:
        # all zero column indexes point to a target value.
        assert self.col_idx.shape[0] - torch.count_nonzero(self.col_idx, dim=0) == self.bip_matrix.shape[0]
        # get the ijv coordinates for the targets
        target_col_idx = torch.where(self.col_idx == 0)[0]
        target_row_idx = self.rows_idx[target_col_idx]
        target_data = self.bip_matrix[::, 0]
        assert self.col_idx[target_col_idx].unique().item() == 0  # column zero all values
        # assert target_row_idx == target_col_idx
        assert target_col_idx.shape == target_data.shape
        # now on target_col_idx we set which edges will be predicted at which one not
        # lets say the last 200 will be used for prediction
        weights = torch.full(size=target_col_idx.size(), fill_value=1 / target_col_idx.shape[0])
        idx = torch.multinomial(weights, 200, replacement=False)
        assert idx.unique().shape[0] == 200
        # Column indexes to predict our targets
        test_target_col_idx = target_col_idx[idx]
        target = torch.full(size=self.col_idx.size(), fill_value=0)
        target[test_target_col_idx] = 1

        self.rows_idx = self.rows_idx.detach().cpu()
        self.col_idx = self.col_idx.detach().cpu()
        latent_w = self.latent_w.detach().cpu()
        latent_z = self.latent_z.detach().cpu()
        gamma_rows = self.gamma_rows.detach().cpu()
        gamma_cols = self.gamma_rows.detach().cpu()

        # distance difference between the two embedding space
        delta_latent = ((latent_z[self.rows_idx] - latent_w[self.col_idx]) ** 2).sum(-1) ** (1 / 2)
        # compute equation 6
        logit_u_miss = gamma_rows[self.rows_idx] + gamma_cols[self.col_idx] - self.delta * delta_latent
        rates = logit_u_miss

        # calculate AUC-PR
        precision, recall, thresholds = metrics.precision_recall_curve(target.cpu().data.numpy(),
                                                                       rates.cpu().data.numpy())
        auc_pr = metrics.auc(recall, precision)

        auc_roc = metrics.roc_auc_score(target.cpu().data.numpy(), rates.cpu().data.numpy())
        fpr, tpr, thresholds = metrics.roc_curve(target.cpu().data.numpy(), rates.cpu().data.numpy())


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
    # plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel('Poisson-NLL')
    plt.axvline(x=model.config["train_random_effects_epochs_num"],
                color='b', label='Full model training', ls='dotted')

    plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
    plt.show()
    return losses

# %% link prediction
def link_prediction_hm(model):
    # remove random edges to the target (AHI, first column of the matrix)
    # target = self.bip_matrix[::, 0]
    # self.rows_idx
    # self.col_idx

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

# %% Community detection
def community_detection(model):
    # HOW DO WE USE THE TWO EMBEDDINGS?
    # Calculate maximum community membership
    membership_w = model.latent_w.argmax(1).cpu().numpy()
    membership_z = model.latent_z.argmax(1).cpu().numpy()
    labels = model.bip_matrix[::, 0].cpu().numpy()
    # NMI calculation
    nmi_w = normalized_mutual_info_score(labels, membership_w)
    nmi_z = normalized_mutual_info_score(labels, membership_z)

    # ARI calculation
    ari_w = adjusted_rand_score(labels, membership_w)
    ari_z = adjusted_rand_score(labels, membership_z)

    print(f'Normalized mutual information score for delta={model.delta} is NMI_Z={nmi_z:.4f}')
    print(f'Adjusted rand index score for delta={model.delta} is ARI_Z={ari_z:.4f}')
    pass

# %% Visualuze embedding
def visualize_embedding(model):
    """
    The visualization is done on the softmax of the embeddings
    :param model:
    :return:
    """
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
    plt.scatter(model.latent_z_soft[:, 0].detach().cpu().numpy(),
                model.latent_z_soft[:, 1].detach().cpu().numpy(),
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
    plt.scatter(model.latent_w_soft[:, 0].detach().cpu().numpy(),
                model.latent_w_soft[:, 1].detach().cpu().numpy(),
                # c=labels,
                s=20)
    plt.show()

    # subplots plotting each dimension against each other like in a correlation matrix
    labels = model.bip_matrix[::, 0].cpu().numpy()
    plt.figure(figsize=(200, 200), dpi=300)
    # plt.subplots_adjust(hspace=0.1, wspace=0.1)
    fig, ax = plt.subplots(8, 8,)
    fig.subplots_adjust(wspace=0.2, hspace=0.2)
    for row_ in range(0, model.latent_z_soft.shape[1]):
        # ax = plt.subplot(8, 8, n + 1)
        for col_ in range(0, model.latent_z_soft.shape[1]):
            if row_> col_:
                ax[row_, col_].scatter(model.latent_z_soft[:, row_].detach().cpu().numpy(),
                            model.latent_z_soft[:, col_].detach().cpu().numpy(),
                                       c=labels,
                                       s=.2)
                ax[row_, col_].axis('off')
                ax[row_, col_].grid(False)
                plt.xlabel('Z1')
                plt.ylabel('Z2')
            elif row_ == col_:
                fig.delaxes(ax[row_][col_])
            else:
                # fig.delaxes(ax[row_][col_])
                ax[row_, col_].scatter(model.latent_w_soft[:, row_].detach().cpu().numpy(),
                                       model.latent_w_soft[:, col_].detach().cpu().numpy(),
                                       # c=labels,
                                       s=.2)
                ax[row_, col_].axis('off')
                ax[row_, col_].grid(False)
    plt.tight_layout()
    # plt.show()
    plt.draw()
    plt.savefig(r'C:\Users\giorg\OneDrive_ItaloCol\DTU\MSc_courses_material\Graph Representation Learning\graph_asq\reports\figures\embedding_space\soft_latent_w_z.png',
                dpi=300)

    plt.figure(figsize=(200, 200), dpi=300)
    fig, ax = plt.subplots(8, 8,)
    for row_ in range(0, model.latent_z_soft.shape[1]):
        # ax = plt.subplot(8, 8, n + 1)
        for col_ in range(0, model.latent_z_soft.shape[1]):
            if row_>= col_:
                ax[row_, col_].scatter(model.latent_w_soft[:, row_].detach().cpu().numpy(),
                            model.latent_w_soft[:, col_].detach().cpu().numpy(),
                                       # c=labels,
                                       s=.2)
                ax[row_, col_].axis('off')
                ax[row_, col_].grid(False)
            else:
                fig.delaxes(ax[row_][col_])
    plt.tight_layout()
    # plt.show()
    plt.draw()
    plt.savefig(r'C:\Users\giorg\OneDrive_ItaloCol\DTU\MSc_courses_material\Graph Representation Learning\graph_asq\reports\figures\embedding_space\soft_latent_w.png',
                dpi=300)

# %% Visualize Communities from the adjacency matrix
def re_ordering_adjacency_matrix(model):
    """
    Visualize the matrix, where there are connections between the two nodes. And then visualize it
    when the rows and columns are sorted based on the embeddings. Also use the target to color show each
    row value and see if there is a communist cluster for the AHI
    :param model:
    :return:
    """
    plt.rcParams["figure.figsize"] = (10, 10)
    from matplotlib import cm
    import matplotlib.patches as mpatches
    import seaborn as sns
    # # Adjacency matrix for a bipartite matrix
    # Affinity_matrix = model.bip_matrix.cpu().numpy()
    # # Make the adjency matrix of a bipartite == square matrix B=[[1, A], [A.T, 0]]
    # B = np.zeros(shape=(Affinity_matrix.shape[0] + Affinity_matrix.shape[1],
    #                     Affinity_matrix.shape[0] + Affinity_matrix.shape[1]))
    # B[0:Affinity_matrix.shape[0], B.shape[1] - Affinity_matrix.shape[1]::] = Affinity_matrix
    # B[B.shape[1] - Affinity_matrix.shape[1]::, 0:Affinity_matrix.shape[0]] = Affinity_matrix.T
    # data_nnz = model.bip_matrix[model.rows_idx, model.col_idx].type(torch.int64).cpu().numpy()
    # adjacency_matrix = B.copy()

    # we do not use the weights
    D = csr_matrix((np.ones(model.rows_idx.shape[0]), (model.rows_idx.cpu().numpy(), model.col_idx.cpu().numpy())),
                   shape=(model.input_size_rows, model.input_size_cols))

    # if ORIGINAL_PLOT:
    # sparse matrix, where the rows have the AHI value
    D_color = D.copy()
    for row_ in range(0, D_color.shape[0]):
        val_on_row = np.where(D_color.getrow(row_).toarray() > 0)[1]
        D_color[row_, :] = int(model.bip_matrix[row_, 0].item())

    plt.figure(figsize=(3,10))
    plt.title('Initial adjacency matrix')
    plt.spy(D, markersize=0.1, alpha=0.5)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6), dpi=300)
    fig, ax = plt.subplots(1, 2)
    my_cmap = cm.get_cmap('tab10').copy()
    my_cmap.set_under('white')
    ax[0].spy(D, markersize=0.1, alpha=0.5)
    ax[0].axis('off')
    plt.grid(False)
    im = ax[1].imshow(D_color.todense(), interpolation='none', cmap=my_cmap, vmin=1)
    plt.axis('off')
    plt.grid(False)
    # plt.tight_layout()
    # get the colors of the values, according to the
    targets = torch.unique(model.bip_matrix[:, 0]).detach().cpu().numpy()
    # colormap used by imshow
    colors = [im.cmap(im.norm(value)) for value in targets]
    # create a patch (proxy artist) for every color
    patches = [mpatches.Patch(color=colors[i], label=f"AHI lvl {targets[i]}")
               for i in range(len(targets))]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.suptitle(f'HM-LDM adjacency matrix')
    plt.draw()
    plt.show()


    # embedding ordering
    w_idx = model.latent_w.argmax(1)
    z_idx = model.latent_z.argmax(1)

    f_w = w_idx.argsort().cpu().numpy()
    f_z = z_idx.argsort().cpu().numpy()


    # order the adj matrix based on the community allocations
    D = D[:, f_w][f_z]
    plt.figure(figsize=(3,10))
    plt.title('HM-LDM re-ordered adjacency matrix')
    plt.spy(D, markersize=0.1, alpha=0.5)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # sparse matrix, where the rows have the AHI value
    D_color = D.copy()
    for row_ in f_z:
        val_on_row = np.where(D_color.getrow(row_).toarray() > 0)[1]
        D_color[row_, :] = int(model.bip_matrix[row_, 0].item())
    D_color = D_color[:, f_w][f_z]

    plt.figure(figsize=(10, 6), dpi=300)
    fig, ax = plt.subplots(1, 2)
    my_cmap = cm.get_cmap('tab10').copy()
    my_cmap.set_under('white')
    ax[0].spy(D, markersize=0.1, alpha=0.5)
    ax[0].axis('off')
    plt.grid(False)
    im = ax[1].imshow(D_color.todense(), interpolation='none', cmap=my_cmap, vmin=1)
    plt.axis('off')
    plt.grid(False)
    # plt.tight_layout()
    # get the colors of the values, according to the
    targets = torch.unique(model.bip_matrix[:, 0]).detach().cpu().numpy()
    # colormap used by imshow
    colors = [im.cmap(im.norm(value)) for value in targets]
    # create a patch (proxy artist) for every color
    patches = [mpatches.Patch(color=colors[i], label=f"AHI lvl {targets[i]}")
               for i in range(len(targets))]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.suptitle(f'HM-LDM re-ordered adjacency matrix')
    plt.draw()
    plt.show()

    # # sparse matrix, where the rows have the AHI value
    # for row_ in range(0, D.shape[0]):
    #     val_on_row = np.where(D.getrow(row_).toarray() > 0)[1]
    #     D[row_, val_on_row] = int(model.bip_matrix[row_, 0].item())
    #
    # from matplotlib import cm
    # d=D.todense()
    # my_cmap = cm.get_cmap('tab10').copy()
    # my_cmap.set_under('white')
    # plt.imshow(d,interpolation='none', cmap=my_cmap, vmin=1)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()
    #
    # fig, ax = plt.subplots()
    # im = ax.imshow(D.todense(), interpolation='nearest', vmin=1, cmap='Set1')
    # # fig.colorbar(im, extend='min')
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()


def apply_pca_embeddings(model):

    pca = PCA(n_components=2)
    from sklearn.preprocessing import normalize
    normalize_embedding =normalize(model.latent_z_soft.cpu().detach().numpy(), axis=1, norm='max')

    pca_components = pca.fit_transform(model.latent_z_soft.cpu().detach().numpy())

    labels = model.bip_matrix[::, 0].cpu().numpy()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)

    plt.scatter(pca_components[:,0],
                pca_components[:,1],
                c=labels,
                s=5)
                # ax=ax)
    ax.legend(labels)
    ax.grid()
    plt.show()
    print(pca.explained_variance_ratio_)

# %%
def poisson_matrix_svm_pca(model):
    """
    pair_{i,j} = (latent_distance_{i, j} , gamma_{i}, gamma_{j})
    The idea is to use the pair_ij three elements as a three feature vector, the dimension will be equal to the number
    of rows, which we have their targets
    :return:
    """
    # target_col_idx = torch.where(model.col_idx == 0)[0].to(device)
    # model.rows_idx changes value when it encounter the first element of the second row as is the target (never zero)
    # target_row_idx = np.where(np.diff(model.rows_idx.detach().cpu().numpy(), prepend=np.nan))[0]
    target_col_idx = 0

    delta_latent = ((model.latent_z_soft[model.rows_idx] - model.latent_w_soft[model.col_idx]) ** 2).sum(-1) ** (
            1 / 2)

    delta_latent = ((model.latent_z_soft -model.latent_w_soft[target_col_idx, :]) ** 2).sum(-1) ** (1 / 2)
    gamma_col = torch.full_like(model.gamma_rows, fill_value= model.gamma_cols[target_col_idx].item())
    # logit_u_miss = model.gamma_rows + gamma_col - model.delta * delta_latent

    # concatenate features
    pair_ij = np.vstack((model.delta * delta_latent.detach().cpu().numpy(),
                                model.gamma_rows.detach().cpu().numpy(),
                                 gamma_col.detach().detach().cpu().numpy())).T

                                 # model.bip_matrix[::, 0].detach().cpu().numpy())).T
    pair_ij = np.concatenate((pair_ij, model.latent_z_soft.detach().cpu().numpy()), axis=1)
    # concatenate the target
    target = model.bip_matrix[::, 0].detach().cpu().numpy().copy()
    target = np.expand_dims(a=target, axis=1)
    pair_ij = np.concatenate((pair_ij, target ), axis=1)

    _ = plt.hist(pair_ij[::,-1], bins='auto')  # arguments are passed to np.histogram
    plt.title("Histogram with 'auto' bins")
    plt.show()


    clf = SVC(gamma='auto')
    clf.fit(X=pair_ij[::, 0:-1], y=pair_ij[::,-1])

    pred = clf.predict(pair_ij[::, 0:-1])
    f1_score(y_true=pair_ij[::,-1], y_pred=pred, average='weighted')

    plt.figure(figsize=(10, 10))
    plt.title('SVM on the pairs of Poisson')
    # plt.xlabel('Z1')
    # plt.ylabel('Z2')
    # plt.legend()
    plt.grid(True)
    plt.scatter(pair_ij[::, 0],
                pair_ij[::, 1],
                c=pair_ij[::,-1],
                s=20)
    plt.show()

    pca = PCA(n_components=8)
    pca_components = pca.fit_transform(pair_ij[::, 0:-1])
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)

    plt.scatter(pca_components[:,0],
                pca_components[:,1],
                c=pair_ij[::,-1],
                s=5)
                # ax=ax)
    ax.legend(pair_ij[::,-1])
    ax.grid()
    plt.show()
    print(pca.explained_variance_ratio_)

def implement_xgb(model):
    import xgboost as xgb
    target = model.bip_matrix[::, 0].detach().cpu().numpy().copy()
    xgb_model = xgb.XGBClassifier(
        random_state=config['seed'],
        tree_method='gpu_hist',
        predictor='gpu_predictor',  # 'cpu_predictor', #'gpu_predictor',

        objective='multi:softprob',  # multi:softproba
        eval_metric='mlogloss',  # 'mlogloss',
        n_estimators=2000,  # 100000, #100000,
        # early_stopping_rounds=1000,
        eta=1e-04,
        # learning_rate=0.01,
        reg_alpha=1,
        reg_lambda=1,
        gamma=1,
        max_bin=256,
        max_depth=20,
        sampling_method='gradient_based',
        # verbose_eval=400,
        num_class=np.unique(target))
    # create feature matrix
    delta_latent = ((model.latent_z_soft -model.latent_w_soft[0, :]) ** 2).sum(-1) ** (1 / 2)
    gamma_col = torch.full_like(model.gamma_rows, fill_value= model.gamma_cols[0].item())
    # concatenate features
    pair_ij = np.vstack((model.delta * delta_latent.detach().cpu().numpy(),model.gamma_rows.detach().cpu().numpy(),
                                 gamma_col.detach().detach().cpu().numpy())).T
    pair_ij = np.concatenate((pair_ij, model.latent_z_soft.detach().cpu().numpy()), axis=1)
    # concatenate the target
    target = np.expand_dims(a=target, axis=1).astype(int)
    # XGB class needs class 0
    target = target - 1
    data_matrix = np.concatenate((pair_ij, target), axis=1)
    # split data
    D_train, D_test = train_test_split(data_matrix, test_size=.3,
                                       train_size=.7,
                                       random_state=7182,
                                       shuffle=True,
                                       stratify=data_matrix[:, -1])
    D_train, D_test = train_test_split(D_train, test_size=fixed_test_samples,
                                       train_size=D_train.shape[0] - fixed_test_samples,
                                       random_state=self.config['seed'],
                                       shuffle=self.shuffle_plits,
                                       stratify=dataset.ahi)
    # get the validation et from training
    D_train, D_val = train_test_split(D_train, test_size=.2, train_size=.8,
                                      random_state=self.config['seed'],
                                      shuffle=self.shuffle_plits,
                                      stratify=D_train.ahi)

    # train the model
    xgb_model.fit(X=D_train[:, 0:-1], y=D_train[:,-1].astype(int))
    # predict on training
    dtrain_predprob = xgb_model.predict_proba(D_train[:, 0:-1])
    train_f1_score = f1_score(y_true=D_train[:,-1].astype(int), y_pred=dtrain_predprob.argmax(axis=1), average='weighted')
    print(f'\n Training F1 score: {train_f1_score}')
    # predict on test
    dtest_predprob = xgb_model.predict_proba(D_test[:, 0:-1])
    test_f1_score = f1_score(y_true=D_test[:, -1].astype(int), y_pred=dtest_predprob.argmax(axis=1), average='weighted')
    print(f'\n Test F1 score: {test_f1_score}')
    pass

def pca(model):
    """
    Plotting of the PCA utilizing the PCA library. Much better visualization :)
    https://stackoverflow.com/questions/50654620/add-legend-to-scatter-plot-pca
    :param model:
    :return:
    """
    target = model.bip_matrix[::, 0].detach().cpu().numpy().copy()
    # create feature matrix
    delta_latent = ((model.latent_z_soft -model.latent_w_soft[0, :]) ** 2).sum(-1) ** (1 / 2)
    gamma_col = torch.full_like(model.gamma_rows, fill_value= model.gamma_cols[0].item())
    # concatenate features
    pair_ij = np.vstack((model.delta * delta_latent.detach().cpu().numpy(),model.gamma_rows.detach().cpu().numpy(),
                                 gamma_col.detach().detach().cpu().numpy())).T
    pair_ij = np.concatenate((pair_ij, model.latent_z_soft.detach().cpu().numpy()), axis=1)
    # concatenate the target
    target = np.expand_dims(a=target, axis=1).astype(int)
    # XGB class needs class 0
    target = target - 1
    data_matrix = np.concatenate((pair_ij, target), axis=1)
    from pca import pca

    # Or reduce the data towards 2 PCs
    pca_model = pca(n_components=2)

    # Load example dataset
    import pandas as pd
    import sklearn
    from sklearn.datasets import load_iris
    columns = ['Z_1','Z_2','Z_3','Z_4','Z_5','Z_6','Z_7','Z_8', 'dij', 'Gi', 'Gj']
    X = pd.DataFrame(data=data_matrix[:, 0:-1], columns=columns, index=np.squeeze(a=target, axis=1))

    # Fit transform
    results = pca_model.fit_transform(X)

    # Plot explained variance
    fig, ax = pca_model.plot()
    # Scatter first 2 PCs
    fig, ax = pca_model.scatter()

    # Make biplot with the number of features
    fig, ax = pca_model.biplot(n_feat=4)

# %% Fully Connected Network


# %% main
if __name__ == '__main__':
    # latent dimensions of the embeddings W
    latent_dim = 8

    # Define the LDM model
    model = HMLDM(config=config, latent_dim=latent_dim, delta=32, p=8).to(device)

    # Start the training process
    losses = train_model(model=model, epoch_num=2000)
    # visualize_embedding(model=model)

