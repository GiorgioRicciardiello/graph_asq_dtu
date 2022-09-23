"""
Hybrid-Membership Latent Distance Model for bipartite newtork
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
import xgboost as xgb
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
        # initialize class to get the data
        ASQ_Graph.__init__(self, config=config)

        # bipartite matrix
        self.get_graph()
        self.bip_matrix = self.get_adjacency_matrix()
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
        # get indexes of the matrix where value >0. For rows and columns
        self.bip_matrix = torch.from_numpy(self.bip_matrix).float().to(device)
        self.rows_idx, self.col_idx = torch.where(self.bip_matrix > 0)

        self.input_size_rows = self.bip_matrix.shape[0]  # rows
        self.input_size_cols = self.bip_matrix.shape[1]  # columns

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

            # graph_weights * (gamma_i + delta_j - d_{z_{i},w_{j}})
            link_nll = self.bip_matrix[self.rows_idx, self.col_idx] * (
                    self.gamma_rows[self.rows_idx] + self.gamma_cols[self.col_idx])  # - delta_latent)

            link_nll = link_nll.sum()

            # calculate the total nll of the LDM -L = -sum[Weights_ij*ln(λ_ij)] + sum[λ_ij]
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

            # matrix multiplication sum[exp(gamma_i + delta_j - dist_matrix)] == exp(gamma_i) * exp(delta_j) *
            # dist_matrix.T)
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
        return loss


# %% Model training
def train_model(model, epoch_num=2000):
    """
    Function for training the HM-LDM

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
    # plt.savefig(r'...\soft_latent_w_z.png',
    #             dpi=300)

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
    # plt.savefig(r'...\soft_latent_w_z.png',
    #             dpi=300)

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

def implement_xgb(model):
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

def implement_pca(model):
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
    visualize_embedding(model=model)

    # visualize embeddings
    re_ordering_adjacency_matrix(model=model)
    implement_pca(model=model)

    # implement the classifier
    implement_xgb(model=model)


