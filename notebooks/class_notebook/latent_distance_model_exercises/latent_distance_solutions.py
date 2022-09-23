"""
Let us now define a Latent Distance Model  (𝐿𝐷𝑀) with
random effects under a Poisson likelihood formulation.

working with unipartite undirected networksand thus
we only consider the upper triangular part of the
symmetric adjacency matrix (i<j)
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


# %% LDM model
# class LDM(nn.Module, Spectral_clustering_init):
#     """    Latent Distance Model class"""
#
#     def __init__(self, input_size:int, latent_dim:int):
#         """
#
#         :param input_size: dimension of the ASQ (rows, columns)
#         :param latent_dim: dimension of the latent space we want to generate
#         """
#         super(LDM, self).__init__()
#         # initialization of the class that is responsible for the spectral initialization of the latent variables Z
#         # available initialization choices are: 'Adjacency', 'Normalized_sym', 'Normalized', 'MDS'
#         Spectral_clustering_init.__init__(self, num_of_eig=latent_dim, method='Adjacency', device=device)
#
#         # dimensions
#         self.input_size = input_size
#         self.latent_dim = latent_dim
#
#         # initially we want to learn the scales of the random effects separately
#         self.scaling_RE = True
#
#     def read_data(self, path):
#         # MODIFY FOR BIPARTITE
#         """
#         reads input data:
#         Netwok Edgelist (upper triangular part of the adjacency matrix for the unipartite undirected network case):
#         - edge_pos_i: input data, link row positions i with i<j (edge row position)
#         - edge_pos_j: input data, link column positions j with i<j (edge column position)
#         """
#         # input data, link (edge) rows i positions with i<j
#         self.edge_pos_i = torch.from_numpy(np.loadtxt(edge_path + 'edge_pos_i.txt')).long().to(device)
#
#         # input data, link (edge) column positions with i<j
#         self.edge_pos_j = torch.from_numpy(np.loadtxt(edge_path + 'edge_pos_j.txt')).long().to(device)
#
#     def init_parameters(self):
#         """define and initialize model parameters"""
#         # Parameters
#         # Random effects, the random effects are single values for each observation
#         self.gamma = nn.Parameter(torch.randn(self.input_size, device=device))
#         print(f'\ngamma shape: {self.gamma.shape}')
#         # Latent Variables
#         # initialize Z based on the leading eigenvectors of the adjacency matrix
#         # self.spectral_data = self.spectral_clustering() -> comment for the ASQ
#         # print(f'\nSpectral cluster data shape: {self.spectral_data.shape}')
#
#         # define the latent space we want to generate
#         self.latent_z = nn.Parameter(torch.rand(self.input_size_n1, 0))
#         self.latent_w = nn.Parameter(torch.rand(self.input_size_n2, 0))
#
#         print(f'\nlatent_z shape: {self.latent_z.shape}')
#         print(f'\nlatent_w shape: {self.latent_w.shape}')
#
#     def LDM_Poisson_NLL(self, epoch):
#         """Poisson log-likelihood ignoring the log(k!) constant"""
#         self.epoch = epoch
#
#         if self.scaling_RE:
#             # We will spend 500 iteration on learning the random effects, defining a rate as exp(gamma_i+gamma_j)
#             mat = torch.exp(self.gamma.unsqueeze(1) + self.gamma)  # NxN matrix containing all pairs i,j
#
#             # multiply with 0.5 to account for the fact that we calculated the whole NxN rate matrix
#             # subtract the diagonal of the rate matrix since self-links are not allowed
#             non_link_nll = 0.5 * (mat - torch.diag(torch.diagonal(mat))).sum()
#
#             # calculate now the link term of the log-likelihood
#             link_nll = (self.gamma[self.edge_pos_i] + self.gamma[self.edge_pos_j]).sum()
#
#             # calculate the total nll of the LDM
#             loss = -link_nll + non_link_nll
#
#             if self.epoch == 500:
#                 # after 500 iteration stop the scaling and switch to full model training
#                 self.scaling_RE = False
#         else:
#             # PLEASE ADD HERE THE LOSS FUNCTION
#             # cdist: Computes batched the p-norm distance between each pair of the two collections of row vectors.
#             mat = torch.exp(-((torch.cdist(self.latent_z, self.latent_z, p=2)) + 1e-06))
#             z_pdist1 = 0.5 * torch.mm(
#                 torch.exp(self.gamma.unsqueeze(0)),
#                 (torch.mm( (mat - torch.diag(torch.diagonal(mat))), torch.exp(self.gamma).unsqueeze(-1) ))
#                 )
#
#             eucl_dist_z_pdist2 = (self.latent_z[self.edge_pos_i] - self.latent_z[self.edge_pos_j] + 1e-06) ** 2
#             z_pdist2 = ((self.gamma[self.edge_pos_i] + self.gamma[self.edge_pos_j])-eucl_dist_z_pdist2.sum(-1) ** 0.5
#                         ).sum()
#
#             # z_pdist2 = (-((
#             #     ((self.latent_z[self.edge_pos_i] - self.latent_z[self.edge_pos_j] + 1e-06) ** 2).sum(-1))) ** 0.5 +
#             #             self.gamma[self.edge_pos_i] + self.gamma[self.edge_pos_j]
#             #             ).sum()
#
#             loss = -z_pdist2 + z_pdist1
#
#         return loss

#%% HM-LDM model
class HMLDM(nn.Module, Spectral_clustering_init):
    '''
    Hybrid Membership-Latent Distance Model class
    '''

    def __init__(self, input_size, latent_dim, delta=1):
        super(HMLDM, self).__init__()

        # initialization of the class that is responsible for the spectral initialization of the latent variables Z
        # available initialization choices are: 'Adjacency', 'Normalized_sym', 'Normalized', 'MDS'
        Spectral_clustering_init.__init__(self, num_of_eig=latent_dim, method='Adjacency', device=device,
                                          bipartite=False)

        # dimensions
        self.input_size = input_size
        self.latent_dim = latent_dim

        # initially we want to learn the scales of the random effects separately
        self.scaling_RE = True

        # hyperparameter controlling the simplex volume
        self.delta = delta

    def read_data(self):
        '''
        reads input data:

        Netwok Edgelist (upper triangular part of the adjacency matrix for the unipartite undirected network case):

        - edge_pos_i: input data, link row positions i with i<j (edge row position)

        - edge_pos_j: input data, link column positions j with i<j (edge column position)

        '''

        edge_pos_i_path = r'C:\Users\giorg\OneDrive_ItaloCol\DTU\MSc_courses_material\Graph Representation Learning\graph_asq\notebooks\class_notebook\latent_distance_model_exercises\networks\cora\edge_pos_i.txt'
        self.edge_pos_i = torch.from_numpy(np.loadtxt(edge_pos_i_path)).long().to(device)

        edge_pos_j_path = r'C:\Users\giorg\OneDrive_ItaloCol\DTU\MSc_courses_material\Graph Representation Learning\graph_asq\notebooks\class_notebook\latent_distance_model_exercises\networks\cora\edge_pos_j.txt'
        self.edge_pos_j = torch.from_numpy(np.loadtxt(edge_pos_j_path)).long().to(device)

        # # input data, link (edge) rows i positions with i<j
        # self.edge_pos_i = torch.from_numpy(np.loadtxt(edge_path + 'edge_pos_i.txt')).long().to(device)
        # np.savetxt(edge_path + 'edge_pos_i.txt', np.loadtxt(edge_path + 'edge_pos_i.txt'), fmt='%s')
        #
        # # input data, link (edge) column positions with i<j
        # self.edge_pos_j = torch.from_numpy(np.loadtxt(edge_path + 'edge_pos_j.txt')).long().to(device)
        # np.savetxt(edge_path + 'edge_pos_j.txt', np.loadtxt(edge_path + 'edge_pos_j.txt'), fmt='%s')

    def init_parameters(self):
        '''
        define and initialize model parameters
        '''

        self.Softmax = nn.Softmax(1)

        # Parameters

        # Random effects
        self.gamma = nn.Parameter(torch.randn(self.input_size, device=device))

        # Latent Variables

        # initialize Z based on the leading eigenvectors of the adjacency matrix
        self.spectral_data = self.spectral_clustering()
        self.latent_w1 = nn.Parameter(self.spectral_data)

    def LDM_Poisson_NLL(self, epoch):
        '''
        Poisson log-likelihood ignoring the log(k!) constant

        '''
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

            mat = torch.exp(-self.delta * ((torch.cdist(self.latent_w, self.latent_w, p=2)) + 1e-06))
            z_pdist1 = 0.5 * torch.mm(torch.exp(self.gamma.unsqueeze(0)), (
                torch.mm((mat - torch.diag(torch.diagonal(mat))), torch.exp(self.gamma).unsqueeze(-1))))

            z_pdist2 = (-self.delta * (((
                ((self.latent_w[self.edge_pos_i] - self.latent_w[self.edge_pos_j] + 1e-06) ** 2).sum(-1))) ** 0.5) +
                        self.gamma[self.edge_pos_i] + self.gamma[self.edge_pos_j]).sum()

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

    return losses


def link_prediction_hm(model, samples_path):
    # file denoting rows i of hidden links and negative samples, with i<j
    total_samples_i = torch.from_numpy(np.loadtxt(samples_path + 'total_samples_i.txt')).long().to(device)

    # file denoting columns j of hidden links and negative samples, with i<j
    total_samples_j = torch.from_numpy(np.loadtxt(samples_path + 'total_samples_j.txt')).long().to(device)

    # target vector having 0 if the missing pair is a negative sample and 1 if the pair considers a hidden (removed) edge
    target = torch.from_numpy(np.loadtxt(samples_path + 'target.txt')).long().to(device)

    with torch.no_grad():

        z_pdist_miss = (((model.latent_w[total_samples_i] - model.latent_w[total_samples_j]) ** 2).sum(-1)) ** 0.5
        logit_u_miss = model.gamma[total_samples_i] + model.gamma[total_samples_j] -model.delta * z_pdist_miss
        rates = logit_u_miss

        # calculate AUC-PR
        precision, recall, thresholds = metrics.precision_recall_curve(target.cpu().data.numpy(),
                                                                       rates.cpu().data.numpy())
        auc_pr = metrics.auc(recall, precision)

        # calculate AUC-ROC

        auc_roc = metrics.roc_auc_score(target.cpu().data.numpy(), rates.cpu().data.numpy())
        fpr, tpr, thresholds = metrics.roc_curve(target.cpu().data.numpy(), rates.cpu().data.numpy())

    return auc_roc, fpr, tpr, auc_pr, precision, recall


# %% main
if __name__ == '__main__':
    #Network characteristics
    dataset = 'cora'  # dataset name

    # latent dimensions of the embeddings Z
    latent_dim = 2

    # path to the folder containing the network edge list
    # this version contains only 50% of the network edges, the rest will be used for link prediction
    # edge_path = f'./networks/{dataset}/'
    edge_path = r'C:\Users\giorg\OneDrive_ItaloCol\DTU\MSc_courses_material\Graph Representation Learning\graph_asq\notebooks\class_notebook\latent_distance_model_exercises\networks\cora'
    # path to the folder containing the sampled pairs for link prediction

    # (the 50% hidden edges and the same amount of negative samples)
    # samples_path = f'./networks/{dataset}/samples_link_prediction/'
    samples_path = r'C:\Users\giorg\OneDrive_ItaloCol\DTU\MSc_courses_material\Graph Representation Learning\graph_asq\notebooks\class_notebook\latent_distance_model_exercises\networks\cora\samples_link_prediction'
    # Size of the network
    N_size = int(np.loadtxt(edge_path + 'network_size.txt'))

    # # Define the LDM model
    # model = LDM(input_size=N_size, latent_dim=latent_dim).to(device)

    # Define the HMLDM model
    model = HMLDM(input_size=N_size, latent_dim=latent_dim, delta=10).to(device)

    # Start the training process
    losses = train_model(model=model)

    # make link prediction
    link_prediction_hm(model=model,samples_path=samples_path )
    # plot curves
    plot_auc(auc_roc, fpr, tpr, auc_pr, precision, recall)














    def visualize_embedding():
        # Read the node labels
        labels = np.loadtxt(edge_path + "labels.txt")[:, 1]
        plt.figure(figsize=(10, 10))
        plt.title('LDM 2-Dimensional Embedding Space')
        plt.xlabel('Z1')
        plt.ylabel('Z2')
        plt.grid(False)
        plt.axis('off')
        plt.scatter(model.latent_z[:, 0].detach().cpu().numpy(), model.latent_z[:, 1].detach().cpu().numpy(), c=labels,
                    s=20)
        plt.show()

        pass
