# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 09:09:10 2021

@author: nnak
https://towardsdatascience.com/spectral-clustering-82d3cff3d3b7
Clustering is a widely used unsupervised learning method. The grouping is such that points
in a cluster are similar to each other, and less similar to points in other clusters. Thus,
it is up to the algorithm to find patterns in the data and group it for us and, depending
on the algorithm used, we may end up with different clusters.

In spectral clustering, the data points are treated as nodes of a graph. Thus, clustering is treated as
a graph partitioning problem. The nodes are then mapped to a low-dimensional space that can be easily
segregated to form clusters. An important point to note is that no assumption is made about the shape/form
of the clusters.
"""

import scipy
from scipy import sparse
import numpy as np
import torch
from typing import Optional
import networkx as nx
from sklearn.manifold import MDS


class Spectral_clustering_init():
    """The methods of this class must have the same names form the HM-LDM model"""
    def __init__(self, num_of_eig=10, method='Adjacency', device=None, bipartite:Optional[bool]=False):

        self.num_of_eig = num_of_eig
        self.method = method
        self.device = device

    def spectral_clustering(self):

        # from our class
        # sparse_i = self.rows_idx.cpu().numpy()
        # sparse_j = self.col_idx.cpu().numpy()
        # for notebook
        sparse_i=self.edge_pos_i.cpu().numpy()
        sparse_j=self.edge_pos_j.cpu().numpy()

        idx_shape = sparse_i.shape[0]
        if (sparse_i < sparse_j).sum() == idx_shape:
            sparse_i_new = np.concatenate((sparse_i, sparse_j))
            sparse_j_new = np.concatenate((sparse_j, sparse_i))

            sparse_i = sparse_i_new
            sparse_j = sparse_j_new

        V = np.ones(sparse_i.shape[0])

        # affinity for the notebook exercise
        Affinity_matrix = sparse.coo_matrix((V, (sparse_i, sparse_j)), shape=(self.input_size, self.input_size))

        # affinity for our class
        # Affinity_matrix = sparse.coo_matrix((V, (sparse_i, sparse_j)),
        #                                     shape=(self.bip_matrix.shape[0], self.bip_matrix.shape[1]))
        # the following commented code workrs must be implemented form the bipartite matrix
        # if bipartite:
        #     # Make the adjency matrix of a bipartite == square matrix B=[[1, A], [A.T, 0]]
        #     B = np.zeros(shape=(Affinity_matrix.shape[0] + Affinity_matrix.shape[1],
        #                        Affinity_matrix.shape[0] + Affinity_matrix.shape[1]))
        #     B[0:Affinity_matrix.shape[0], B.shape[1]-Affinity_matrix.shape[1]::] = Affinity_matrix.toarray()
        #     B[B.shape[1]-Affinity_matrix.shape[1]::, 0:Affinity_matrix.shape[0]] = Affinity_matrix.toarray().T
        #
        #     Affinity_matrix = B.copy()

        if self.method == 'Adjacency':
            # Requires a N x N matrix as input
            eig_val, eig_vect = scipy.sparse.linalg.eigsh(Affinity_matrix,
                                                          self.num_of_eig, which='LM')
            X = eig_vect.real
            rows_norm = np.linalg.norm(X, axis=1, ord=2)
            U_norm = (X.T / rows_norm).T

        elif self.method == 'Normalized_sym':
            n, m = Affinity_matrix.shape
            diags = Affinity_matrix.sum(axis=1).flatten()
            D = sparse.spdiags(diags, [0], m, n, format="csr")
            L = D - Affinity_matrix
            with scipy.errstate(divide="ignore"):
                diags_sqrt = 1.0 / np.sqrt(diags)
            diags_sqrt[np.isinf(diags_sqrt)] = 0
            DH = sparse.spdiags(diags_sqrt, [0], m, n, format="csr")
            tem = DH @ (L @ DH)
            eig_val, eig_vect = scipy.sparse.linalg.eigsh(tem, self.num_of_eig, which='SA')
            X = eig_vect.real
            self.X = X
            rows_norm = np.linalg.norm(X, axis=1, ord=2)
            U_norm = (X.T / rows_norm).T

        elif self.method == 'Normalized':
            n, m = Affinity_matrix.shape
            diags = Affinity_matrix.sum(axis=1).flatten()
            D = sparse.spdiags(diags, [0], m, n, format="csr")
            L = D - Affinity_matrix
            with scipy.errstate(divide="ignore"):
                diags_inv = 1.0 / diags
            diags_inv[np.isinf(diags_inv)] = 0
            DH = sparse.spdiags(diags_inv, [0], m, n, format="csr")
            tem = DH @ L
            eig_val, eig_vect = scipy.sparse.linalg.eigs(tem, self.num_of_eig, which='SR')

            X = eig_vect.real
            self.X = X
            U_norm = X

        elif self.method == 'MDS':
            n, m = Affinity_matrix.shape

            G = nx.Graph(Affinity_matrix)

            max_l = 0
            N = G.number_of_nodes()
            pmat = np.zeros((N, N)) + np.inf
            paths = nx.all_pairs_shortest_path_length(G)
            for node_i, node_ij in paths:
                for node_j, length_ij in node_ij.items():
                    pmat[node_i, node_j] = length_ij
                    if length_ij > max_l:
                        max_l = length_ij

            pmat[pmat == np.inf] = max_l + 1
            print('shortest path done')

            embedding = MDS(n_components=self.num_of_eig, dissimilarity='precomputed')
            U_norm = embedding.fit_transform(pmat)
        else:
            print('Invalid Spectral Clustering Method')

        return torch.from_numpy(U_norm).float().to(self.device)
