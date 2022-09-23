"""
Class dedicate to create the biprartite graph and also returns the matrix of the bipartite

Bipartite matrix = ASQ_Graph.get_adjacency_matrix()
graph = ASQ_Graph.get_graph()

"""
import pathlib

from config.config import config
from typing import Optional, Union, Dict, Tuple, List, Any, Type
from pathlib import Path
from src.data_preproc.pre_processing import PreProcessASQ
import pandas as pd
import seaborn as sns
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
from collections import Counter, OrderedDict


class ASQ_Graph(PreProcessASQ):
    def __init__(self, config: dict):
        # super().__init__(config)
        self.config = config

        self.pre_proces_path = self.config['root_dir'].joinpath(pathlib.Path(r'data/processed'))
        self.pre_proces_path_graph = self.pre_proces_path.parents[0] / 'graph' / 'bipartite_asq.graphml'
        self.figure_path = self.config['root_dir'].joinpath(pathlib.Path(r'reports/figures/graphs'))
        if not self.pre_proces_path_graph.is_file():
            PreProcessASQ.__init__(self, config=config)
            self.get_preprocessed_asq()
            print(f'\nGenerating graph')
            self.generate_bipartite_graph()
            self._save_graph()
        else:
            print(f'\nGraph is ready to be called from the class')

    # %% save functions
    def _save_graph(self):
        """
        Save the graph using nx documentation
        https://networkx.org/documentation/latest/reference/readwrite/generated/networkx.readwrite.graphml.
        write_graphml.html
        """
        nx.write_graphml_lxml(G=self.bip, path=self.pre_proces_path_graph)
        if self.pre_proces_path_graph.is_file():
            print(f'\nGraph saved in {self.pre_proces_path_graph}')

    # %% getters
    def get_graph(self) -> nx.Graph:
        """Get bipartite graph from preprocessed ASQ"""
        graph = nx.read_graphml(path=self.pre_proces_path_graph)
        nx.info(graph)
        self.bip = graph
        return graph

    def get_adjacency_matrix(self):
        """Return the adjency matrix of the bipartite graph"""
        subject_nodes = [node_ for node_ in self.bip.nodes() if node_.isnumeric()] # subjects are the rows
        question_nodes = [node_ for node_ in self.bip.nodes() if not node_.isnumeric()] # subjects are the rows

        M = bipartite.biadjacency_matrix(G=self.bip,row_order=subject_nodes, column_order =question_nodes,
                                         weight='weight')
        M = M.toarray()
        print(f'\nReturning Bipartite Matrix of dimension {M.shape}')
        return M


    # %% Generate bipartite
    def generate_bipartite_graph(self):
        """
        Bipartite graphs B = (U, V, E) have two node sets U,V and edges in E that only connect nodes from opposite sets
        :return:
        """
        DISPLAY_GRAPH = False
        if not hasattr(self, 'asq'):
            self.asq = self.get_preprocessed_asq()

        predictors = [col_ for col_ in [*self.asq.columns] if col_ != 'subject']
        # predictors = [col_ for col_ in predictors if col_ != 'subject']
        # select only the predictors
        asq_r = self.asq.loc[0:10, predictors[0:16]]
        # asq_r = self.asq[predictors]

        # Generate the bipartite graph structure
        bip = nx.Graph()
        # Two columns of noes U=subjects, V=questions
        bip.add_nodes_from(asq_r.index, bipartite='subjects')
        bip.add_nodes_from(asq_r.columns, bipartite='questions')

        # Assign the weighted edges between the nodes U and V but not across
        s = asq_r.stack()
        # Connect only where weights>0
        weighted_edges = [(s_[0][0], s_[0][1], s_[1]) for s_ in s.iteritems() if s_[1]>0]  # ((node_u, node_v, weight))
        bip.add_weighted_edges_from(weighted_edges)

        # Visualize
        print(bip.edges(data=True))
        # verify is bipartite and weighted
        if bipartite.is_bipartite(bip): #nx.is_connected(bip) and
            print(f'\nBipartite graph has been connected')
        else:
            raise ValueError(f'\nBipartite graph was NOT been connected')
        if nx.is_weighted(bip, edge=None, weight="weight"):
            print(f'\nBipartite graph is weighted')
        else:
            raise ValueError(f'\nBipartite graph is NOT weighted')
        # visualize
        if DISPLAY_GRAPH:
            plt.figure(figsize=(10, 15))
            plt.title('Bipartite Graph 10 Subjects')
            nx.draw_networkx(
                bip,
                pos=nx.drawing.layout.bipartite_layout(G=bip, nodes=[*asq_r.index]),
                width=2)
            edges = bip.edges()
            print(f'\nedges: {edges}')
            plt.tight_layout()

            # fig1 = plt.gcf()
            plt.savefig(self.figure_path / r'sub_bipartite.png', bbox_inches='tight', dpi=300)
            plt.show()
            plt.draw()

        self.bip = bip

    def _plot_graph(self):
        # https://www.youtube.com/watch?v=PouhDHfssYA&ab_channel=SepinoudAzimi
        # https://coderzcolumn.com/tutorials/data-science/network-analysis-in-python-important-structures-a
        # nd-bipartite-graphs-networkx#5
        # https://ericmjl.github.io/Network-Analysis-Made-Simple/04-advanced/01-bipartite/
        if not hasattr(self, 'bip'):
            self.bip = self.get_graph()
        nx.draw_networkx(self.bip, pos=nx.drawing.layout.bipartite_layout(self.bip, ['0', '1', '2', '3']), width=2)
        plt.show()

    def _degrees_graph(self):
        # person_nodes = self.bip.nodes()
        # edges = self.bip.edges()
        subject_nodes = [node_ for node_ in self.bip.nodes() if isinstance(node_, int)]

        deg_question,deg_subjects = bipartite.degrees(B=self.bip, nodes=subject_nodes)
        dict(deg_question)
        dict(deg_subjects)
        print(f'\nCount of different degrees for the subjects nodes {Counter(dict(deg_subjects).values())}')
        self.subject_degree = dict(deg_subjects)

    def graph_density(self):
        """
        Graph density represents the ratio between the edges present in a graph and the maximum number of edges that
        the graph can contain. Conceptually, it provides an idea of how dense a graph is in terms of edge connectivity.
        :return:
        """
        # bottom_nodes, top_nodes = bipartite.sets(self.bip)  # sujects, questions
        subject_nodes = [node_ for node_ in self.bip.nodes() if isinstance(node_, int)]
        question_nodes = [node_ for node_ in self.bip.nodes() if not isinstance(node_, str)]

        print(f'\nGraph density subject nodes: {round(bipartite.density(self.bip, subject_nodes), 4)}')
        print(f'\nGraph density question nodes: {round(bipartite.density(self.bip, question_nodes), 4)}')

    def visualize_degree_ahi_scatter_plot(self):
        """Here the AHI must be viewed as a continuous variables"""
        subject_degree_sorted = dict(OrderedDict(sorted(self.subject_degree.items())))
        # Dataframe of the subjects and degree
        subject_degree_sorted_df = pd.DataFrame.from_dict(subject_degree_sorted, orient='index',
                                                          columns=['subject_degree'])
        # Dataframe with subjects, degree, and AHI as continuous variables
        self.get_raw_asq()
        subj_ahi_degree_df = pd.concat([subject_degree_sorted_df, self.raw_asq.ahi], axis=1)
        subj_ahi_degree_df.rename(columns={'ahi':'ahi_continuous'}, inplace=True)
        # Add categorical AHI for the color map
        subj_ahi_degree_df = pd.concat([subj_ahi_degree_df, self.asq.ahi], axis=1)
        subj_ahi_degree_df.rename(columns={'ahi': 'ahi_ordinal'}, inplace=True)
        assert subj_ahi_degree_df.shape[0] == self.asq.shape[0]
        # Design the plot
        # Scatter plot, color embedding by AHI category. Subject degree VS AHI as continious
        # https://kanoki.org/2020/08/30/matplotlib-scatter-plot-color-by-category-in-python/
        sns.lmplot(x='subject_degree', y='ahi_continuous', data=subj_ahi_degree_df,
                   hue='ahi_ordinal', fit_reg=False)
        plt.grid()
        plt.title("AHI vs Bipartite Subjects Degree", fontsize=18)
        plt.show()

        # Subplot, ech subplot is a different AHI level
        categories = [*set(self.asq.ahi)]
        plt.figure(figsize=(15, 12))
        plt.subplots_adjust(hspace=0.5)
        plt.suptitle("AHI vs Bipartite Subjects Degree", fontsize=18, y=0.95)
        for n, cat_ in enumerate(categories):
            # add a new subplot iteratively
            ax = plt.subplot(2, 2, n + 1)
            sns.scatterplot(x='subject_degree', y='ahi_continuous',
                                 data=subj_ahi_degree_df[subj_ahi_degree_df['ahi_ordinal'] == cat_],ax=ax )
            # chart formatting
            ax.set_title(f'AHI category: {cat_}')
            ax.set_ylim(0, max(self.raw_asq.ahi))
            ax.grid()
            # ax.get_legend().remove()
            # ax.set_xlabel("")
        plt.show()

    def circular_plot(self):
        """Plot a circular_layout of the bipartite graph"""
        if not hasattr(self, 'bip'):
            self.bip = self.get_graph()
        pos = nx.circular_layout(self.bip)
        nx.draw_networkx(self.bip, pos=pos)
        plt.show()


# %% main to data reader
if __name__ == '__main__':
    asq_graph = ASQ_Graph(config=config)
    graph = asq_graph.get_graph()
    asq_bip_matrix = asq_graph.get_adjacency_matrix()
    asq_graph.visualize_degree_ahi_scatter_plot()


