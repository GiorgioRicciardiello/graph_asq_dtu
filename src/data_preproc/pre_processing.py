import pathlib

from config.config import config
from typing import Optional, Union, Dict, Tuple, List, Any, Type
from pathlib import Path
from src.data_reader.data_reader import DataReader
import pandas as pd
from tqdm import tqdm
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
from collections import Counter

class PreProcessASQ(DataReader):
    def __init__(self, config: dict):
        super().__init__(config)
        self.config = config
        self.pre_proces_path = self.config['root_dir'].joinpath(pathlib.Path(r'data/processed'))
        self.pre_proces_path_asq = self.pre_proces_path / 'asq_preprocessed.xlsx'
        self.pre_proces_path_nan_idx = self.pre_proces_path / 'nan_index_df.xlsx'
        self.pre_proces_path_graph = self.pre_proces_path.parents[0] /'graph'/ 'bipartite_asq.graphml'

        if self.pre_proces_path_asq.is_file() and self.pre_proces_path_nan_idx.is_file():
            print(f'\nPreprocessed ASQ and nan index matrix already exist and are not modified')
            print(f'\nASQ path: {self.pre_proces_path_asq}')
            print(f'\nNan index matrix path: {self.pre_proces_path_nan_idx}')
        else:
            print(f'\nApplying preprocess')
            # Get the raw ASQ
            DataReader.__init__(self, config=self.config)
            self.get_raw_asq()
            # preprocessing methods
            self.raw_asq[self.config['target']] = self._ahi_class(x=self.raw_asq[self.config['target']]).values
            self._float_to_int()
            self._negatives_to_nans()
            self._save_nan_idx(rewrite=True)
            self._nans_to_zeros_int()
            self._save_preprocessed_asq()

        # if self.pre_proces_path_graph.is_file():
        #     print(f'Graph already found in {self.pre_proces_path_graph}')
        # else:
        #     print(f'\nGenerating graph')
        #     self._generate_bipartite_graph()
        #     self._save_graph()
        print(f'\nPreprocessed ASQ is ready to be called from the class')

    # %% pre processing functions
    @staticmethod
    def _ahi_class(x) -> pd.DataFrame:
        """The AHI is given as a float index. We will work with a four-class problem"""
        ahi_range = {'Normal': 1, 'Mild': 2, 'Moderate': 3, 'Severe': 4}

        def threshold_ahi(ahi: float, ahi_range: dict):
            """Apply the threshold to the AHI column. Implement the functions as df['ahi'].apply(...) """
            if ahi < 5:
                return ahi_range['Normal']
            if 5 <= ahi < 15:
                return ahi_range['Mild']
            if 15 <= ahi < 30:
                return ahi_range['Moderate']
            if ahi >= 30:
                return ahi_range['Severe']

        if not x.name == 'ahi':
            raise ValueError(f'Please give the correct target for the class problem conversion')

        x = x.apply(func=threshold_ahi, ahi_range=ahi_range).astype(int)

        assert sorted([*ahi_range.values()]) == sorted(set(x))
        return x

    def _float_to_int(self):
        """
        Poisson's distribution only accepts integers. We must parse what we can as integer, while preserving some logic
        in the feature. Otherwise, is removed. We have ~36 float predictors
        """
        # float_col = [*self.raw_asq.select_dtypes(include=['float64']).columns]
        for float_col_ in self.raw_asq.select_dtypes(include=['float64']):
            if float_col_ == 'dem_0800':
                # BMI set to ordinal
                def bmi_to_ordinal(bmi: float):
                    bmi_range = {'underweight': 0, 'overweight': 1, 'obese': 2}
                    """Apply the threhsold to the BMI column. Implement the functions as df['ahi'].apply(...) """
                    if bmi < 25:
                        return bmi_range['underweight']
                    if 25 <= bmi < 30:
                        return bmi_range['underweight']
                    if bmi >= 30:
                        return bmi_range['obese']

                self.raw_asq[float_col_] = self.raw_asq[float_col_].apply(func=bmi_to_ordinal).astype(int)

            if float_col_ == 'dem_high_meters':
                # convert meters to cm and round and parse as int
                self.raw_asq[float_col_] = self.raw_asq[float_col_] * 100
                self.raw_asq[float_col_] = self.raw_asq[float_col_].astype(int)

            if float_col_ == 'fosq_1100':
                # round nearest decimal and parse as int
                self.raw_asq[float_col_].round(0).astype(int)

            if 'cos' in float_col_ or 'sin' in float_col_:
                # Time conversions to sin and cos are removed
                self.raw_asq.drop(labels=float_col_, axis=1, inplace=True)

            else:
                # remove the dataset
                self.raw_asq.drop(labels=float_col_, axis=1, inplace=True)

    def _negatives_to_nans(self):
        """Poisson's distribution can not have negative values, so we decode them as nans"""
        predictors = [col_ for col_ in [*self.raw_asq.columns] if col_ != 'ahi']
        predictors = [col_ for col_ in predictors if col_ != 'subject']
        self.raw_asq.replace(-4, 0, inplace=True)  # Participant responded "Do Not Know"
        self.raw_asq.replace(-4, 0, inplace=True)  # Participant did not respond to question
        self.raw_asq[self.raw_asq[predictors] < 0] = np.nan

    def _nans_to_zeros_int(self):
        """Once the nan index matrix has been created we can parse the nans=0 and convert all to integers"""
        if self.pre_proces_path_nan_idx.is_file():
            self.raw_asq.fillna(0, inplace=True)
            for float_col_ in self.raw_asq.select_dtypes(include=['float64']):
                self.raw_asq[float_col_] = self.raw_asq[float_col_].round(0).astype(int)
            assert self.raw_asq.isnull().values.any() == False

        else:
            print(f'Unable to parse nan to zeros + int dtype. Because nan index matrix was not found')

    # %% Save variables
    def _save_nan_idx(self, rewrite: Optional[bool] = False):
        """
        Create a pandas containing ones where nan values are present in the pre-processed ASQ
        :return:
        """
        predictors = [col_ for col_ in [*self.raw_asq.columns] if col_ != 'ahi']
        predictors = [col_ for col_ in predictors if col_ != 'subject']
        # Get 1's for all non null values
        nan_index_df = self.raw_asq[predictors].notnull().astype('int')
        # invert so 1's are the nans and 0 where we find real values
        nan_index_df = nan_index_df.apply(lambda x: 1 - x)
        # concatenate with non predictors
        nan_index_df = pd.concat([self.raw_asq[self.config['nonpredictors']], nan_index_df], axis=1)
        # save
        if rewrite:
            print(f'\nSaving nan index matrix in {self.pre_proces_path_nan_idx}')
            nan_index_df.to_excel(self.pre_proces_path_nan_idx)
        else:
            print(f'\nnan index matrix not modified, change parameter rewrite if you wish to do so')
        print(f'\nNumber nan values: {nan_index_df[predictors].values.sum()}')

    def _save_preprocessed_asq(self):
        """Save the preprocesssed ASQ"""
        print(f'\nSaving pre processed asq in {self.pre_proces_path_asq}')
        self.raw_asq.to_excel(self.pre_proces_path_asq)
        self.asq = self.raw_asq
        # columns_dict = {idx : col_ for idx, col_ in enumerate([*self.raw_asq.columns])}
        # columns_dict = pd.DataFrame(columns_dict)
        # columns_dict.to_csv()
        del self.raw_asq

    def _save_graph(self):
        """
        Save the graph using nx documentation
        https://networkx.org/documentation/latest/reference/readwrite/generated/networkx.readwrite.graphml.write_graphml.html
        """
        nx.write_graphml_lxml(G=self.bip, path=self.pre_proces_path_graph )
        if self.pre_proces_path_graph.is_file():
            print(f'\nGraph saved in { self.pre_proces_path_graph }')

    # %% getters
    def get_nan_idx_frame(self)->pd.DataFrame:
        """Get the binary matrix containing the nans positions"""
        return pd.read_excel(self.pre_proces_path_nan_idx)

    def get_preprocessed_asq(self):
        """Get the preprocesssed ASQ"""
        print(f'\n Returning preprocessed')
        self.asq = pd.read_excel(self.pre_proces_path_asq, index_col=0)


    def get_graph(self)->nx.read_graphml:
        """Get bipartite graph from preprocessed ASQ"""
        graph = nx.read_graphml(path=self.pre_proces_path_graph)
        nx.info(graph)
        return graph

    # %% Generate bipartite
    def _generate_bipartite_graph(self):
        """
        Bipartite graphs B = (U, V, E) have two node sets U,V and edges in E that only connect nodes from opposite sets
        :return:
        """
        DISPLAY_GRAPH = True
        asq = self.get_preprocessed_asq()
        predictors = [col_ for col_ in [*self.raw_asq.columns] if col_ != 'subject']
        # predictors = [col_ for col_ in predictors if col_ != 'subject']
        # select only the predictors
        asq_r = asq.loc[0:10, predictors[0:100]]
        # asq_r = asq[predictors]

        # Generate the bipartite graph structure
        bip = nx.Graph()
        # Two columns of noes U=subjects, V=questions
        bip.add_nodes_from(asq_r.index, bipartite='subjects')
        bip.add_nodes_from(asq_r.columns, bipartite='questions')
        # bip.nodes(data=True)
        # dummy graph to check
        # bip.add_edges_from([(0, 'bthbts_0300'),
        #                     (0, 'cir_0100'),
        #                     (2, 'dem_0110'),
        #                     ])
        # bipartite.is_bipartite(bip)
        # edges = bip.edges()
        # nx.draw_networkx(
        #     bip,
        #     pos=nx.drawing.layout.bipartite_layout(G=bip, nodes=[*asq_r.index]),
        #     width=2)
        # print(edges)
        # plt.show()

        # Assign the weighted edges between the nodes U and V but not across
        s = asq_r.stack()
        weighted_edges = [(s_[0][0],s_[0][1], s_[1]) for s_ in s.iteritems()]  # ((node_u, node_v, weight))
        bip.add_weighted_edges_from(weighted_edges)
        # visualize
        print(bip.edges(data=True))
        # verify is bipartite and weighted
        if nx.is_connected(bip) and bipartite.is_bipartite(bip):
            print(f'\nBipartite graph has been connected')
        else:
            raise ValueError(f'\nBipartite graph was NOT been connected')
        if nx.is_weighted(bip, edge=None, weight="weight"):
            print(f'\nBipartite graph is weighted')
        else:
            raise ValueError(f'\nBipartite graph is NOT weighted')
        # visualize
        if DISPLAY_GRAPH:
            nx.draw_networkx(
                bip,
                pos=nx.drawing.layout.bipartite_layout(G=bip, nodes=[*asq_r.index]),
                width=2)
            edges = bip.edges()
            print(f'\nedges: {edges}')
            plt.show()

        self.bip = bip

    def _plot_graph(self):
        # https://www.youtube.com/watch?v=PouhDHfssYA&ab_channel=SepinoudAzimi
        # https://coderzcolumn.com/tutorials/data-science/network-analysis-in-python-important-structures-and-bipartite-graphs-networkx#5
        # https://ericmjl.github.io/Network-Analysis-Made-Simple/04-advanced/01-bipartite/
        if not hasattr(self, 'bip'):
            self.bip = self.get_graph()
        nx.draw_networkx(self.bip, pos=nx.drawing.layout.bipartite_layout(self.bip, ['0', '1', '2', '3']), width=2)
        plt.show()

        pos = nx.circular_layout(self.bip)
        nx.draw_networkx(self.bip, pos=pos)
        plt.show()

        pass

    def _degrees_graph(self):
        # person_nodes = self.bip.nodes()
        subject_nodes = [node_ for node_ in self.bip.nodes() if node_.isnumeric()]

        deg_question, deg_subjects = bipartite.degrees(B=self.bip, nodes=subject_nodes)
        dict(deg_question)
        dict(deg_subjects)
        print(f'\nCount of different degrees for the subjects nodes {Counter(dict(deg_subjects).values())}')

    def graph_density(self):
        """
        Graph density represents the ratio between the edges present in a graph and the maximum number of edges that
        the graph can contain. Conceptually, it provides an idea of how dense a graph is in terms of edge connectivity.
        :return:
        """
        bottom_nodes, top_nodes = bipartite.sets(self.bip)  # sujects, questions
        print(f'\nGraph density bottom_nodes: {round(bipartite.density(self.bip, bottom_nodes), 2)}')
        print(f'\nGraph density top_nodes: {round(bipartite.density(self.bip, top_nodes), 2)}')

# %% main to data reader
if __name__ == '__main__':
    reader = PreProcessASQ(config=config)
    asq_pre = reader.get_preprocessed_asq()
    raw_asq = reader.raw_asq()
    nan_index_df = reader.get_nan_idx_frame()
