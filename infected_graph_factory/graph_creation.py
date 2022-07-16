import networkx as nx
import random


class GraphFactory:
    def __init__(self,
                 file_path='',
                 graph_type='',
                 **kwargs):
        self.G = self._create_graph(file_path, graph_type)

    @staticmethod
    def _create_random_graph(graph_type: str):
        graphs = {'er': nx.erdos_renyi_graph(1000, 0.1),
                  'ba': nx.barabasi_albert_graph(1000, 100)}
        return graphs[graph_type]

    def _create_graph(self, file_dir: str, graph_type: str):
        if not len(file_dir):
            return self._create_random_graph(graph_type)
        return nx.read_edgelist(file_dir,
                                create_using=nx.Graph(),
                                nodetype=int)

    def select_random_sources(self):
        return random.sample(list(self.G.nodes()), 15)
