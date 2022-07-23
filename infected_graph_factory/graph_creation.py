import networkx as nx
import random


class GraphFactory:
    def __init__(self,
                 n_nodes=0,
                 file_path='',
                 graph_type='',
                 **kwargs):
        self.G = self._create_graph(file_path, graph_type, n_nodes)

    @staticmethod
    def _create_random_graph(graph_type: str, n_nodes):
        graphs = {'er': nx.erdos_renyi_graph(n_nodes, 0.1),
                  'ba': nx.barabasi_albert_graph(n_nodes, 50)}
        return graphs[graph_type]

    def _create_graph(self, file_dir: str, graph_type: str, n_nodes: int):
        if not len(file_dir):
            return self._create_random_graph(graph_type, n_nodes)
        return nx.read_edgelist(file_dir,
                                create_using=nx.Graph(),
                                nodetype=int)

    def select_random_sources(self):
        return random.sample(list(self.G.nodes()), 15)
