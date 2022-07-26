import networkx as nx


class GraphFactory:
    def __init__(self,
                 file_path='',
                 graph_type='',
                 artificial_graph_config={},
                 **kwargs):
        self.G = self._create_graph(file_path, graph_type, artificial_graph_config)

    @staticmethod
    def _create_random_graph(graph_type: str, artificial_graph_config):
        graphs = {'er': nx.erdos_renyi_graph,
                  'ba': nx.barabasi_albert_graph}
        return graphs[graph_type](**artificial_graph_config)

    def _create_graph(self, file_dir: str, graph_type: str, artificial_graph_config: dict):
        if not len(file_dir):
            return self._create_random_graph(graph_type, artificial_graph_config)
        return nx.read_edgelist(file_dir,
                                create_using=nx.Graph(),
                                nodetype=int)
