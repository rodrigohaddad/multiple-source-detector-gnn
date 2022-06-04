import networkx as nx


class GraphFactory:
    def __init__(self,
                 file_dir='',
                 ):
        self.G = self._create_graph(file_dir)

    @staticmethod
    def _create_random_graph():
        return nx.erdos_renyi_graph(1000, 0.1)

    def _create_graph(self, file_dir):
        if not len(file_dir):
            return self._create_random_graph()
        return nx.read_edgelist(file_dir,
                                create_using=nx.Graph(),
                                nodetype=int)

