import networkx as nx
import numpy as np
import statistics


class GraphTransform:
    def __init__(self, g_inf, k):
        self.G = g_inf.G
        self.model = g_inf.model
        self.k = k

        self.shortest_paths = nx.shortest_path_length(self.G)
        self.G_transf = self.transform_graph()

    def transform_graph(self):
        for u, v_dict in self.shortest_paths:
            for v, distance in v_dict.items():
                pass

