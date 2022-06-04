import networkx as nx
import numpy as np


class GraphTransform:
    def __init__(self, g_inf, k):
        self.G = g_inf.G
        self.model = g_inf.model
        self.k = k
        self.nodes_alphas = dict()

        self.shortest_paths = nx.shortest_path_length(self.G)

    def calculate_ring_infection(self):
        for u, v_dict in self.shortest_paths:
            alpha_numerator = [0] * self.k
            alpha_denominator = [0] * self.k
            for v, distance in v_dict.items():
                alpha_numerator[distance] += self.model.status[v]
                alpha_denominator[distance] += 1
            self.nodes_alphas[u] = [n/d for n, d in zip(alpha_numerator, alpha_denominator)]

    def calculate_neighbourhood_infection(self):
        pass


