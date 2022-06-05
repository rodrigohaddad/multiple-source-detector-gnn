import networkx as nx


class GraphTransform:
    nodes_alphas = dict()
    nodes_etas = dict()

    def __init__(self, g_inf, k):
        self.G = g_inf.G
        self.model = g_inf.model
        self.k = k

        self.shortest_paths = nx.shortest_path_length(self.G)

        self._calculate_metrics()

    def _calculate_ring_infection(self):
        for u, v_dict in self.shortest_paths:
            alpha_numerator = [0] * self.k
            alpha_denominator = [0] * self.k
            neighbors_infection = [0] * self.k
            for v, distance in v_dict.items():
                alpha_numerator[distance] += self.model.status[v]
                alpha_denominator[distance] += 1
                n_inf = self._calculate_neighbourhood_infection(v)
                neighbors_infection[distance] += n_inf

            self.nodes_etas[u] = [n/d for n, d in zip(neighbors_infection, alpha_denominator)]
            self.nodes_alphas[u] = [n/d for n, d in zip(alpha_numerator, alpha_denominator)]

    def _calculate_neighbourhood_infection(self, v):
        n_inf = 0
        for neighbor in self.G.neighbors(v):
            n_inf += self.model.status[neighbor]
        return n_inf / len(self.G[v])

    def _calculate_metrics(self):
        self._calculate_ring_infection()

