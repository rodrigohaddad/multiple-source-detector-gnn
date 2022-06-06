import networkx as nx


class GraphTransform:
    nodes_alphas = dict()
    nodes_etas = dict()
    G_new = None

    def __init__(self, g_inf, k, alpha_weight):
        self.G = g_inf.G
        self.model = g_inf.model
        self.k = k + 1
        self.alpha_weight = alpha_weight

        self.shortest_paths = nx.shortest_path_length(self.G)

        self._calculate_metrics()
        self._create_new_graph()

    def _calculate_ring_infection(self):
        for u, v_dict in self.shortest_paths:
            alpha_numerator = [0] * self.k
            alpha_denominator = [0] * self.k
            neighbors_infection = [0] * self.k
            for v, distance in v_dict.items():
                if distance == 0:
                    continue
                if distance >= self.k:
                    break
                alpha_numerator[distance] += self.model.status[v]
                alpha_denominator[distance] += 1
                n_inf = self._calculate_neighbourhood_infection(v)
                neighbors_infection[distance] += n_inf

            self.nodes_etas[u] = [d and n / d or 0 for n, d in zip(neighbors_infection, alpha_denominator)]
            self.nodes_alphas[u] = [d and n / d or 0 for n, d in zip(alpha_numerator, alpha_denominator)]

    def _calculate_neighbourhood_infection(self, v):
        n_inf = 0
        for neighbor in self.G.neighbors(v):
            n_inf += self.model.status[neighbor]
        return len(self.G[v]) and n_inf / len(self.G[v]) or 0

    def _calculate_node_infection_index(self, ring_index, neighbor_index):
        return self.alpha_weight*ring_index+(1-self.alpha_weight)*neighbor_index

    def _calculate_infection_similarity(self, nodes_inf):
        pass

    def _calculate_metrics(self):
        self._calculate_ring_infection()
        self._calculate_infection_similarity(self.nodes_alphas)
        self._calculate_infection_similarity(self.nodes_etas)

    def _create_new_graph(self):
        self.G_new.add_edges_from([(1, 2, {'weight': 0.3}), (1, 3, {'weight': 0.3})])
