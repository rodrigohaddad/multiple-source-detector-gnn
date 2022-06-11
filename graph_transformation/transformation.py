import networkx as nx

from graph_transformation.node_metrics import NodeMetrics
from utils.save_to_pickle import save_to_pickle


class GraphTransform:
    nodes_metrics = dict()
    G_new = nx.Graph()

    def __init__(self, g_inf, k, alpha_weight):
        self.G = g_inf.G
        self.model = g_inf.model
        self.k = k + 1
        self.alpha_weight = alpha_weight

        self.shortest_paths = nx.shortest_path_length(self.G)

        self._calculate_infection()
        self._create_new_graph(self._calculate_nodes_weights())
        save_to_pickle(self.G_new, 'graph_transformed',
                       f'{g_inf.name}-transformed')

    def _calculate_neighbourhood_infection(self, v):
        n_inf = 0
        for neighbor in self.G.neighbors(v):
            n_inf += self.model.status[neighbor]
        return len(self.G[v]) and n_inf / len(self.G[v]) or 0

    def _calculate_infection(self):
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

            self.nodes_metrics[u] = NodeMetrics(neighbors_infection,
                                                alpha_numerator,
                                                alpha_denominator)

    def _calculate_weight(self, ring_index, neighbor_index):
        return self.alpha_weight * ring_index + (1 - self.alpha_weight) * neighbor_index

    def _calculate_sum_of_difference_infections(self, u_metrics, v_metrics):
        f_uv, g_uv = 0, 0
        for u_eta, u_alpha, v_eta, v_alpha in (u_metrics.eta, u_metrics.alpha, v_metrics.eta, v_metrics.alpha):
            f_uv += abs(u_alpha - v_alpha)
            g_uv += abs(u_eta - v_eta)
        return 1 - f_uv / self.k, 1 - g_uv / self.k

    def _calculate_nodes_weights(self):
        weights = []
        for u, u_metrics in self.nodes_metrics.items():
            for v, v_metrics in self.nodes_metrics.items():
                if u == v:
                    continue
                f_uv, g_uv = self._calculate_sum_of_difference_infections(u_metrics,
                                                                          v_metrics)
                weights.append((u, v, {'weight': self._calculate_weight(f_uv, g_uv)}))
        return weights

    def _create_new_graph(self, weights):
        self.G_new.add_edges_from(weights)
        nx.set_node_attributes(self.G_new, self.model.status, name='infected')
