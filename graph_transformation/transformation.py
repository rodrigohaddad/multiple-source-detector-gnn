import networkx as nx
from dataclasses import dataclass
from torch_geometric.utils import from_networkx
import torch

from create_model import DEVICE
from graph_transformation.node_metrics import NodeMetrics
from utils.save_to_pickle import save_to_pickle

DEVICE = torch.device('cpu' if torch.cuda.is_available() else 'cpu')


@dataclass
class GGraph:
    def __init__(self, g):
        self.G = g
        self.pyG = from_networkx(G=g,
                                 # group_node_attrs=['source'],
                                 group_node_attrs=['infected'],
                                 # group_edge_attrs=['weight']
                                 ).to(device=DEVICE)


class GraphTransform:
    nodes_metrics = dict()
    G_new = nx.Graph()

    def __init__(self, g_inf, k, min_weight, alpha_weight):
        self.G = g_inf.G
        self.model = g_inf.model
        self.k = k + 1
        self.min_weight = min_weight
        self.alpha_weight = alpha_weight

        self.shortest_paths = nx.shortest_path_length(self.G)

        self._calculate_infection()
        self._create_new_graph(self._calculate_nodes_weights())
        print(f'C. Components: {nx.number_connected_components(self.G_new)}')
        save_to_pickle(GGraph(self.G_new), 'graph_transformed',
                       f'{g_inf.infection_config.name}-transformed')

    def _calculate_neighbourhood_infection(self, v):
        n_inf = 0
        n_neighbors = 0
        for neighbor in self.G.neighbors(v):
            n_inf += self.model.status[neighbor]
            n_neighbors += 1
        return len(self.G[v]) and n_inf / n_neighbors or 0

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
                weight = self._calculate_weight(f_uv, g_uv)
                if weight >= self.min_weight:
                    # Add the lowest connection if node u is alone
                    # Do not connect if nodes were neighbors previously
                    # Stop infecting when reaching desired infection percentage
                    # Add edges to edge_weight on graphsage (see github question)
                    weights.append((u, v, {'edge_weight': weight, 'weight': weight}))
        return weights

    def _create_new_graph(self, weights):
        self.G_new.add_edges_from(weights)
        nx.set_node_attributes(self.G_new, self.model.status, name='infected')
        # nx.set_node_attributes(self.G_new, self.model.initial_status, name='y')
