import itertools
from typing import Any

import networkx as nx
import concurrent.futures
import torch
import numpy as np

from dataclasses import dataclass
from torch_geometric.utils import from_networkx
from create_model import DEVICE
from graph_transformation.node_metrics import NodeMetrics
from utils.save_to_pickle import save_to_pickle
from datetime import datetime


DEVICE = torch.device('cpu' if torch.cuda.is_available() else 'cpu')


@dataclass
class GGraph:
    def __init__(self, g):
        self.G = g
        self.pyG = from_networkx(G=g,
                                 # group_node_attrs=['source'],
                                 group_node_attrs=['infected', 'eta', 'alpha']
                                 # group_edge_attrs=['weight']
                                 ).to(device=DEVICE)


class WeightConfig:
    def __init__(self, u, v, weight):
        self.u = u
        self.v = v
        self.weight = weight

    def __call__(self, *args, **kwargs):
        return self.u, self.v, {'edge_weight': self.weight, 'weight': self.weight}


class GraphTransform:
    eta_dict = dict()
    alpha_dict = dict()
    G_new = nx.Graph()
    threads = 5

    def __init__(self, g_inf, k: int, min_weight: float, alpha_weight: float):
        self.G = g_inf.G
        self.model = g_inf.model
        self.k = k + 1
        self.min_weight = min_weight
        self.alpha_weight = alpha_weight

        self.multithreading_bfs(self._split_nodes())
        self._create_new_graph(self._calculate_nodes_weights())

        print(f'C. Components: {nx.number_connected_components(self.G_new)}')
        save_to_pickle(GGraph(self.G_new), 'graph_transformed',
                       f'{g_inf.graph_config.name}-transformed')

    def _calculate_neighbourhood_infection(self, v: int) -> float:
        n_inf = 0
        n_neighbors = 0
        for neighbor in self.G.neighbors(v):
            n_inf += self.model.status[neighbor]
            n_neighbors += 1
        return len(self.G[v]) and n_inf / n_neighbors or 0

    def multithreading_bfs(self, nodes_split):
        print(datetime.now())
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for nodes in nodes_split:
                futures.append(executor.submit(self.bfs, nodes=nodes))
            for future in concurrent.futures.as_completed(futures):
                self.eta_dict = {**self.eta_dict, **future.result()[0]}
                self.alpha_dict = {**self.eta_dict, **future.result()[1]}
        print(datetime.now())

    def _split_nodes(self):
        return np.array_split(self.G.nodes, self.threads)

    def bfs(self, nodes):
        eta_dict = dict()
        alpha_dict = dict()
        for node in nodes:
            alpha_numerator = [0] * self.k
            alpha_denominator = [0] * self.k
            neighbors_infection = [0] * self.k

            distance = 0
            visited = [False] * self.G.size()
            queue = []

            visited[node] = True
            queue.append((node, distance))
            while queue:
                s, distance = queue.pop(0)
                if distance == self.k:
                    break
                # print(s, end=" ")
                for neighbour in self.G.neighbors(s):
                    if not visited[neighbour]:
                        visited[neighbour] = True
                        queue.append((neighbour, distance + 1))

                        alpha_numerator[distance] += self.model.status[neighbour]
                        alpha_denominator[distance] += 1
                        n_inf = self._calculate_neighbourhood_infection(neighbour)
                        neighbors_infection[distance] += n_inf

            nm = NodeMetrics(neighbors_infection,
                             alpha_numerator,
                             alpha_denominator)

            eta_dict[node] = nm.eta
            alpha_dict[node] = nm.alpha
        return eta_dict, alpha_dict

    def _calculate_weight(self, ring_index: float, neighbor_index: float) -> float:
        return self.alpha_weight * ring_index + (1 - self.alpha_weight) * neighbor_index

    def _calculate_sum_of_difference_infections(self, u, v) -> tuple[float, float]:
        f_uv, g_uv = 0, 0
        for u_eta, u_alpha, v_eta, v_alpha in zip(
                self.eta_dict[u], self.alpha_dict[u], self.eta_dict[v], self.alpha_dict[v]):
            f_uv += abs(u_alpha - v_alpha)
            g_uv += abs(u_eta - v_eta)
        return 1 - f_uv / self.k, 1 - g_uv / self.k

    def _calculate_nodes_weights(self) -> list[tuple[Any, Any, dict[str, float]]]:
        weights = []
        self.all_weights = dict()
        nodes_address = list(itertools.combinations(self.eta_dict.keys(), 2))
        for u, v in nodes_address:
            f_uv, g_uv = self._calculate_sum_of_difference_infections(u, v)
            weight = self._calculate_weight(f_uv, g_uv)

            self.all_weights[u] = {**self.all_weights.get(u, {}), **{v: weight}}

            if weight >= self.min_weight and not self.G[u].get(v):
                weights.append((u, v, {'edge_weight': weight, 'weight': weight}))

        return weights

    def _create_new_graph(self, weights: list[tuple[Any, Any, dict[str, float]]]):
        self.G_new.add_edges_from(weights)
        nx.set_node_attributes(self.G_new, self.model.status, name='infected')
        nx.set_node_attributes(self.G_new, self.model.initial_status, name='source')
        nx.set_node_attributes(self.G_new, self.eta_dict, name='eta')
        nx.set_node_attributes(self.G_new, self.alpha_dict, name='alpha')
