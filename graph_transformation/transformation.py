import itertools
from typing import Any

import networkx as nx
import concurrent.futures
import numpy as np

from graph_transformation.node_metrics import NodeMetrics
from utils.save_to_pickle import save_to_pickle, read_as_pyg_data
from datetime import datetime


class GraphTransform:
    eta_dict = dict()
    alpha_dict = dict()
    G_new = nx.Graph()
    threads = 5

    def __init__(self, g_inf, k: int, cut_type: str, alpha_weight: float):
        self.G = g_inf.G
        self.model = g_inf.model
        self.k = k + 1
        self.cut_type = cut_type
        self.alpha_weight = alpha_weight

        self.multithreading_bfs(self._split_nodes())
        self._create_new_graph(self._calculate_nodes_weights())

        # print(f'Infected graph diameter: {nx.diameter(self.G)}')
        # print(f'Transformed graph diameter: {nx.diameter(self.G_new)}')

        print(f'Infected graph n nodes: {self.G.number_of_nodes()}')
        print(f'Transformed graph n nodes: {self.G_new.number_of_nodes()}')

        save_to_pickle(read_as_pyg_data(self.G_new), 'graph_transformed',
                       f'{g_inf.graph_config.name}-transformed')

        print('')

    def _calculate_neighbourhood_infection(self, v: int) -> float:
        n_inf = 0
        n_neighbors = 0
        for neighbor in self.G.neighbors(v):
            n_inf += self.model.status[neighbor]
            n_neighbors += 1
        return len(self.G[v]) and n_inf / n_neighbors or 0

    def multithreading_bfs(self, nodes_split):
        print(f"Before calculating metrics {datetime.now()}")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for nodes in nodes_split:
                futures.append(executor.submit(self.bfs, nodes=nodes))
            for future in concurrent.futures.as_completed(futures):
                self.eta_dict = {**self.eta_dict, **future.result()[0]}
                self.alpha_dict = {**self.eta_dict, **future.result()[1]}
        print(f"After calculating metrics {datetime.now()}")

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

    def _calculate_cut(self, total_weight, weights):
        # print(f"{sum(total_weight)/len(weights)}\n"
        #       f"{np.percentile(total_weight, 75)}\n")
        return {'mean': sum(total_weight)/len(weights),
                'percentile': np.percentile(total_weight, 60)}[self.cut_type]

    def _calculate_nodes_weights(self) -> list[tuple[Any, Any, float]]:
        print(f"Before calculating weights {datetime.now()}")
        weights = []
        total_weight = list()
        self.all_weights = dict()
        nodes_address = list(itertools.combinations(self.eta_dict.keys(), 2))
        for u, v in nodes_address:
            f_uv, g_uv = self._calculate_sum_of_difference_infections(u, v)
            weight = self._calculate_weight(f_uv, g_uv)
            # self.all_weights[u] = {**self.all_weights.get(u, {}), **{v: weight}}

            if self.G[u].get(v) is None:
                # weights.append((u, v, {'edge_weight': weight, 'weight': weight}))
                weights.append((u, v, weight))
                total_weight.append(weight)

        cut = self._calculate_cut(total_weight, weights)

        cut_weights = [(u, v, w) for u, v, w in weights if w >= cut]

        print(f"After calculating weights {datetime.now()}")
        print(f"Weight: {cut} +- {np.std(total_weight)}, min {min(total_weight)} max {max(total_weight)}")

        return cut_weights

    def _create_new_graph(self, weights: list[tuple[Any, Any, Any]]):
        self.G_new.add_weighted_edges_from(weights, 'edge_weight')
        nx.set_node_attributes(self.G_new, self.model.status, name='infected')
        nx.set_node_attributes(self.G_new, self.model.initial_status, name='source')
        nx.set_node_attributes(self.G_new, self.eta_dict, name='eta')
        nx.set_node_attributes(self.G_new, self.alpha_dict, name='alpha')
