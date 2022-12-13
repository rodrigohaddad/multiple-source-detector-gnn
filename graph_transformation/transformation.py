import copy
import itertools
from typing import Any

import networkx as nx
import concurrent.futures
import numpy as np

from graph_transformation.node_metrics import NodeMetrics
from utils.propagation_score import calculate_propagation_score
from utils.save_to_pickle import save_to_pickle, read_as_pyg_data
from datetime import datetime


class GraphTransform:
    threads = 5

    def __init__(self, g_inf, k: int, percentile: int, alpha_weight: float, keep_old: bool, file_name: str, step: str):
        self.eta_dict = dict()
        self.alpha_dict = dict()
        self.keep_old = keep_old

        self.G_new = nx.Graph()
        if keep_old:
            self.G_new = copy.deepcopy(g_inf.G)

        self.G = g_inf.G
        self.model = g_inf.model
        self.k = k
        self.percentile = percentile
        self.alpha_weight = alpha_weight

        # self.propagation_score = calculate_propagation_score(g_inf)
        self.multithreading_bfs(self._split_nodes())
        self._create_new_graph([] if keep_old else self._calculate_nodes_weights())

        # print(f'Infected graph diameter: {nx.diameter(self.G)}')
        # print(f'Transformed graph diameter: {nx.diameter(self.G_new)}')

        print(f'Infected graph n nodes: {self.G.number_of_nodes()}')
        print(f'Transformed graph n nodes: {self.G_new.number_of_nodes()}')

        pyg_data = read_as_pyg_data(self.G_new)

        graph_config = g_inf.graph_config
        infection_config = graph_config.infection_config

        print(f'N sources: {infection_config.n_sources}')
        print(f'Max infection fraction: {infection_config.max_infected_fraction}')

        incomplete_path = f"graph_enriched/{graph_config.graph_type}_{int(100 * infection_config.max_infected_fraction)}inf_{infection_config.n_sources}s"
        graph_path_to_be_saved = f"{incomplete_path}/{step}"

        save_to_pickle(pyg_data,
                       graph_path_to_be_saved,
                       f"{file_name.split('.')[0]}-enriched")
        del pyg_data

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
                self.alpha_dict = {**self.alpha_dict, **future.result()[1]}
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

    def _calculate_cut(self, total_weight):
        return np.percentile(total_weight, self.percentile)

    def _calculate_nodes_weights(self) -> list[tuple[Any, Any, float]]:
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

        cut = self._calculate_cut(total_weight)

        cut_weights = [(u, v, w) for u, v, w in weights if w >= cut]

        print(f"After calculating weights {datetime.now()}")
        print(f"Weight: {cut} +- {np.std(total_weight)}, min {min(total_weight)} max {max(total_weight)}")

        return cut_weights

    def _create_new_graph(self, weights: list[tuple[Any, Any, Any]]):
        if not self.keep_old:
            self.G_new.add_weighted_edges_from(weights, 'edge_weight')
        # nx.set_node_attributes(self.G_new, self.propagation_score, name='propagation_score')
        nx.set_node_attributes(self.G_new, self.model.status, name='infected')
        nx.set_node_attributes(self.G_new, self.model.initial_status, name='source')
        nx.set_node_attributes(self.G_new, self.eta_dict, name='eta')
        nx.set_node_attributes(self.G_new, self.alpha_dict, name='alpha')
