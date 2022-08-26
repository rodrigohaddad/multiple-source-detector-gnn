import pickle
import networkx as nx

from torch_geometric.utils import from_networkx

from utils.constants import DEVICE


def save_to_pickle(obj, dire, name):
    pickle.dump(obj, open(f"data/{dire}/{name}.pickle", "wb"))


def save_to_edge_list(obj, dire, name):
    nx.write_weighted_edgelist(obj, f"data/{dire}/{name}.edgelist")


def save_to_gml(obj, dire, name):
    nx.write_gml(obj, f"data/{dire}/{name}.gml")


def read_as_pyg_data(g):
    return from_networkx(G=g,
                         # group_node_attrs=['source'],
                         group_node_attrs=['infected', 'propagation_score', 'eta', 'alpha']
                         # group_edge_attrs=['weight']
                         ).to(device=DEVICE)

