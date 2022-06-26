import os
import pickle
import torch
import networkx as nx
from torch_geometric.data import Data

from gnn_embedding.gnn import GraphSAGE

TRANSFORMED_GRAPH = 'data/graph_transformed'
EMBEDDING = 'data/embedding'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    for filename in os.listdir(TRANSFORMED_GRAPH):
        file = os.path.join(TRANSFORMED_GRAPH, filename)
        if not os.path.isfile(file):
            continue

        ana_dir = os.path.join(EMBEDDING)
        if not os.path.exists(ana_dir):
            os.mkdir(ana_dir)
        # Data
        pyg_graph = pickle.load(open(file, 'rb'))
        data = pyg_graph.pyG
        # data.edge_weight
        model = GraphSAGE(in_channels=data.num_features,
                          out_channels=int(data.y.max() + 1),
                          num_layers=3).to(DEVICE)
        model.fit(data, 200)


if __name__ == '__main__':
    main()
