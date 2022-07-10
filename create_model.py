import os
import pickle
import torch
from torch_geometric.loader import NeighborSampler

from gnn_embedding.gnn_3 import SAGE
from utils.save_to_pickle import save_to_pickle
from utils.test_model import test

TRANSFORMED_GRAPH = 'data/graph_transformed'
EMBEDDING = 'data/embedding'
DEVICE = torch.device('cpu' if torch.cuda.is_available() else 'cpu')


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

        train_loader = NeighborSampler(data.edge_index, sizes=[20, 15, 10], batch_size=256,
                                       shuffle=True, num_nodes=data.num_nodes)

        model = SAGE(in_channels=data.num_node_features,
                     hidden_channels=10,
                     num_layers=3,
                     train_loader=train_loader)

        model = model.to(DEVICE)

        model.fit(data, DEVICE, 200)

        save_to_pickle(model, 'model', 'sagemodel')

        # print(f'\nGraphSAGE test accuracy: {test(model, data) * 100:.2f}%\n')


if __name__ == '__main__':
    main()
