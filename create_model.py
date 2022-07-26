import os
import pickle
import torch
from torch_geometric.loader import NeighborSampler

from constants import TRANSFORMED_DIR, EMBEDDING_DIR
from gnn_embedding.gnn import SAGE
from utils.save_to_pickle import save_to_pickle


DEVICE = torch.device('cpu' if torch.cuda.is_available() else 'cpu')


def main():
    try:
        model = pickle.load(open('data/model/sagemodel.pickle', 'rb'))
    except:
        model = SAGE(in_channels=9,  # data.num_node_features
                     hidden_channels=64,
                     num_layers=3)
    model = model.to(DEVICE)

    for filename in os.listdir(TRANSFORMED_DIR):
        file = os.path.join(TRANSFORMED_DIR, filename)
        if not os.path.isfile(file):
            continue

        ana_dir = os.path.join(EMBEDDING_DIR)
        if not os.path.exists(ana_dir):
            os.mkdir(ana_dir)

        # Data
        pyg_graph = pickle.load(open(file, 'rb'))
        data = pyg_graph.pyG

        train_loader = NeighborSampler(data.edge_index, sizes=[20, 15, 10], batch_size=256,
                                       shuffle=True, num_nodes=data.num_nodes)

        model.fit(data, DEVICE, 50, train_loader)

    save_to_pickle(model, 'model', 'sagemodel')


if __name__ == '__main__':
    main()
