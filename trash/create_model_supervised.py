import os
import pickle

from torch_geometric.loader import NeighborSampler

from trash.gnn_supervised import SAGESupervised
from utils.constants import DEVICE, MODEL_GRAPH_DIR, MODEL_SUPERVISED_FILE, NOT_TRANSFORMED_DIR
from utils.save_to_pickle import save_to_pickle


def main():
    try:
        model = pickle.load(open(f'{MODEL_GRAPH_DIR}{MODEL_SUPERVISED_FILE}', 'rb'))
    except:
        model = SAGESupervised(in_channels=8,  # data.num_node_features
                               hidden_channels=2,  # 512
                               num_layers=4)
    model = model.to(DEVICE)

    for filename in os.listdir(NOT_TRANSFORMED_DIR):
        file = os.path.join(NOT_TRANSFORMED_DIR, filename)
        if not os.path.isfile(file):
            continue

        # Data
        data = pickle.load(open(file, 'rb'))
        data.y = data.x[:, -1].long()
        data.x = data.x[:, :-1]
        train_loader = NeighborSampler(edge_index=data.edge_index,
                                       sizes=[-1, 30, 30, 30],
                                       batch_size=100,
                                       shuffle=True,
                                       num_nodes=data.num_nodes)

        model.fit(data, DEVICE, 1000, train_loader)

    save_to_pickle(model, 'model', 'sagemodel_supervised')


if __name__ == '__main__':
    main()
