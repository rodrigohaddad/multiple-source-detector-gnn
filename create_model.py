import os
import pickle

from utils.constants import TRANSFORMED_DIR, DEVICE, MODEL_GRAPH_DIR, MODEL_FILE
from gnn_embedding.gnn import SAGE
from gnn_embedding.sampler import PosNegSampler
from utils.save_to_pickle import save_to_pickle


def main():
    try:
        model = pickle.load(open(f'{MODEL_GRAPH_DIR}{MODEL_FILE}', 'rb'))
    except:
        model = SAGE(in_channels=8,  # data.num_node_features
                     hidden_channels=128,  # 512
                     num_layers=4)
    model = model.to(DEVICE)

    for filename in os.listdir(TRANSFORMED_DIR):
        file = os.path.join(TRANSFORMED_DIR, filename)
        if not os.path.isfile(file):
            continue

        # Data
        data = pickle.load(open(file, 'rb'))

        train_loader = PosNegSampler(edge_index=data.edge_index,
                                     sizes=[-1, 30, 30, 30],
                                     batch_size=100,
                                     shuffle=True,
                                     num_nodes=data.num_nodes)

        # data.n_id = torch.arange(data.num_nodes)
        # train_loader = NeighborLoader(data, num_neighbors=[20, 15, 10], batch_size=64,
        #                               shuffle=True)

        model.fit(data, DEVICE, 100, train_loader)

    save_to_pickle(model, 'model', 'sagemodel')


if __name__ == '__main__':
    main()
