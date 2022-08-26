import os
import pickle

from gnn_embedding.gnn_weightless_vertex import SAGEWeightless
from utils.constants import DEVICE, MODEL_DIR, MODEL_WEIGHTLESS_FILE, NOT_TRANSFORMED_DIR
from gnn_embedding.sampler import PosNegSampler
from utils.save_to_pickle import save_to_pickle


def main():
    try:
        model = pickle.load(open(f'{MODEL_DIR}{MODEL_WEIGHTLESS_FILE}', 'rb'))
    except:
        model = SAGEWeightless(in_channels=8,  # data.num_node_features
                               hidden_channels=128,  # 512
                               num_layers=4)
    model = model.to(DEVICE)

    for filename in os.listdir(NOT_TRANSFORMED_DIR):
        file = os.path.join(NOT_TRANSFORMED_DIR, filename)
        if not os.path.isfile(file):
            continue

        # Data
        data = pickle.load(open(file, 'rb'))

        train_loader = PosNegSampler(edge_index=data.edge_index,
                                     sizes=[-1, 30, 30, 30],
                                     batch_size=100,
                                     shuffle=True,
                                     num_nodes=data.num_nodes)

        model.fit(data, DEVICE, 100, train_loader)

    save_to_pickle(model, 'model', 'sagemodel_weightless')


if __name__ == '__main__':
    main()
