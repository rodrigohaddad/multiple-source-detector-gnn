import os
import pickle

from torch_geometric.loader import NeighborLoader

from gnn_embedding.gnn_test import GraphSAGE
from utils.constants import DEVICE, MODEL_DIR, TEST_MODEL_SUPERVISED_FILE, LABELED_DIR
from utils.save_to_pickle import save_to_pickle


def main():
    try:
        model = pickle.load(open(f'{MODEL_DIR}{TEST_MODEL_SUPERVISED_FILE}', 'rb'))
    except:
        model = GraphSAGE(dim_in=8,  # data.num_node_features
                          dim_h=512,  # 64
                          dim_out=2,
                          n_layers=3)
    model = model.to(DEVICE)

    for filename in os.listdir(LABELED_DIR):
        file = os.path.join(LABELED_DIR, filename)
        if not os.path.isfile(file):
            continue

        # Data
        data = pickle.load(open(file, 'rb'))
        data.y = data.x[:, -1].long()
        data.x = data.x[:, :-1].float()
        train_loader = NeighborLoader(
            data,
            num_neighbors=[-1, 30, 30, 30],
            batch_size=100,
        )

        model.fit(data, 200, train_loader)

    save_to_pickle(model, 'model', 'test_sagemodel_supervised')


if __name__ == '__main__':
    main()
