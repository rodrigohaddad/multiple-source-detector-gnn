import os
import pickle

from torch_geometric.loader import NeighborLoader

from gnn_embedding.gnn_sup_untrans_bin import SUSAGEBin
from utils.constants import DEVICE, MODEL_GRAPH_DIR, GRAPH_SUP_UNTRANS_FILE, NOT_TRANSFORMED_DIR
from utils.save_to_pickle import save_to_pickle


def main():
    try:
        model = pickle.load(open(f'{MODEL_GRAPH_DIR}{GRAPH_SUP_UNTRANS_FILE}', 'rb'))
    except:
        model = SUSAGEBin(dim_in=7,  # data.num_node_features
                          dim_h=128,  # 64
                          dim_out=1,
                          n_layers=3,
                          aggr='max')
    model = model.to(DEVICE)

    for filename in os.listdir(NOT_TRANSFORMED_DIR):
        file = os.path.join(NOT_TRANSFORMED_DIR, filename)
        if not os.path.isfile(file):
            continue

        # Data
        data = pickle.load(open(file, 'rb'))
        data.y = data.x[:, -1].float()
        data.x = data.x[:, :-1].float()

        train_loader = NeighborLoader(
            data,
            num_neighbors=[-1, 10, 10],
            batch_size=100,
        )

        model.fit(data, 300, train_loader)

    save_to_pickle(model, 'model', 'sage_model_sup_untrans_pooling')


if __name__ == '__main__':
    main()
