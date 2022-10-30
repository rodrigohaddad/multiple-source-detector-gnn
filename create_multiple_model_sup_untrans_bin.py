import os
import pickle

from torch_geometric.loader import NeighborLoader

from gnn_embedding.gnn_sup_untrans_bin import SUSAGEBin
from utils.constants import DEVICE, MODEL_GRAPH_DIR, NOT_TRANSFORMED_DIR, \
    GRAPH_SUP_UNTRANS_BIN_FILE, GRAPH_SUP_UNTRANS_BIN_FULL_2_LAYERS_FILE
from utils.save_to_pickle import save_to_pickle


def main():
    try:
        model = pickle.load(open(f'{MODEL_GRAPH_DIR}{GRAPH_SUP_UNTRANS_BIN_FULL_2_LAYERS_FILE}', 'rb'))
    except:
        model = SUSAGEBin(dim_in=9,  # data.num_node_features
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

        # testar com directed=False
        train_loader = NeighborLoader(
            data,
            # num_neighbors=[-1, 10, 10], # todos de todos (duas camadas)
            num_neighbors=[-1, -1, -1],
            batch_size=1500,
            directed=False,
        )

        model.fit(data, 500, train_loader)

    save_to_pickle(model, 'model', GRAPH_SUP_UNTRANS_BIN_FULL_2_LAYERS_FILE.split('.')[0])


if __name__ == '__main__':
    main()
