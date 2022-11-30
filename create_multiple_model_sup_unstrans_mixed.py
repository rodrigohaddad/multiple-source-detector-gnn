import os
import pickle

from torch_geometric.loader import NeighborLoader

from gnn_embedding.gnn_sup_untrans_bin import SUSAGEBin
from utils.constants import DEVICE, NOT_TRANSFORMED_DIR
from utils.save_to_pickle import save_to_pickle


GRAPH_LIST = ['er_15inf_3s', 'er_15inf_10s', 'er_15inf_15s', 'er_15inf_20s']


def main():
    model = SUSAGEBin(dim_in=9,  # data.num_node_features
                      dim_h=128,  # 64
                      dim_out=1,
                      n_layers=3,
                      aggr='max')
    model = model.to(DEVICE)

    for filename in GRAPH_LIST:
        path = os.path.join(NOT_TRANSFORMED_DIR, filename)
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if not os.path.isfile(file_path):
                continue

            # Data
            data = pickle.load(open(file_path, 'rb'))
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

    save_to_pickle(model, 'model', f'graph-sage-er_15inf_3-10-15-20s')


if __name__ == '__main__':
    main()
