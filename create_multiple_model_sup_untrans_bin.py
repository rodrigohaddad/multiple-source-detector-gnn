import os
import pickle

from torch_geometric.loader import NeighborLoader

from gnn_embedding.gnn_sup_untrans_bin import SUSAGEBin
from utils.constants import DEVICE, NOT_TRANSFORMED_DIR
from utils.save_to_pickle import save_to_pickle


def main():
    for filename in os.listdir(NOT_TRANSFORMED_DIR):
        if 'powergrid' not in filename:
            continue
        model = SUSAGEBin(dim_in=9,  # data.num_node_features
                          dim_h=128,  # 64
                          dim_out=1,
                          n_layers=3,
                          aggr='max')
        model = model.to(DEVICE)

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
                # num_neighbors=[-1, 10, 10],
                num_neighbors=[-1, -1, -1],
                batch_size=len(data.x),
                directed=False,
            )

            model.fit(data, 500, train_loader)

        save_to_pickle(model, 'model', f'graph-sage-{filename}')


if __name__ == '__main__':
    main()
