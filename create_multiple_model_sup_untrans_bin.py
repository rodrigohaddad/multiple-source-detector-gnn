import os
import pickle

from torch_geometric.loader import NeighborLoader

from gnn_embedding.gnn_sup_untrans_bin import SUSAGEBin
from utils.constants import DEVICE, NOT_TRANSFORMED_DIR
from utils.save_to_pickle import save_to_pickle


def train():
    for dirname in os.listdir(NOT_TRANSFORMED_DIR):
        model = SUSAGEBin(dim_in=9,  # data.num_node_features
                          dim_h=128,  # 64
                          dim_out=1,
                          n_layers=3,
                          aggr='max')
        model = model.to(DEVICE)

        path = os.path.join(NOT_TRANSFORMED_DIR, dirname)
        path_train = f'{path}/train'
        path_val = f'{path}/val'
        for file_train, file_val in zip(os.listdir(path_train), os.listdir(path_val)):
            # Data train
            file_path_train = os.path.join(path_train, file_train)
            data = pickle.load(open(file_path_train, 'rb'))
            data.y = data.x[:, -1].float()
            data.x = data.x[:, :-1].float()

            # Data val
            file_path_val = os.path.join(path_val, file_val)
            data_val = pickle.load(open(file_path_val, 'rb'))
            data_val.y = data_val.x[:, -1].float()
            data_val.x = data_val.x[:, :-1].float()

            # testar com directed=False
            train_loader = NeighborLoader(
                data,
                # num_neighbors=[-1, 10, 10],
                num_neighbors=[-1, -1, -1],
                batch_size=len(data.x),
                directed=False,
            )

            model.fit(data, 500, train_loader, data_val)

        save_to_pickle(model, 'model/graph', f'graph-sage-{dirname}')


if __name__ == '__main__':
    train()
