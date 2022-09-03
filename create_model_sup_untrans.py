import os
import pickle

from torch_geometric.loader import NeighborLoader

from gnn_embedding.gnn_sup_untrans import SUSAGE
from utils.constants import DEVICE, MODEL_GRAPH_DIR, GRAPH_SUP_UNTRANS_FILE, NOT_TRANSFORMED_DIR
from utils.save_to_pickle import save_to_pickle


def main():
    try:
        model = pickle.load(open(f'{MODEL_GRAPH_DIR}{GRAPH_SUP_UNTRANS_FILE}', 'rb'))
    except:
        model = SUSAGE(dim_in=8,  # data.num_node_features
                       dim_h=512,  # 64
                       dim_out=2,
                       n_layers=3)
    model = model.to(DEVICE)

    for filename in os.listdir(NOT_TRANSFORMED_DIR):
        file = os.path.join(NOT_TRANSFORMED_DIR, filename)
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

    save_to_pickle(model, 'model', 'sage_model_sup_untrans')


if __name__ == '__main__':
    main()
