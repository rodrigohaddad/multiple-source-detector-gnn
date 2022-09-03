import os
import pickle

from gnn_embedding.gnn_unsup_untrans import UUSAGE
from utils.constants import TRANSFORMED_DIR, DEVICE, MODEL_GRAPH_DIR, GRAPH_UNSUP_UNTRANS_FILE
from gnn_embedding.sampler import PosNegSampler
from utils.save_to_pickle import save_to_pickle


def main():
    try:
        model = pickle.load(open(f'{MODEL_GRAPH_DIR}{GRAPH_UNSUP_UNTRANS_FILE}', 'rb'))
    except:
        model = UUSAGE(dim_in=8,
                       dim_h=128,
                       dim_out=128,
                       n_layers=3)
    model = model.to(DEVICE)

    for filename in os.listdir(TRANSFORMED_DIR):
        file = os.path.join(TRANSFORMED_DIR, filename)
        if not os.path.isfile(file):
            continue

        # Data
        data = pickle.load(open(file, 'rb'))
        data.x = data.x[:, :-1].float()

        train_loader = PosNegSampler(edge_index=data.edge_index,
                                     sizes=[30, 30, 30],
                                     batch_size=30,
                                     shuffle=True,
                                     num_nodes=data.num_nodes)

        model.fit(data, DEVICE, 100, train_loader)

    save_to_pickle(model, 'model', 'sage_model_unsup_untrans')


if __name__ == '__main__':
    main()
