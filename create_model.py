import os
import pickle
import torch

from gnn_embedding.gnn import GraphSAGE
from gnn_embedding.gnn_2 import GraphSAGE2
from utils.save_to_pickle import save_to_pickle
from utils.test_model import test

TRANSFORMED_GRAPH = 'data/graph_transformed'
EMBEDDING = 'data/embedding'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    for filename in os.listdir(TRANSFORMED_GRAPH):
        file = os.path.join(TRANSFORMED_GRAPH, filename)
        if not os.path.isfile(file):
            continue

        ana_dir = os.path.join(EMBEDDING)
        if not os.path.exists(ana_dir):
            os.mkdir(ana_dir)
        # Data
        pyg_graph = pickle.load(open(file, 'rb'))
        data = pyg_graph.pyG

        model = GraphSAGE2(dim_in=data.num_features,
                           dim_h=64,
                           dim_out=int(data.x.max() + 1))
        model.fit(data, 200)

        save_to_pickle(model, 'model', 'sagemodel')

        print(f'\nGraphSAGE test accuracy: {test(model, data)*100:.2f}%\n')


if __name__ == '__main__':
    main()
