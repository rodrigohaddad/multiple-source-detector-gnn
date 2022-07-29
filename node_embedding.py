import os
import pickle

from utils.save_to_pickle import save_to_pickle
from utils.test_model import test_embedding

MODEL = 'data/model'
GRAPH_TRANSFORMED = 'data/graph_transformed'


def main():
    for filename in os.listdir(GRAPH_TRANSFORMED):
        file = os.path.join(GRAPH_TRANSFORMED, filename)
        if not os.path.isfile(file):
            continue

        pyg_graph = pickle.load(open(file, 'rb'))
        data = pyg_graph.pyG

        model = pickle.load(open('data/model/sagemodel.pickle', 'rb'))
        embedding = test_embedding(model, data)

        # print(f'\nGraphSAGE test accuracy: {acc*100:.2f}%\n')

        save_to_pickle(embedding, 'embedding', f'{filename.split("-")[0]}-embedding')


if __name__ == '__main__':
    main()
