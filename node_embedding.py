import os
import pickle

from utils.save_to_pickle import save_to_pickle
from utils.test_model import test

MODEL = 'data/model'
GRAPH_TRANSFORMED = 'data/graph_transformed'


def main():
    model_filename = os.listdir(MODEL)
    model_filename = os.path.join(MODEL, model_filename[0])
    for filename in os.listdir(GRAPH_TRANSFORMED):
        file = os.path.join(GRAPH_TRANSFORMED, filename)
        if not os.path.isfile(file):
            continue

        pyg_graph = pickle.load(open(file, 'rb'))
        data = pyg_graph.pyG

        model = pickle.load(open(model_filename, 'rb'))
        embedding = test(model, data)

        # print(f'\nGraphSAGE test accuracy: {test(model, data)*100:.2f}%\n')

        save_to_pickle(embedding, 'embedding', f'{filename.split("-")[0]}-embedding')
        print("")


if __name__ == '__main__':
    main()
