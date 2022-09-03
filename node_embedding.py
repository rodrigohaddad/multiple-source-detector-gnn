import os
import pickle

from utils.constants import TRANSFORMED_DIR, MODEL_GRAPH_DIR, GLOBAL_MODEL_FILE, NOT_TRANSFORMED_DIR
from utils.save_to_pickle import save_to_pickle
from utils.test_model import test_embedding


def main():
    for filename in os.listdir(NOT_TRANSFORMED_DIR):
        file = os.path.join(NOT_TRANSFORMED_DIR, filename)
        if not os.path.isfile(file):
            continue

        # Data
        data = pickle.load(open(file, 'rb'))

        model = pickle.load(open(f'{MODEL_GRAPH_DIR}{GLOBAL_MODEL_FILE}', 'rb'))
        embedding = test_embedding(model, data, True)

        # print(f'\nGraphSAGE test accuracy: {acc*100:.2f}%\n')

        save_to_pickle(embedding, 'embedding', f'{filename.split("-")[0]}-embedding')


if __name__ == '__main__':
    main()
