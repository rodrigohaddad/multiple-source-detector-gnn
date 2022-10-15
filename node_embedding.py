import os
import pickle

from utils.constants import MODEL_GRAPH_DIR, NOT_TRANSFORMED_DIR, \
    GRAPH_UNSUP_UNTRANS_FILE, GRAPH_UNSUP_TRANS_FILE, TRANSFORMED_DIR, MODEL_FILE
from utils.save_to_pickle import save_to_pickle
from utils.test_model import test_embedding, test_pred


def main():
    for filename in os.listdir(TRANSFORMED_DIR):
        file = os.path.join(TRANSFORMED_DIR, filename)
        if not os.path.isfile(file):
            continue

        # Data
        data = pickle.load(open(file, 'rb'))
        data.x = data.x[:, :-1].float()

        model = pickle.load(open(f'{MODEL_GRAPH_DIR}{MODEL_FILE}', 'rb'))
        embedding = test_embedding(model, data, False)
        # embedding = test_pred(model, data)

        # print(f'\nGraphSAGE test accuracy: {acc*100:.2f}%\n')

        save_to_pickle(embedding, 'embedding', f'{filename.split("-")[0]}-embedding')


if __name__ == '__main__':
    main()
