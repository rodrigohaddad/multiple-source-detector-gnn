import os
import pickle

from torch_geometric.utils import accuracy, f1_score
from sklearn.metrics import f1_score as sk_f1_score

from utils.constants import MODEL_GRAPH_DIR, GRAPH_SUP_UNTRANS_FILE, NOT_TRANSFORMED_DIR
from utils.test_model import test_pred


def main():
    # Load embedding model
    sage = pickle.load(open(f'{MODEL_GRAPH_DIR}{GRAPH_SUP_UNTRANS_FILE}', 'rb'))

    # Load test graphs
    for filename in os.listdir(f"{NOT_TRANSFORMED_DIR}/test"):
        file = os.path.join(f"{NOT_TRANSFORMED_DIR}/test", filename)
        if not os.path.isfile(file):
            continue

        # Data
        data = pickle.load(open(file, 'rb'))
        data.y = data.x[:, -1].long()
        data.x = data.x[:, :-1].float()

        # Apply embedding model
        y_pred = test_pred(sage, data)

        # Eval
        acc = accuracy(data.y, y_pred)
        f_score = f1_score(y_pred, data.y, 2)
        sk_f = sk_f1_score(data.y, y_pred, average='micro')

        print(f'Acc: {acc}, F_score: {f_score}, {sk_f}')


if __name__ == '__main__':
    main()
