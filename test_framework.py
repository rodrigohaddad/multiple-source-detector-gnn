import os
import pickle

from sklearn.metrics import accuracy_score

from constants import MODEL_DIR, TRANSFORMED_DIR
from utils.test_model import test


def main():
    # Load embedding model
    sage = pickle.load(open(f'{MODEL_DIR}/sagemodel.pickle', 'rb'))

    # Load classifier model
    clf = pickle.load(open(f'{MODEL_DIR}/node_classifier.pickle', 'rb'))

    # Load test graphs
    pyg_graph = pickle.load(open(f'{TRANSFORMED_DIR}/test', 'rb'))
    data = pyg_graph.pyG

    # Apply embedding model
    embedding = test(sage, data)

    # Apply classifier model
    pred = clf.predict(x_test)
    print(f'Acc: {accuracy_score(pred, y_test)}')


if __name__ == '__main__':
    main()
