import os
import pickle

import torch
from sklearn.metrics import accuracy_score

from utils.constants import MODEL_DIR, TRANSFORMED_DIR
from utils.test_model import test_embedding, concatenate_sources


def main():
    # Load embedding model
    sage = pickle.load(open(f'{MODEL_DIR}/sagemodel.pickle', 'rb'))

    # Load classifier model
    clf = pickle.load(open(f'{MODEL_DIR}/node_classifier.pickle', 'rb'))

    # Load test graphs
    conj_emb = torch.Tensor()
    sources = torch.Tensor()
    for filename in os.listdir(f"{TRANSFORMED_DIR}/test"):
        file = os.path.join(f"{TRANSFORMED_DIR}/test", filename)
        if not os.path.isfile(file):
            continue

        # Data
        data = pickle.load(open(file, 'rb'))

        # Apply embedding model
        embedding = test_embedding(sage, data)

        # Apply classifier model
        conj_emb, sources = concatenate_sources(file, filename, sources, conj_emb, embedding)
        pred = clf.predict(conj_emb)
        print(f'Acc: {accuracy_score(pred, sources)}')


if __name__ == '__main__':
    main()
