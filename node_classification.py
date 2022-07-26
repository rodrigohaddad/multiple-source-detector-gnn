import os
from sklearn import svm
import torch

from constants import EMBEDDING_DIR
from utils.save_to_pickle import save_to_pickle
from utils.test_model import concatenate_sources


def main():
    conj_emb = torch.Tensor()
    sources = torch.Tensor()
    for filename in os.listdir(EMBEDDING_DIR):
        file = os.path.join(EMBEDDING_DIR, filename)
        if not os.path.isfile(file):
            continue

        conj_emb, sources = concatenate_sources(file, filename, sources, conj_emb)

    # x_train, x_test, y_train, y_test = train_test_split(conj_emb, sources, test_size=0, random_state=2)
    clf = svm.SVC()
    clf.fit(conj_emb, sources)
    save_to_pickle(clf, 'model', 'node_classifier')


if __name__ == '__main__':
    main()
