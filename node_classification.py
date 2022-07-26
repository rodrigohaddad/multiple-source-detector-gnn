import os
import pickle

from sklearn import svm
import torch

from constants import EMBEDDING_DIR, INFECTED_DIR
from utils.save_to_pickle import save_to_pickle


def main():
    conj_emb = torch.Tensor()
    sources = torch.Tensor()
    for filename in os.listdir(EMBEDDING_DIR):
        file = os.path.join(EMBEDDING_DIR, filename)
        if not os.path.isfile(file):
            continue

        emb = pickle.load(open(file, 'rb'))
        inf_model = pickle.load(open(f'{INFECTED_DIR}/{filename.split("-")[0]}-infected.pickle', 'rb'))

        emb = torch.column_stack((emb, torch.Tensor(list(inf_model.model.status.values()))))

        sources = torch.concat((sources, torch.Tensor(list(inf_model.model.initial_status.values()))))
        conj_emb = torch.concat((conj_emb, emb))

    # x_train, x_test, y_train, y_test = train_test_split(conj_emb, sources, test_size=0, random_state=2)
    clf = svm.SVC()
    clf.fit(conj_emb, sources)
    save_to_pickle(clf, 'model', 'node_classifier')


if __name__ == '__main__':
    main()
