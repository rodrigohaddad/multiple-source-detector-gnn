import os
import pickle

from sklearn import svm
from sklearn.model_selection import train_test_split
import torch

EMBEDDING_DIR = 'data/embedding'
INFECTION_DIR = 'data/infected_graph'


def main():
    conj_emb = torch.Tensor()
    sources = torch.Tensor()
    for filename in os.listdir(EMBEDDING_DIR):
        file = os.path.join(EMBEDDING_DIR, filename)
        if not os.path.isfile(file):
            continue

        emb = pickle.load(open(file, 'rb'))
        inf_model = pickle.load(open(f'{INFECTION_DIR}/{filename.split("-")[0]}-infected.pickle', 'rb'))

        emb = torch.column_stack((emb, torch.Tensor(list(inf_model.model.initial_status.values()))))

        sources = torch.concat((sources, torch.Tensor(list(inf_model.model.status.values()))))
        conj_emb = torch.concat((conj_emb, emb))
        print("")

    x_train, x_test, y_train, y_test = train_test_split(conj_emb, sources, test_size=0.33, random_state=2)

    clf = svm.SVC()
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    print(pred)


if __name__ == '__main__':
    main()
