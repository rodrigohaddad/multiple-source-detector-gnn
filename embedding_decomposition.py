from sklearn.decomposition import PCA
import os
import torch

from utils.constants import EMBEDDING_DIR
from utils.test_model import concatenate_sources


def main():
    conj_emb = torch.Tensor()
    sources = torch.Tensor()

    for filename in os.listdir(EMBEDDING_DIR):
        file = os.path.join(EMBEDDING_DIR, filename)
        if not os.path.isfile(file):
            continue

        conj_emb, sources = concatenate_sources(file, filename, sources, conj_emb)

    pca = PCA(n_components=2)
    pca.fit(conj_emb)
    data_pca = pca.transform(conj_emb)
    # Colocar aqui o print dos dados


if __name__ == '__main__':
    main()
