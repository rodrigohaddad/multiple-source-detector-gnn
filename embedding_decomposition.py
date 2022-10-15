from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import torch
import umap

from utils.constants import EMBEDDING_DIR
from utils.test_model import concatenate_sources

import matplotlib.pyplot as plt
import seaborn as sns


def main():
    conj_emb = torch.Tensor()
    sources = torch.empty(0, dtype=torch.short)

    for filename in os.listdir(f'{EMBEDDING_DIR}/test'):
        file = os.path.join(f'{EMBEDDING_DIR}/test', filename)
        if not os.path.isfile(file):
            continue

        conj_emb, sources, infections = concatenate_sources(file, filename, sources, conj_emb)

    # data = PCA(n_components=2).fit_transform(conj_emb)
    data = umap.UMAP().fit_transform(conj_emb)
    # data = TSNE(n_components=2, learning_rate='auto',
    #             init='random').fit_transform(conj_emb)

    # palette = {0: 'C0', 1: 'C1'}
    plt.figure(figsize=(10, 10))
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')

    sns.scatterplot(x=data.T[0], y=data.T[1], hue=infections)
    plt.savefig("data/figures/embd_sage_infections.png", dpi=120)

    sns.scatterplot(x=data.T[0], y=data.T[1], hue=sources)
    plt.savefig("data/figures/embd_sage_sources.png", dpi=120)


if __name__ == '__main__':
    main()
