from sklearn.decomposition import PCA
import os
import torch
import umap

from utils.constants import EMBEDDING_DIR
from utils.test_model import concatenate_sources

import matplotlib.pyplot as plt
import seaborn as sns


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

    data_umap = umap.UMAP().fit_transform(conj_emb)

    palette = {0: 'C0', 1: 'C1'}

    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=data_umap.T[0], y=data_umap.T[1], hue=sources, palette=palette)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.savefig("data/figures/embd_sage.png", dpi=120)


if __name__ == '__main__':
    main()
