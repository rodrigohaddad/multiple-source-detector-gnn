import os
import pickle
import numpy as np

from utils.constants import INFECTED_DIR
from graph_transformation.transformation import GraphTransform


def transform():
    for g_dir in os.listdir(INFECTED_DIR):
        if '5000' not in g_dir:
            continue
        path = os.path.join(INFECTED_DIR, g_dir)
        dirs = np.array_split(os.listdir(path), 3)
        for step, directory in zip(['train', 'val', 'test'], dirs):
            for filename in directory:
                file = os.path.join(path, filename)
                g_inf = pickle.load(open(file, 'rb'))
                GraphTransform(g_inf=g_inf,
                               k=4,
                               percentile=50,
                               alpha_weight=.5,
                               keep_old=True,
                               file_name=filename,
                               step=step)


if __name__ == '__main__':
    transform()
