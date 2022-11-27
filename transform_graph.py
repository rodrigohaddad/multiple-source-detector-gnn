import os
import pickle

from utils.constants import INFECTED_DIR
from graph_transformation.transformation import GraphTransform


def main():
    for g_dir in os.listdir(INFECTED_DIR):
        if g_dir[-3:] in ['10s', '15s', '20s']:
            path = os.path.join(INFECTED_DIR, g_dir)
            for filename in os.listdir(path):
                file = os.path.join(path, filename)
                g_inf = pickle.load(open(file, 'rb'))
                GraphTransform(g_inf=g_inf,
                               k=4,
                               percentile=50,
                               alpha_weight=.5,
                               keep_old=True,
                               file_name=filename)


if __name__ == '__main__':
    main()
