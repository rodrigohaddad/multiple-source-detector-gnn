import os
import pickle

from utils.constants import INFECTED_DIR
from graph_transformation.transformation import GraphTransform


def main():
    for filename in os.listdir(INFECTED_DIR):
        file = os.path.join(INFECTED_DIR, filename)
        g_inf = pickle.load(open(file, 'rb'))
        GraphTransform(g_inf=g_inf,
                       k=2,
                       cut_type='percentile',
                       alpha_weight=.5)


if __name__ == '__main__':
    main()
