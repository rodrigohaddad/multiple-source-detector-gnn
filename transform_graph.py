import os
import pickle

from graph_transformation.transformation import GraphTransform

INFECTED_PATH = 'data/infected_graph'


def main():
    for filename in os.listdir(INFECTED_PATH):
        file = os.path.join(INFECTED_PATH, filename)
        g_inf = pickle.load(open(file, 'rb'))
        GraphTransform(g_inf=g_inf,
                       k=3,
                       min_weight=0.45,
                       alpha_weight=.5)


if __name__ == '__main__':
    main()
