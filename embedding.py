import os
import pickle
from torch_geometric.utils.convert import from_networkx

TRANSFORMED_GRAPH = 'data/infected_graph'
EMBEDDING = 'data/embedding'


def main():
    for filename in os.listdir(TRANSFORMED_GRAPH):
        file = os.path.join(TRANSFORMED_GRAPH, filename)
        if not os.path.isfile(file):
            continue

        ana_dir = os.path.join(EMBEDDING)
        if not os.path.exists(ana_dir):
            os.mkdir(ana_dir)

        network_obj = pickle.load(open(file, 'rb'))
        pyg_graph = from_networkx(network_obj)


if __name__ == '__main__':
    main()
