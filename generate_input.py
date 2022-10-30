import os

from utils.constants import INFECTED_DIR
from utils.load_graph_config import load_config
from infected_graph_factory.graph_creation import GraphFactory
from infected_graph_factory.provision_graph import InfectedGraphProvision


def main():
    configs = load_config()
    # configs = [load_config()[1]]
    for graph_config in configs:
        for idx in range(graph_config.n_graphs):
            g_factory = GraphFactory(**graph_config.__dict__)

            # existent_graph = graph_config.name in [i.split('-')[0] for i in os.listdir(INFECTED_DIR)]
            # if not graph_config.overwrite_previous and existent_graph:
            #     continue

            InfectedGraphProvision(idx=idx,
                                   graph=g_factory.G,
                                   graph_config=graph_config)


if __name__ == '__main__':
    main()
