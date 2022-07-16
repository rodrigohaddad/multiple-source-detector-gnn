import os

from utils.load_graph_config import load_config
from infected_graph_factory.graph_creation import GraphFactory
from infected_graph_factory.provision_graph import InfectedGraphProvision


def main():
    configs = load_config()
    # configs = [load_config()[1]]
    for inf_config in configs:
        g_factory = GraphFactory(inf_config.file_path)

        inf_config.set_params({'beta': 0.001,
                               'Infected': g_factory.select_random_sources(),
                               'fraction_infected': 0.05})

        existent_graph = inf_config.name in [i.split('-')[0] for i in os.listdir('data/infected_graph')]
        if not inf_config.overwrite_previous and existent_graph:
            continue

        InfectedGraphProvision(graph=g_factory.G,
                               infection_config=inf_config)


if __name__ == '__main__':
    main()
