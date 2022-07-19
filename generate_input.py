import os

from utils.load_graph_config import load_config
from infected_graph_factory.graph_creation import GraphFactory
from infected_graph_factory.provision_graph import InfectedGraphProvision


def main():
    configs = load_config()
    # configs = [load_config()[1]]
    for infection_config in configs:
        g_factory = GraphFactory(**infection_config.__dict__)

        infection_config.set_params({'Infected': g_factory.select_random_sources(),
                                     'fraction_infected': 0.05})

        existent_graph = infection_config.name in [i.split('-')[0] for i in os.listdir('data/infected_graph')]
        if not infection_config.overwrite_previous and existent_graph:
            continue

        InfectedGraphProvision(graph=g_factory.G,
                               infection_config=infection_config)


if __name__ == '__main__':
    main()
