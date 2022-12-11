from utils.load_graph_config import load_config
from infected_graph_factory.graph_creation import GraphFactory
from infected_graph_factory.provision_graph import InfectedGraphProvision


def generate():
    configs = load_config()
    # configs = [load_config()[1]]
    for graph_config in configs:
        for n_sources in graph_config.infection_config.n_sources:
            for max_infected_fraction in graph_config.infection_config.max_infected_fraction:
                for idx in range(graph_config.n_graphs):
                    g_factory = GraphFactory(**graph_config.__dict__)

                    InfectedGraphProvision(idx=idx,
                                           graph=g_factory.G,
                                           graph_config=graph_config,
                                           n_sources=n_sources,
                                           max_infected_fraction=max_infected_fraction)


if __name__ == '__main__':
    generate()
