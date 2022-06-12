from utils.load_graph_config import load_config
from infected_graph_factory.graph_creation import GraphFactory
from infected_graph_factory.provision_graph import InfectedGraphProvision
from graph_transformation.transformation import GraphTransform


def main():
    # configs = load_config()
    configs = [load_config()[1]]
    for inf_config in configs:
        g_factory = GraphFactory(inf_config.file_path)

        inf_config.set_params({'beta': 0.001,
                               'Infected': g_factory.select_random_sources(),
                               'fraction_infected': 0.15})

        g_inf = InfectedGraphProvision(graph=g_factory.G,
                                       infection_config=inf_config)

        g_transformed = GraphTransform(g_inf=g_inf,
                                       k=3,
                                       min_weight=0.3,
                                       alpha_weight=.5)


if __name__ == '__main__':
    main()
