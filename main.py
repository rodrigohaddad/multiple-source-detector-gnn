from graph_transformation.graph_creation import GraphFactory
from graph_transformation.provision_graph import InfectedGraphProvision, InfectionConfig
from graph_transformation.transformation import GraphTransform
import random


INF_CONFIGS = [InfectionConfig(model='SIR',
                               n_iter=200,
                               params={'beta': 0.001,
                                       'gamma': 0.01,
                                       'fraction_infected': 0.05}),
               InfectionConfig(model='SI',
                               n_iter=200,
                               ),
               ]


def main():
    G = GraphFactory('networks/powergrid.edgelist.txt').G
    sources = random.sample(list(G.nodes()), 15)
    INF_CONFIGS[1].set_params({'beta': 0.001,
                               'Infected': sources,
                               'fraction_infected': 0.15})
    g_inf = InfectedGraphProvision(graph=G,
                                   infection_config=INF_CONFIGS[1])

    g_transformed = GraphTransform(g_inf=g_inf,
                                   k=3,
                                   alpha_weight=.5)


if __name__ == '__main__':
    main()
