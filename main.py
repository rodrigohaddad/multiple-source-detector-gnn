from graph_transformation.graph_creation import GraphFactory
from graph_transformation.provision_graph import InfectedGraphProvision
from graph_transformation.transformation import GraphTransform
import networkx as nx

PARAMS = [{'model': 'SIR',
           'params': {'beta': 0.001,
                      'gamma': 0.01,
                      'fraction_infected': 0.05},
           'n_iter': 200},
          {'model': 'SI',
           'params': {'beta': 0.001,
                      'Infected': [0, 1, 2, 3]},
           'n_iter': 200},
          ]


def main():
    G = GraphFactory('networks/powergrid.txt')
    g_inf = InfectedGraphProvision(**{'graph': G, **PARAMS[1]})

    g_transformed = GraphTransform(g_inf)


if __name__ == '__main__':
    main()
