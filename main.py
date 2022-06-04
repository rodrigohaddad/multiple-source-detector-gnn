from graph_transformation.provision_graph import InfectedGraphProvision
from graph_transformation.transformation import GraphTransform

PARAMS = {'file_dir': 'networks/powergrid.txt',
          'params': {'beta': 0.001,
                     'gamma': 0.01,
                     'fraction_infected': 0.05},
          'n_iter': 200}


def main():
    g_inf = InfectedGraphProvision(**PARAMS)

    g_transformed = GraphTransform(g_inf)


if __name__ == '__main__':
    main()
