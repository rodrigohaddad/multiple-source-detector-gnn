import networkx as nx
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc


class InfectedGraphProvision:
    def __init__(self, file_dir, params, n_iter):
        self.G = self._create_graph(file_dir)

        self.model = ep.SIModel(self.G)
        self.config = mc.Configuration()
        self.trends = None

        self._add_model_params(params)
        self._infect_graph(n_iter)

    def __call__(self):
        return self.model

    @staticmethod
    def _create_graph(file_dir):
        # return nx.erdos_renyi_graph(1000, 0.1)
        return nx.read_edgelist(file_dir,
                                create_using=nx.Graph(),
                                nodetype=int)

    def _add_model_params(self, params):
        for param_name, param_value in params.items():
            self.config.add_model_parameter(param_name, param_value)
        self.model.set_initial_status(self.config)

    def _infect_graph(self, n_iter):
        iterations = self.model.iteration_bunch(n_iter)
        self.trends = self.model.build_trends(iterations)
