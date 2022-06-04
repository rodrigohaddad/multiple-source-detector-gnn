import networkx as nx
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc

MODELS = {'SI': ep.SIModel,
          'SIR': ep.SIRModel}


class InfectedGraphProvision:
    def __init__(self,
                 model,
                 params,
                 n_iter,
                 graph,
                 ):
        self.G = graph

        self.model = MODELS[model](self.G)
        self.config = mc.Configuration()
        self.trends = None

        self._add_model_params(params)
        self._infect_graph(n_iter)

    def _add_model_params(self, params):
        for param_name, param_value in params.items():
            self.config.add_model_parameter(param_name, param_value)
        self.model.set_initial_status(self.config)

    def _infect_graph(self, n_iter):
        iterations = self.model.iteration_bunch(n_iter)
        self.trends = self.model.build_trends(iterations)
