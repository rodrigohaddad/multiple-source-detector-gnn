import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc

from utils.save_to_pickle import save_to_pickle

MODELS = {'SI': ep.SIModel,
          'SIR': ep.SIRModel}


class InfectedGraphProvision:
    def __init__(self,
                 graph,
                 infection_config,
                 ):
        self.G = graph
        self.infection_config = infection_config

        self.model = MODELS[infection_config.model](self.G)
        self.config = mc.Configuration()
        self.trends = None

        self._add_model_params(infection_config.params)
        self._infect_graph(infection_config.n_iter)

        save_to_pickle(self, 'infected_graph',
                       f'{self.infection_config.name}-infected')

    def _add_model_params(self, params):
        for param_name, param_value in params.items():
            self.config.add_model_parameter(param_name, param_value)
        self.model.set_initial_status(self.config)

    def _infect_graph(self, n_iter):
        iterations = self.model.iteration_bunch(n_iter)
        self.trends = self.model.build_trends(iterations)