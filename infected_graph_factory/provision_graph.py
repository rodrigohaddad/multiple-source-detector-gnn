import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
import random

from utils.save_to_pickle import save_to_pickle

MODELS = {'SI': ep.SIModel,
          'SIR': ep.SIRModel}


class InfectedGraphProvision:
    trends = None

    def __init__(self,
                 graph,
                 graph_config):
        self.G = graph
        self.graph_config = graph_config
        infection_config = graph_config.infection_config

        self.model = MODELS[infection_config.model](self.G)
        self.config = mc.Configuration()

        sources = self._select_random_sources(infection_config.n_sources)

        self._add_model_params({**infection_config.params,
                               **{'Infected': sources}})
        self._infect_graph(infection_config.max_infected_fraction)

        save_to_pickle(self, 'infected_graph',
                       f'{graph_config.name}-infected')

    def _add_model_params(self, params):
        for param_name, param_value in params.items():
            self.config.add_model_parameter(param_name, param_value)
            if param_name == 'Infected':
                self.config.add_model_initial_configuration('Infected', params['Infected'])
        self.model.set_initial_status(self.config)

    def _infect_graph(self, infected_fraction):
        # iterations = self.model.iteration_bunch(1)
        # self.trends = self.model.build_trends(iterations)
        size = len(self.model.status)
        end_iterations = False

        while not end_iterations:
            iterations = self.model.iteration()
            # if iterations['iteration'] != 0 and iterations['iteration'] % 1 == 0:
            total_infected = sum(self.model.status.values())
            if total_infected / size >= infected_fraction:
                print(f'Inf. {total_infected / size}')
                end_iterations = True
        self.trends = self.model.build_trends([iterations])

    def _select_random_sources(self, n_sources):
        return random.sample(list(self.G.nodes()), n_sources)
