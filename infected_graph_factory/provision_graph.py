import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
import random

from utils.save_to_pickle import save_to_pickle

MODELS = {'SI': ep.SIModel,
          'SIR': ep.SIRModel,
          'IC': ep.IndependentCascadesModel}

SI_TRANSFORMATION = ['SIR', 'IC']


class InfectedGraphProvision:
    trends = None

    def __init__(self,
                 idx,
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

        if infection_config.model in SI_TRANSFORMATION:
            self._convert_removed_to_not_infected()

        save_to_pickle(self,
                       f'infected_graph/{graph_config.graph_type}_{int(100*graph_config.infection_config.max_infected_fraction)}inf_{graph_config.infection_config.n_sources}s',
                       f'{idx}-{graph_config.graph_type}{int(100*graph_config.infection_config.max_infected_fraction)}inf{graph_config.infection_config.n_sources}s-infected')

    def _add_edge_config(self, param_value, param_name):
        for e in self.G.edges():
            self.config.add_edge_configuration(param_name, e, param_value)

    def _add_model_params(self, params):
        for param_name, param_value in params.items():
            self.config.add_model_parameter(param_name, param_value)
            if param_name == 'threshold':
                self._add_edge_config(param_value, param_name)
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

    def _convert_removed_to_not_infected(self):
        status = {node: 0 if infection > 1 else infection for node, infection in self.model.status.items()}
        self.model.status = status

