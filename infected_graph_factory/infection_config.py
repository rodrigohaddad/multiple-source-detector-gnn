class InfectionConfig:
    def __init__(self, params={}, **kwargs):
        self.params = params
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def set_params(self, pr):
        self.params = {**self.params, **pr}


class GraphConfig:
    def __init__(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            if key != 'infection_config':
                setattr(self, key, kwargs[key])
        self.infection_config = InfectionConfig(**kwargs['infection_config'])

    def set_params(self, pr):
        self.infection_config.set_params(pr)
        return self
