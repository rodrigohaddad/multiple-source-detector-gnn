class InfectionConfig:
    def __init__(self, params={}, *initial_data, **kwargs):
        self.params = params
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def set_params(self, pr):
        self.params = {**self.params, **pr}
        return self
