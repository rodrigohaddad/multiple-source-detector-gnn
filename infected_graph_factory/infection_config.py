class InfectionConfig:
    def __init__(self,
                 model,
                 n_iter,
                 name,
                 overwrite_previous,
                 file_path='',
                 params={}):
        self.model = model
        self.n_iter = n_iter
        self.params = params
        self.name = name
        self.overwrite_previous = overwrite_previous
        self.file_path = file_path

    def set_params(self, pr):
        self.params = {**self.params, **pr}
        return self
