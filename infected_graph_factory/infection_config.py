class InfectionConfig:
    def __init__(self,
                 model,
                 n_iter,
                 name,
                 file_path,
                 params={}):
        self.model = model
        self.n_iter = n_iter
        self.params = params
        self.name = name
        self.file_path = file_path

    def set_params(self, pr):
        self.params = pr
        return self
