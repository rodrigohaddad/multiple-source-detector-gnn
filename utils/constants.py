import torch

DEVICE = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

EMBEDDING_DIR = 'data/embedding'
INFECTED_DIR = 'data/infected_graph'

TRANSFORMED_DIR = 'data/graph_transformed'
NOT_TRANSFORMED_DIR = 'data/graph_not_transformed'
TRANSFORMED_TEST_DIR = 'data/graph_transformed/test'

MODEL_DIR = 'data/model/'
MODEL_FILE = 'sagemodel.pickle'
MODEL_WEIGHTLESS_FILE = 'sagemodel_weightless.pickle'

GLOBAL_MODEL_FILE = 'sagemodel_weightless.pickle'
